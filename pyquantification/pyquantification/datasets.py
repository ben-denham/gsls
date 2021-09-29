from datetime import datetime
from functools import partial
from math import ceil
import os
import numpy as np
import pandas as pd
import requests
import re
from copy import deepcopy
from typing import cast, Any, Callable, Optional, Mapping, Dict, Set, Sequence, NamedTuple
from joblib import dump, load

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
DATASET_RANDOM_SEED = 1_000_000_000


# Types

Concepts = Sequence[str]
ClassPriors = Dict[str, float]


class Component(NamedTuple):
    concepts: Concepts
    weight: float
    class_priors: ClassPriors


Components = Dict[str, Component]


class Dataset():

    def __init__(self,
                 df: pd.DataFrame, *,
                 train_n: int,
                 test_n: int,
                 calib_size: float = 0.5,
                 numeric_features: Optional[Set[str]] = None) -> None:
        self.name = 'UNNAMED'
        self.df = df.reset_index()
        self.classes = sorted(self.df['class'].unique())
        self.train_n = train_n
        self.test_n = test_n
        self.calib_size = calib_size
        self.numeric_features = (set() if numeric_features is None
                                 else set(numeric_features))

        # To support our sampling approach, there should be enough
        # instances for each concept+y_class pair to fill the entire
        # train and test sets.
        full_sample_size = self.train_n + self.test_n
        class_concept_counts = self.df.groupby(['concept', 'class']).count()['index']
        for concept in self.df['concept']:
            for y_class in self.classes:
                subset_count = class_concept_counts[concept, y_class]
                if subset_count < full_sample_size:
                    raise ValueError(f'Only {subset_count} instances available for '
                                     f'class "{y_class}" in concept "{concept}", while the '
                                     f'maximum number that may be sampled is the full '
                                     f'train and test size: {full_sample_size}')

    def set_name(self, name: str) -> None:
        self.name = name

    @property
    def all_features(self) -> Set[str]:
        return cast(Set[str], set()).union(
            self.numeric_features,
        )

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.all_features]

    @property
    def y(self) -> pd.Series:
        return self.df['class']

    @property
    def indexes(self) -> np.ndarray:
        return self.df.index.to_numpy()

    @property
    def concepts(self) -> Concepts:
        return sorted(self.df['concept'].unique())

    @property
    def calib_n(self) -> int:
        # Uses ceil for consistency with
        # sklearn.model_selection.train_test_split()
        return ceil(self.train_n * self.calib_size)

    def components_index(self, components: Components) -> np.ndarray:
        """Returns a numpy array with an entry for each dataset row,
        containing the component name corresponding to that row's concept."""
        concept_to_component = {concept: component_name
                                for component_name, component in components.items()
                                for concept in component.concepts}
        return self.df['concept'].map(concept_to_component).to_numpy()

    def subset(self, subset_df: pd.DataFrame) -> 'Dataset':
        # Don't use the constructor, otherwise we'd reset df.index.
        subdataset = deepcopy(self)
        subdataset.df = subset_df
        return subdataset


# Downloading / Caching

def uci_url(path: str) -> str:
    return 'https://archive.ics.uci.edu/ml/machine-learning-databases/{}'.format(path)


def cache_path(filename: str) -> str:
    return os.path.join('../data', filename)


def cache_file_locally(local_path: str, remote_url: str) -> None:
    """If local_path does not exist, Downloads the remote_url and saves it
    to local_url."""
    if os.path.isfile(local_path):
        return
    r = requests.get(remote_url, verify=True)
    with open(local_path, 'wb') as f:
        f.write(r.content)


def cache_and_unzip(local_compressed_path: str, local_target_dir_path: str,
                    remote_url: str, targz: bool = False) -> None:
    """If local_compressed_path does not exist, downloads the remote_url,
    and ensures it is uncompressed inside local_target_dir_path."""
    cache_file_locally(local_compressed_path, remote_url)
    # Decompress
    if not os.path.isdir(local_target_dir_path):
        os.mkdir(local_target_dir_path)
        unzip_command = 'tar -xvzf' if targz else 'unzip'
        command = (f'cd {local_target_dir_path} && '
                   f'{unzip_command} ../{os.path.basename(local_compressed_path)}')
        os.system(command)


def cache_computation(local_path: str, func: Callable) -> Any:
    """Returns the result of calling func, with caching into local_path."""
    if not os.path.isfile(local_path):
        dump({local_path: func(), 'timestamp': datetime.now()}, local_path)
    return load(local_path)[local_path]


# DATASETS

# Each dataset function:
# * Makes use of caching to prevent redownloading each dataset.
# * Returns a Dataset containing a DataFrame and metadata.
# * Stores the target feature for classification in a 'class' column.

def arabic_digits_dataset(train_n: int = 330, test_n: int = 110) -> Dataset:
    # NOTE: It appears that this dataset contains 107 duplicate
    # rows. These have been left in to maintain consistency with other
    # usage of the dataset (https://doi.org/10.1145/3219819.3220059).

    # We will have 13 mean MFCC values for each instance.
    feature_names = [f'f{i}' for i in range(13)]

    def read_blocks(filepath: str) -> Sequence[str]:
        with open(filepath, 'r') as f:
            text = f.read()
            # Each block is separated by a line containing only spaces.
            text = re.sub('[ ]+', ' ', text)
            blocks = text.split('\n \n')
            return blocks

    def block_to_features(block: str) -> np.ndarray:
        # Each line is 13 space-separated mfcc values.
        lines = block.strip().splitlines()
        mfccs = np.array([
            [float(value) for value in line.split(' ')]
            for line in lines
        ])
        # We take the mean MFCC values over each block (as performed
        # in: https://doi.org/10.1145/3219819.3220059)
        return np.mean(mfccs, axis=0).reshape((mfccs.shape[1],))

    def file_to_df(filepath: str, chunk_size: int) -> pd.DataFrame:
        feature_rows = [block_to_features(block) for block in read_blocks(filepath)]
        df = pd.DataFrame(feature_rows, columns=feature_names)
        assert df.shape[1] == 13
        df['concept'] = None
        df['class'] = None
        # Chunks: 1=male,0; 2=female,0; 3=male,1; ...
        assert df.shape[0] % chunk_size == 0
        chunk_counter = 0
        for digit in range(10):
            for sex in ['male', 'female']:
                chunk_start = chunk_counter * chunk_size
                chunk_end = chunk_start + chunk_size
                # Sex is the class, digit is the concept
                df.loc[chunk_start:chunk_end, ['class', 'concept']] = sex, f'd{digit}'
                chunk_counter += 1
        assert df.shape[0] / chunk_counter == chunk_size
        return df

    train_local_file = cache_path('Train_Arabic_Digit.txt')
    cache_file_locally(train_local_file, uci_url('00195/Train_Arabic_Digit.txt'))
    test_local_file = cache_path('Test_Arabic_Digit.txt')
    cache_file_locally(test_local_file, uci_url('00195/Test_Arabic_Digit.txt'))

    train_df = file_to_df(train_local_file, chunk_size=330)
    test_df = file_to_df(test_local_file, chunk_size=110)
    return Dataset(
        df=pd.concat([train_df, test_df]),
        train_n=train_n,
        test_n=test_n,
        numeric_features=set(feature_names),
    )


def insect_species_dataset(train_n: int = 1500, test_n: int = 500) -> Dataset:
    local_file = cache_path('insect_species.csv')
    cache_file_locally(local_file, 'https://raw.githubusercontent.com/denismr/Unsupervised-Context-Switch-For-Classification-Tasks/master/data/AedesQuinx.csv')
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={'species': 'class',
                            'temp_range': 'concept'})
    return Dataset(
        df=df,
        train_n=train_n,
        test_n=test_n,
        numeric_features=(set(df.columns) - {'class', 'concept'}),
    )


def insect_sex_dataset(train_n: int = 1500, test_n: int = 500) -> Dataset:
    local_file = cache_path('insect_sex.csv')
    cache_file_locally(local_file, 'https://raw.githubusercontent.com/denismr/Unsupervised-Context-Switch-For-Classification-Tasks/master/data/AedesSex.csv')
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={'sex': 'class',
                            'temp_range': 'concept'})
    return Dataset(
        df=df,
        train_n=train_n,
        test_n=test_n,
        numeric_features=(set(df.columns) - {'class', 'concept'}),
    )


def handwritten_letters_dataset(*,
                                letter_target: Optional[bool] = None,
                                author_target: Optional[bool] = None) -> Dataset:
    if letter_target:
        concept, target = 'author', 'letter'
    elif author_target:
        concept, target = 'letter', 'author'
    else:
        raise ValueError('Must select a target.')

    local_file = cache_path('handwritten_letters.csv')
    cache_file_locally(local_file, 'https://raw.githubusercontent.com/denismr/Unsupervised-Context-Switch-For-Classification-Tasks/master/data/Handwritten.csv')
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={target: 'class',
                            concept: 'concept'})
    return Dataset(
        df=df,
        train_n=135,
        test_n=45,
        numeric_features=(set(df.columns) - {'class', 'concept'}),
    )


def named_datasets(datasets: Mapping[str, Callable[[], Dataset]]) -> Mapping[str, Callable[[], Dataset]]:
    """Given a dict of dataset names to dataset building functions, return
    an analogous dict where the dataset building function will set the
    dataset's name after construction."""

    def named_dataset_decorator(
            name: str,
            dataset_func: Callable[[], Dataset]) -> Callable[[], Dataset]:

        def named_dataset_func() -> Dataset:
            dataset = dataset_func()
            dataset.set_name(name)
            return dataset

        return named_dataset_func

    return {
        name: named_dataset_decorator(name, dataset_func)
        for name, dataset_func in datasets.items()
    }


DATASETS = named_datasets({
    'arabic-digits': arabic_digits_dataset,
    'insect-species': insect_species_dataset,
    'insect-sex': insect_sex_dataset,
    'insect-sex_smaller': partial(insect_sex_dataset, test_n=250),
    'insect-sex_smallest': partial(insect_sex_dataset, test_n=50),
    'handwritten-letters-letter': partial(handwritten_letters_dataset, letter_target=True),
    'handwritten-letters-author': partial(handwritten_letters_dataset, author_target=True),
})
