# coding: utf-8

from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from math import ceil
import os
import numpy as np
import pandas as pd
import requests
import re
from copy import deepcopy
import scipy.stats
from typing import cast, Any, Callable, Optional, Mapping, Dict, Set, Sequence, NamedTuple, Tuple
from joblib import dump, load
from zipfile import ZipFile
from pyreadr import read_r

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
DATASET_RANDOM_SEED = 1_000_000_000


# Types

class Dataset(ABC):

    def __init__(self,
                 df: pd.DataFrame, *,
                 calib_size: float,
                 numeric_features: Optional[Set[str]]) -> None:
        self.name = 'UNNAMED'
        self.df = df.reset_index()
        self.classes = sorted(self.df['class'].unique())
        self.calib_size = calib_size
        self.numeric_features = (set() if numeric_features is None
                                 else set(numeric_features))

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

    def subset(self, subset_df: pd.DataFrame) -> 'Dataset':
        # Don't use the constructor, otherwise we'd reset df.index.
        subdataset = deepcopy(self)
        subdataset.df = subset_df
        return subdataset

    def without_X(self):
        """Return a new copy of the dataset without any feature columns (to
        save space)."""
        without_X_df = self.df.drop(self.all_features, axis='columns')
        return self.subset(without_X_df)

    @property
    @abstractmethod
    def train_n(self):
        pass


Concepts = Sequence[str]
ClassPriors = Dict[str, float]


class Component(NamedTuple):
    concepts: Concepts
    weight: float
    class_priors: ClassPriors


Components = Dict[str, Component]


class ConceptsDataset(Dataset):
    """A Dataset that can be used to simulate shift between known concepts."""

    def __init__(self,
                 df: pd.DataFrame, *,
                 train_n: int,
                 test_n: int,
                 calib_size: float = 0.5,
                 numeric_features: Optional[Set[str]] = None,
                 skip_concept_check: bool = False) -> None:
        super().__init__(df, calib_size=calib_size, numeric_features=numeric_features)
        self._train_n = train_n
        self.test_n = test_n

        if not skip_concept_check:
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

    @property
    def train_n(self) -> int:
        return self._train_n

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


class SamplesDataset(Dataset):
    """A Dataset that is comprised of many samples with presumed dataset
    shift between samples."""

    def __init__(self,
                 df: pd.DataFrame, *,
                 train_samples: Set[str],
                 calib_size: float = 0.5,
                 numeric_features: Optional[Set[str]] = None) -> None:
        super().__init__(df, calib_size=calib_size, numeric_features=numeric_features)
        self.train_samples = train_samples

    @property
    def test_samples(self):
        if not hasattr(self, '_test_samples'):
            unique_samples = set(self.df['sample'].unique())
            self._test_samples = list(sorted(unique_samples - self.train_samples))
        return self._test_samples

    @property
    def train_n(self):
        return self.get_train_index().shape[0]

    def get_train_index(self) -> np.ndarray:
        return self.df[self.df['sample'].isin(self.train_samples)].index.to_numpy()

    def get_test_index(self, sample_idx: int) -> np.ndarray:
        return self.df[self.df['sample'] == self.test_samples[sample_idx]].index.to_numpy()

    def subset(self, subset_df: pd.DataFrame) -> 'SamplesDataset':
        new_dataset = cast(SamplesDataset, super().subset(subset_df))
        # Clear cached computed variable
        if hasattr(new_dataset, '_test_samples'):
            del new_dataset._test_samples
        return new_dataset


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
    try:
        r = requests.get(remote_url, verify=True)
    except Exception as ex:
        raise Exception(f'Failed to download: {remote_url}') from ex
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

def arabic_digits_dataset(train_n: int = 330, test_n: int = 110) -> ConceptsDataset:
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
    return ConceptsDataset(
        df=pd.concat([train_df, test_df]),
        train_n=train_n,
        test_n=test_n,
        numeric_features=set(feature_names),
    )


def insect_species_dataset(train_n: int = 1500, test_n: int = 500) -> ConceptsDataset:
    local_file = cache_path('insect_species.csv')
    cache_file_locally(local_file, 'https://raw.githubusercontent.com/denismr/Unsupervised-Context-Switch-For-Classification-Tasks/master/data/AedesQuinx.csv')
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={'species': 'class',
                            'temp_range': 'concept'})
    return ConceptsDataset(
        df=df,
        train_n=train_n,
        test_n=test_n,
        numeric_features=(set(df.columns) - {'class', 'concept'}),
    )


def insect_sex_dataset(train_n: int = 1500, test_n: int = 500) -> ConceptsDataset:
    local_file = cache_path('insect_sex.csv')
    cache_file_locally(local_file, 'https://raw.githubusercontent.com/denismr/Unsupervised-Context-Switch-For-Classification-Tasks/master/data/AedesSex.csv')
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={'sex': 'class',
                            'temp_range': 'concept'})
    return ConceptsDataset(
        df=df,
        train_n=train_n,
        test_n=test_n,
        numeric_features=(set(df.columns) - {'class', 'concept'}),
    )


def handwritten_letters_dataset(*,
                                letter_target: Optional[bool] = None,
                                author_target: Optional[bool] = None) -> ConceptsDataset:
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
    return ConceptsDataset(
        df=df,
        train_n=135,
        test_n=45,
        numeric_features=(set(df.columns) - {'class', 'concept'}),
    )


# DATASETS WITH SHIFT

def plankton_dataset(target_class: str = 'auto',
                     real_samples: bool = True) -> Dataset:
    """Based on dataset pre-processing of: https://github.com/pglez82/IFCB_quantification

    See: González, P., Castano, A., Peacock, E. E., Díez, J., Del Coz,
    J. J., & Sosik, H. M. (2019). Automatic plankton quantification
    using deep features. Journal of Plankton Research, 41(4), 449-463."""
    git_commit = 'f8feb19c72df391ea6972fdab09994b0f4b3cdab'
    plankton_dir = cache_path('plankton')
    if not os.path.isdir(plankton_dir):
        os.mkdir(plankton_dir)

    # Load set of samples considered to have complete data.
    local_samples_file = os.path.join(plankton_dir, 'FULLY_ANNOTATED.RData')
    cache_file_locally(local_samples_file, f'https://raw.githubusercontent.com/pglez82/IFCB_quantification/{git_commit}/FULLY_ANNOTATED.RData')
    complete_samples_series = read_r(local_samples_file)[None]['Sample']

    def load_classes_df():
        """Constructs a DataFrame of instance rows (identified by sample +
        roi) with associated auto_class values."""
        # Load mapping of "original class" to "auto class"
        local_classmap_file = os.path.join(plankton_dir, 'classmap.csv')
        cache_file_locally(local_classmap_file, f'https://raw.githubusercontent.com/pglez82/IFCB_quantification/{git_commit}/classes.csv')
        classmap_df = pd.read_csv(local_classmap_file)
        manual_to_auto_class = pd.Series(classmap_df['Auto class'].values, index=classmap_df['Current Manual Class']).to_dict()

        # See:
        # * https://darchive.mblwhoilibrary.org/handle/1912/7341
        # * https://github.com/hsosik/WHOI-Plankton
        annual_image_set_urls = {
            '2006': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7342/2006.zip?sequence=1&isAllowed=y',
            '2007': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7343/2007.zip?sequence=1&isAllowed=y',
            '2008': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7345/2008.zip?sequence=1&isAllowed=y',
            '2009': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7346/2009.zip?sequence=1&isAllowed=y',
            '2010': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7348/2010.zip?sequence=1&isAllowed=y',
            '2011': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7347/2011.zip?sequence=1&isAllowed=y',
            '2012': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7344/2012.zip?sequence=1&isAllowed=y',
            '2013': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7349/2013.zip?sequence=1&isAllowed=y',
            '2014': 'https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7350/2014.zip?sequence=1&isAllowed=y',
        }
        raw_dfs = []
        for year, url in annual_image_set_urls.items():
            local_year_file = os.path.join(plankton_dir, f'images_{year}.zip')
            cache_file_locally(local_year_file, url)
            with ZipFile(local_year_file) as zip_file:
                raw_dfs.append(pd.DataFrame([
                    filepath.split('/') for filepath in zip_file.namelist()
                    if filepath.endswith('.png')
                ], columns=['year', 'original_class', 'image']))
        df = pd.concat(raw_dfs)

        assert (df['image'].str.len() == 31).all()  # E.g. 'IFCB1_2006_270_170728_01626.png'
        df['sample'] = df['image'].str.slice(0, 21)  # E.g. 'IFCB1_2006_270_170728'
        df['roi_number'] = df['image'].str.slice(22, 27)  # E.g. '01626'
        del df['image']

        df['auto_class'] = df['original_class'].map(manual_to_auto_class)
        assert not df['auto_class'].isna().any()

        # Limit to complete samples.
        df = df[df['sample'].isin(complete_samples_series)]

        return df

    def load_full_df(classes_df):
        """Constructs a DataFrame with pre-computed features for each row
        (identified by sample + roi)."""
        # See: https://ifcb-data.whoi.edu/timeline
        sample_features_dir = os.path.join(plankton_dir, 'sample_features')
        if not os.path.isdir(sample_features_dir):
            os.mkdir(sample_features_dir)

        # Download features file for each sample.
        features_dfs = []
        for i, sample in enumerate(complete_samples_series, start=1):
            print(f'Processing sample {i}/{complete_samples_series.shape[0]} ({datetime.now()})')
            sample_features_filename = f'{sample}_features.csv'
            sample_features_file = os.path.join(sample_features_dir, sample_features_filename)
            cache_file_locally(sample_features_file, f'https://ifcb-data.whoi.edu/mvco/{sample_features_filename}')
            sample_features_df = pd.read_csv(sample_features_file)
            sample_features_df['sample'] = sample
            # Erasing this feature as it is present for some samples and not others.
            if 'summedBiovolume' in sample_features_df.columns:
                del sample_features_df['summedBiovolume']
            features_dfs.append(sample_features_df)
        features_df = pd.concat(features_dfs)

        # Merge classes with features, keeping any rows of features
        # that do not have a corresponding class.
        df = features_df.merge(classes_df, how='left', on=['sample', 'roi_number'])

        # Load table mapping low-level classes to high-level functional groups
        local_fg_file = os.path.join(plankton_dir, 'functional_groups.csv')
        cache_file_locally(local_fg_file, f'https://raw.githubusercontent.com/pglez82/IFCB_quantification/{git_commit}/functional_groups.csv')
        fg_df = pd.read_csv(local_fg_file)
        manual_to_fg_class = pd.Series(fg_df['GrupoFinal'].values, index=fg_df['Nombre Carpeta']).to_dict()
        df['fg_class'] = df['original_class'].map(manual_to_fg_class)

        return df

    full_df_file = os.path.join(plankton_dir, 'full_df.joblib')
    if not os.path.isfile(full_df_file):
        classes_df_file = os.path.join(plankton_dir, 'classes_df.csv')
        if not os.path.isfile(classes_df_file):
            classes_df = load_classes_df()
            classes_df.to_csv(classes_df_file, index=False)
        classes_df = pd.read_csv(classes_df_file)
        full_df = load_full_df(classes_df)
        dump(full_df, full_df_file)
    full_df = load(full_df_file)

    numeric_features = set([
        'Area_over_Perimeter',
        'Area_over_PerimeterSquared', 'BoundingBox_xwidth',
        'BoundingBox_ywidth', 'ConvexPerimeter', 'Eccentricity',
        'EquivDiameter', 'Extent', 'FeretDiameter', 'H180', 'H90',
        'H90_over_H180', 'H90_over_Hflip', 'HOG01', 'HOG02', 'HOG03',
        'HOG04', 'HOG05', 'HOG06', 'HOG07', 'HOG08', 'HOG09', 'HOG10',
        'HOG11', 'HOG12', 'HOG13', 'HOG14', 'HOG15', 'HOG16', 'HOG17',
        'HOG18', 'HOG19', 'HOG20', 'HOG21', 'HOG22', 'HOG23', 'HOG24',
        'HOG25', 'HOG26', 'HOG27', 'HOG28', 'HOG29', 'HOG30', 'HOG31',
        'HOG32', 'HOG33', 'HOG34', 'HOG35', 'HOG36', 'HOG37', 'HOG38',
        'HOG39', 'HOG40', 'HOG41', 'HOG42', 'HOG43', 'HOG44', 'HOG45',
        'HOG46', 'HOG47', 'HOG48', 'HOG49', 'HOG50', 'HOG51', 'HOG52',
        'HOG53', 'HOG54', 'HOG55', 'HOG56', 'HOG57', 'HOG58', 'HOG59',
        'HOG60', 'HOG61', 'HOG62', 'HOG63', 'HOG64', 'HOG65', 'HOG66',
        'HOG67', 'HOG68', 'HOG69', 'HOG70', 'HOG71', 'HOG72', 'HOG73',
        'HOG74', 'HOG75', 'HOG76', 'HOG77', 'HOG78', 'HOG79', 'HOG80',
        'HOG81', 'Hflip', 'Hflip_over_H180', 'Orientation',
        'RWcenter2total_powerratio', 'RWhalfpowerintegral', 'Ring01',
        'Ring02', 'Ring03', 'Ring04', 'Ring05', 'Ring06', 'Ring07',
        'Ring08', 'Ring09', 'Ring10', 'Ring11', 'Ring12', 'Ring13',
        'Ring14', 'Ring15', 'Ring16', 'Ring17', 'Ring18', 'Ring19',
        'Ring20', 'Ring21', 'Ring22', 'Ring23', 'Ring24', 'Ring25',
        'Ring26', 'Ring27', 'Ring28', 'Ring29', 'Ring30', 'Ring31',
        'Ring32', 'Ring33', 'Ring34', 'Ring35', 'Ring36', 'Ring37',
        'Ring38', 'Ring39', 'Ring40', 'Ring41', 'Ring42', 'Ring43',
        'Ring44', 'Ring45', 'Ring46', 'Ring47', 'Ring48', 'Ring49',
        'Ring50', 'RotatedArea', 'RotatedBoundingBox_xwidth',
        'RotatedBoundingBox_ywidth', 'Solidity', 'Wedge01', 'Wedge02',
        'Wedge03', 'Wedge04', 'Wedge05', 'Wedge06', 'Wedge07',
        'Wedge08', 'Wedge09', 'Wedge10', 'Wedge11', 'Wedge12',
        'Wedge13', 'Wedge14', 'Wedge15', 'Wedge16', 'Wedge17',
        'Wedge18', 'Wedge19', 'Wedge20', 'Wedge21', 'Wedge22',
        'Wedge23', 'Wedge24', 'Wedge25', 'Wedge26', 'Wedge27',
        'Wedge28', 'Wedge29', 'Wedge30', 'Wedge31', 'Wedge32',
        'Wedge33', 'Wedge34', 'Wedge35', 'Wedge36', 'Wedge37',
        'Wedge38', 'Wedge39', 'Wedge40', 'Wedge41', 'Wedge42',
        'Wedge43', 'Wedge44', 'Wedge45', 'Wedge46', 'Wedge47',
        'Wedge48', 'moment_invariant1', 'moment_invariant2',
        'moment_invariant3', 'moment_invariant4', 'moment_invariant5',
        'moment_invariant6', 'moment_invariant7', 'numBlobs',
        'rotated_BoundingBox_solidity', 'shapehist_kurtosis_normEqD',
        'shapehist_mean_normEqD', 'shapehist_median_normEqD',
        'shapehist_mode_normEqD', 'shapehist_skewness_normEqD',
        'summedArea', 'summedConvexArea', 'summedConvexPerimeter',
        'summedConvexPerimeter_over_Perimeter', 'summedFeretDiameter',
        'summedMajorAxisLength', 'summedMinorAxisLength',
        'summedPerimeter', 'texture_average_contrast',
        'texture_average_gray_level', 'texture_entropy',
        'texture_smoothness', 'texture_third_moment',
        'texture_uniformity',
    ])

    if target_class == 'auto':
        # If manually-labelled images were not present for a given
        # row, set it's class to 'na'
        full_df['class'] = full_df['auto_class'].fillna('na')
        # Remove classes that only appear once (not enough for train/calib split).
        full_df = full_df[~full_df['class'].isin(['Gonyaulax'])]
    elif target_class == 'functional_group':
        full_df['class'] = full_df['fg_class'].fillna('Other')
    elif target_class == 'binary':
        full_df['class'] = full_df['auto_class'].where(full_df['auto_class'] == 'mix', 'other')
    else:
        raise ValueError(f'Unrecognised target_class: {target_class}')

    # Replace feature nans with zero
    full_df.fillna(0, inplace=True)
    # Replace infinite values with the min/max values in each column.
    for feature in ['H90_over_Hflip']:
        non_inf_values = full_df[feature][~np.isinf(full_df[feature])]
        full_df[feature].replace(np.inf, non_inf_values.max(), inplace=True)
        full_df[feature].replace(-np.inf, non_inf_values.min(), inplace=True)
    assert np.isinf(full_df[numeric_features]).sum().sum() == 0

    if real_samples:
        return SamplesDataset(
            full_df,
            # First 200 samples are used for training.
            train_samples=set(complete_samples_series.iloc[:200]),
            numeric_features=numeric_features,
        )
    else:
        full_df['concept'] = pd.Series('na', index=full_df.index)
        return ConceptsDataset(
            full_df,
            train_n=10_000,
            test_n=1_000,
            numeric_features=numeric_features,
        )


class SyntheticComponent(NamedTuple):
    name: str
    weight: float
    class_prior: Sequence[float]
    # Pairs of loc/mean and scale/std-dev for Gaussian distributions
    class_conditional_dist_params: Sequence[Tuple[float, float]]

    @property
    def normalised_class_prior(self):
        return np.array(self.class_prior) / np.sum(self.class_prior)

    def get_x_probs(self, xs: pd.Series) -> pd.Series:
        """
        Return P(X) based on the class_conditional_dist_params P(X|Y):

        P(X) = SUM P(Y) * P(X|Y) FORALL Y
        """
        x_probs = pd.Series(0.0, index=xs.index)
        for class_weight, dist_params in zip(self.normalised_class_prior, self.class_conditional_dist_params):
            class_x_probs = scipy.stats.norm.pdf(xs, loc=dist_params[0], scale=dist_params[1])
            x_probs += class_weight * class_x_probs
        return x_probs

    def replace(self, **kwargs) -> 'SyntheticComponent':
        return self._replace(**kwargs)


def sample_synthetic_components(*, classes: Sequence[str], components: Sequence[SyntheticComponent], n: int, rng: np.random.RandomState) -> pd.DataFrame:
    component_class_dfs = []

    component_prior = np.array([component.weight for component in components])
    component_prior = component_prior / np.sum(component_prior)

    component_choices = rng.choice(len(components), size=n, replace=True, p=component_prior)
    component_counts = np.bincount(component_choices)
    for component, component_n in zip(components, component_counts):
        component_class_choices = rng.choice(len(component.normalised_class_prior),
                                             size=component_n, replace=True,
                                             p=component.normalised_class_prior)
        component_class_counts = np.bincount(component_class_choices)
        for class_label, dist_params, component_class_n in zip(classes, component.class_conditional_dist_params, component_class_counts):
            component_class_dfs.append(pd.DataFrame({
                'component': component.name,
                'class': class_label,
                'x': rng.normal(loc=dist_params[0], scale=dist_params[1], size=component_class_n),
            }))
    full_df = pd.concat(component_class_dfs)
    assert full_df.shape[0] == n
    full_df = cast(pd.DataFrame, full_df.sample(frac=1, replace=False, random_state=rng).reset_index(drop=True))
    return full_df


def synthetic_true_prob_dataset(
        *,
        classes: Sequence[str],
        source_components: Sequence[SyntheticComponent],
        target_components: Sequence[SyntheticComponent],
        sample_n: int = 100,
        train_samples: int = 10,
        test_samples: int = 1000,
) -> Dataset:
    for component in [*source_components, *target_components]:
        assert len(component.class_prior) == len(classes)
        assert len(component.class_conditional_dist_params) == len(classes)

    rng = np.random.RandomState(DATASET_RANDOM_SEED)
    source_df = sample_synthetic_components(classes=classes, components=source_components, n=(sample_n * train_samples), rng=rng)
    source_df['dist'] = 'source'
    target_df = sample_synthetic_components(classes=classes, components=target_components, n=(sample_n * test_samples), rng=rng)
    target_df['dist'] = 'target'

    dataset_df = pd.concat([source_df, target_df]).reset_index(drop=True)
    sample_series = pd.Series([f'sample_{i}' for i in range(train_samples + test_samples)]).repeat(sample_n)
    sample_series.index = dataset_df.index
    dataset_df['sample'] = sample_series

    # Compute the true source distribution probability for each class
    # on every x: P^S(Y|X)
    # P^S(Y|X) = SUM (P(C) * P^C(Y|X)) FORALL COMPONENTS C
    # P^C(Y|X) = P^C(X|Y) * P^C(Y) / P^C(X)
    # P^C(X) = SUM (P^C(Y) * P^C(X|Y)) FORALL CLASSES Y
    source_component_prior = np.array([component.weight for component in source_components])
    source_component_prior = source_component_prior / np.sum(source_component_prior)
    class_prob_columns = [f'source_prob__{class_label}' for class_label in classes]
    for class_idx, class_prob_column in enumerate(class_prob_columns):
        dataset_df[class_prob_column] = pd.Series(0.0, index=dataset_df.index)
        for component, component_weight in zip(source_components, source_component_prior):
            dist_params = component.class_conditional_dist_params[class_idx]
            # component_class_conditional_prob = P^C(X|Y)
            component_class_conditional_probs = scipy.stats.norm.pdf(dataset_df['x'], loc=dist_params[0], scale=dist_params[1])
            # component_y_prob = P^C(Y)
            component_class_prob = component.normalised_class_prior[class_idx]
            # component_x_probs = P^C(X)
            component_x_probs = component.get_x_probs(dataset_df['x'])
            # component_probs = P^C(Y|X)
            component_probs = component_class_conditional_probs * component_class_prob / component_x_probs
            # component_weight = P(C)
            dataset_df[class_prob_column] += component_weight * component_probs

    # Check sum of true prob columns is approximately 1 for each row.
    assert ((dataset_df[class_prob_columns].sum(axis=1) - 1).abs() < (10**(-7))).all()

    return SamplesDataset(
        dataset_df,
        # First samples are the source samples used for training.
        train_samples=set([f'sample_{i}' for i in range(train_samples)]),
        numeric_features=set(['x']),
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


unshifted_component = SyntheticComponent(
    name='unshifted',
    weight=1,
    class_prior=[0.5, 0.5],
    class_conditional_dist_params=[
        (0.2, 0.3),
        (0.8, 0.3),
    ],
)
prior_shifted_component = unshifted_component.replace(
    name='prior_shifted',
    class_prior=[0.25, 0.75],
)
loss_component = unshifted_component.replace(
    name='loss',
    weight=0.5,
    class_prior=[1.0, 0.0],
    class_conditional_dist_params=[
        (-1.0, 0.3),
        (0.8, 0.3),
    ],
)
gain_component = unshifted_component.replace(
    name='gain',
    weight=0.5,
    class_prior=[0.0, 1.0],
    class_conditional_dist_params=[
        (0.2, 0.3),
        (2.0, 0.3),
    ],
)


DATASETS = named_datasets({
    'arabic-digits': arabic_digits_dataset,
    'insect-species': insect_species_dataset,
    'insect-sex': insect_sex_dataset,
    'insect-sex_smaller': partial(insect_sex_dataset, test_n=250),
    'insect-sex_smallest': partial(insect_sex_dataset, test_n=50),
    'handwritten-letters-letter': partial(handwritten_letters_dataset, letter_target=True),
    'handwritten-letters-author': partial(handwritten_letters_dataset, author_target=True),
    'plankton': partial(plankton_dataset, target_class='auto'),
    'binary-plankton': partial(plankton_dataset, target_class='binary'),
    'fg-plankton': partial(plankton_dataset, target_class='functional_group'),
    'synthetic-true-prob-no-shift': partial(
        synthetic_true_prob_dataset,
        classes=['a', 'b'],
        source_components=[unshifted_component],
        target_components=[unshifted_component],
    ),
    'synthetic-true-prob-prior-shift': partial(
        synthetic_true_prob_dataset,
        classes=['a', 'b'],
        source_components=[unshifted_component],
        target_components=[prior_shifted_component],
    ),
    'synthetic-true-prob-gsls-shift': partial(
        synthetic_true_prob_dataset,
        classes=['a', 'b'],
        source_components=[unshifted_component, loss_component],
        target_components=[unshifted_component, gain_component],
    ),
})
