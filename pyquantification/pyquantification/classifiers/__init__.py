import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from typing import cast, Set

from pyquantification.classifiers.transformers import (
    SelectFeatureSubsetDataFrameTransformer,
    TransformFeatureSubsetDataFrameTransformer,
    DataFrameStandardNormaliser,
    NumpyTransformer,
)
from pyquantification.datasets import Dataset

MAX_ITERATIONS = 10_000
# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
CLASSIFIER_RANDOM_SEED = 1_000_000


class CustomXGBClassifier(XGBClassifier):

    def fit(self, X, y):
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)
        super().fit(X, y)
        self.classes_ = self.le.classes_

    def predict(self, X):
        return self.le.inverse_transform(super().predict(X))


def logreg_classifier(dataset: Dataset) -> ClassifierMixin:
    steps = [
        ('select_features', SelectFeatureSubsetDataFrameTransformer(
            list(cast(Set[str], set()).union(
                dataset.numeric_features,
            ))
        )),
        ('numeric', TransformFeatureSubsetDataFrameTransformer(
            list(dataset.numeric_features),
            Pipeline([
                ('normalise', DataFrameStandardNormaliser()),
            ]),
        )),
        ('matrix', NumpyTransformer(np.float64)),
        ('classifier', LogisticRegression(
            # 'auto' uses multinomial logreg when multi-class, but
            # still uses a single logreg for binary-class datasets.
            multi_class='auto',
            solver='lbfgs',
            max_iter=MAX_ITERATIONS,
            # As lbfgs solver is used, random_state should not make a
            # difference, but we set it statically to be sure.
            random_state=CLASSIFIER_RANDOM_SEED,
        ))
    ]
    model = Pipeline(steps)
    return model


def xgboost_classifier(dataset: Dataset) -> ClassifierMixin:
    model = logreg_classifier(dataset)
    model.set_params(classifier=CustomXGBClassifier(
        random_state=CLASSIFIER_RANDOM_SEED,
    ))
    return model


class SourceProbClassifier(ClassifierMixin):
    """Uses pre-calculated source probabilities from the dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.classes_ = np.array(dataset.classes)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob_columns = [f'source_prob__{class_label}' for class_label in self.classes_]
        return X[prob_columns].to_numpy()


CLASSIFIERS = {
    'logreg': logreg_classifier,
    'xgboost': xgboost_classifier,
    'source-prob': SourceProbClassifier,
}
