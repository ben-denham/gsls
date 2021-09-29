import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
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


CLASSIFIERS = {
    'logreg': logreg_classifier,
}
