import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import _fit_calibrator


class PrefitCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Simplification of sklearn.calibration.CalibratedClassifierCV that:

    1. Only supports a base_estimator that has been prefit on a
       different dataset to the X/y passed to fit().
    2. Does not perform unnecessary checks on X/y that exist in
       CalibratedClassifierCV and prevent non-numeric data
       being passed (which is required if the base_estimator
       accepts pd.DataFrames).

    """

    def __init__(self, base_estimator=None, method='sigmoid'):
        self.base_estimator = base_estimator
        self.method = method

    def fit(self, X, y, y_pred):
        self.classes_ = self.base_estimator.classes_
        # Reduce binary predictions to a single column.
        if len(self.classes_) == 2:
            y_pred = y_pred[:, 1:]
        self.calibrated_classifier = _fit_calibrator(
            clf=self.base_estimator,
            predictions=y_pred,
            y=y,
            classes=self.classes_,
            method=self.method,
        )
        return self

    def predict_proba(self, X):
        return self.calibrated_classifier.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
