from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Callable, cast

from pyquantification.shift_tests.base import (
    Priors, Probs
)


# HIST BINS

class HistGrid(ABC):
    """Base-class for different strategies to determine the histogram grids."""

    @abstractmethod
    def get_bin_edges(self, calib_probs: Probs, target_probs: Probs) -> List[np.ndarray]:
        """Return a list of bin_edge arrays to use for histograms of the given
        calib_probs and target_probs for each class."""
        pass


class NullHistGrid(HistGrid):

    def get_bin_edges(self, calib_probs: Probs, target_probs: Probs) -> List[np.ndarray]:
        raise NotImplementedError()


class EqualSpaceHistGrid(HistGrid):
    """Constructs histograms with a fixed number of equal space/width bins."""

    def __init__(self, bin_count: int):
        self.bin_count = bin_count

    def get_bin_edges(self, calib_probs: Probs, target_probs: Probs) -> List[np.ndarray]:
        return [np.linspace(0, 1, (self.bin_count + 1)) for _ in range(calib_probs.shape[1])]


class EqualProbHistGrid(HistGrid):
    """Constructs histograms with a rule-of-thumb based number of bins
    with an equal number of calib_probs in each bin."""

    def get_bin_edges(self, calib_probs: Probs, target_probs: Probs) -> List[np.ndarray]:
        # Set hist_bins according to rule-of-thumb for equiprobable
        # bins (based on minimum count of instances for either
        # distribution sample):
        # https://itl.nist.gov/div898/handbook/prc/section2/prc211.htm
        bin_count = round(2 * (min(calib_probs.shape[0], target_probs.shape[0]) ** (2/5)))
        bin_edges = []
        for class_idx in range(calib_probs.shape[1]):
            # To handle cases with too few unique probs to achieve the
            # desired bin_count, we set duplicates='drop', and allow
            # Hists of different classes to have different lengths.
            _, class_bin_edges = pd.qcut(calib_probs[:, class_idx], bin_count, retbins=True, duplicates='drop')
            # Ensure all class_bin_edges are in the range [0, 1], and
            # that the first and last bin edges are 0 and 1
            # respectively. When used with np.histogram, this final
            # edge will include records where prob = 1.
            class_bin_edges = np.clip(class_bin_edges, 0, 1)
            class_bin_edges[0] = 0
            class_bin_edges[-1] = 1
            bin_edges.append(class_bin_edges)
        return bin_edges


# HISTOGRAMS

@dataclass
class Hist:
    pdf: np.ndarray

    @property
    def cdf(self):
        if not hasattr(self, '_cdf'):
            self._cdf = self.pdf.cumsum()
        return self._cdf


def get_hists(probs: Probs, *, bin_edges: List[np.ndarray]) -> List[Hist]:
    """For a (instances, classes) probs array, return a list containing a
    Hist for each class column in probs. Each entry in bin_edges
    contains the edges for a specific class, and each class may have a
    different number of bins."""
    hists = []
    for class_i in range(probs.shape[1]):
        bin_counts, _ = np.histogram(probs[:, class_i], bins=bin_edges[class_i])
        pdf = bin_counts / probs.shape[0]
        hists.append(Hist(pdf))
    return hists


def mix_per_class_hists(*, class_priors: Priors, per_class_hists: List[List[Hist]]) -> List[Hist]:
    """Return a mixture of the given class-conditional class-prob Hists weighted
    according to the given class priors, returning a list of class-prob Hists."""
    assert class_priors.shape == (len(per_class_hists),)
    class_hists = []
    for class_idx in range(class_priors.shape[0]):
        class_hist = Hist(cast(np.ndarray, sum([
            hists[class_idx].pdf * class_prior
            for class_prior, hists in zip(class_priors, per_class_hists)
        ])))
        class_hists.append(class_hist)
    return class_hists


# HIST DISTANCES

DistanceMeasure = Callable[[Hist, Hist], float]


def null_distance(hist_a: Hist, hist_b: Hist) -> float:
    raise NotImplementedError()


def ks_distance(hist_a: Hist, hist_b: Hist) -> float:
    # KS-distance is the maximum difference between cdfs
    return np.max(np.abs(hist_a.cdf - hist_b.cdf))


def hellinger_distance(hist_a: Hist, hist_b: Hist) -> float:
    return (1 / np.sqrt(2)) * np.sqrt(np.sum(np.square(np.sqrt(hist_a.pdf) - np.sqrt(hist_b.pdf))))


DistanceAggregator = Callable[[List[float]], float]


def mean_distance_aggregator(class_distances: List[float]) -> float:
    return float(np.mean(class_distances))


def max_distance_aggregator(class_distances: List[float]) -> float:
    return float(np.max(class_distances))


# MIXTURE DISTRIBUTIONS

class ClassConditionalMixture(ABC):
    """Representation of a set of class-conditional (potentially
    multivariate) distributions of classifier probs which can be mixed
    with different class priors."""

    def __init__(self, *, per_class_probs: List[Probs]):
        self.per_class_probs = per_class_probs

    @abstractmethod
    def sample_probs(self, *,
                     class_priors: Priors,
                     target_size: int,
                     rng: np.random.RandomState) -> Probs:
        """Return a sample of probabilities of *approximately* target_size
        sampled from the class-conditional distributions mixed
        according to the given class_priors. Sampling is performed
        such that the given class_priors should be the true priors of
        the sample, not the priors that are sampled from."""
        pass


class SampledClassConditionalMixture(ClassConditionalMixture):
    """Samples mixture probabilities directly from the given sets of
    probabilities, without any modelling of distributions of probabilities."""

    def sample_probs(self, *,
                     class_priors: Priors,
                     target_size: int,
                     rng: np.random.RandomState) -> Probs:
        assert class_priors.shape[0] == len(self.per_class_probs)
        sim_per_class_probs = [
            class_probs[rng.choice(
                a=np.arange(class_probs.shape[0]),
                size=int(round(target_size * class_prior)),
                replace=True,
            )]
            for class_probs, class_prior
            in zip(self.per_class_probs, class_priors)
        ]
        return np.vstack(sim_per_class_probs)
