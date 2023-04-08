from pyquantification.quantifiers.count import CountQuantifier
from pyquantification.quantifiers.pcc import PccQuantifier
from pyquantification.quantifiers.em import EmQuantifier
from pyquantification.quantifiers.gsls import (
    GslsQuantifier,
    TrueWeightGslsQuantifier,
)
from pyquantification.quantifiers.bootstrap import EmBootstrapQuantifier

QUANTIFIERS = {
    'count': CountQuantifier,
    'pcc': PccQuantifier,
    'em': EmQuantifier,
    'gsls': GslsQuantifier,
    'true-weight-gsls': TrueWeightGslsQuantifier,
    'em-bs': EmBootstrapQuantifier,
}
