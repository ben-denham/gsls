from pyquantification.shift_tests.lr import LrShiftTester
from pyquantification.shift_tests.ks import KsShiftTester
from pyquantification.shift_tests.sim import (
    KsWpaShiftTester, MaxKsWpaShiftTester,
    HdWpaShiftTester, DynHdWpaShiftTester, MaxDynHdWpaShiftTester,
    KsCdtShiftTester, MaxKsCdtShiftTester,
    HdCdtShiftTester, DynHdCdtShiftTester, MaxDynHdCdtShiftTester,
)
from pyquantification.shift_tests.aks import AksShiftTester

SHIFT_TESTERS = {
    'lr': LrShiftTester,
    'ks': KsShiftTester,
    'wpa-ks': KsWpaShiftTester,
    'wpa-xks': MaxKsWpaShiftTester,
    'wpa-hd': HdWpaShiftTester,
    'wpa-dhd': DynHdWpaShiftTester,
    'wpa-xdhd': MaxDynHdWpaShiftTester,
    'cdt-ks': KsCdtShiftTester,
    'cdt-xks': MaxKsCdtShiftTester,
    'cdt-hd': HdCdtShiftTester,
    'cdt-dhd': DynHdCdtShiftTester,
    'cdt-xdhd': MaxDynHdCdtShiftTester,
    'aks': AksShiftTester,
}
