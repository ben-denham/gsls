from .pcc import PccProbThresholdRejector, PccApproxProbThresholdRejector, PccMipRejector
from .em import EmProbThresholdRejector, EmApproxProbThresholdRejector, EmMipRejector
from .gsls import (GslsProbThresholdRejector, GslsApproxProbThresholdRejector, GslsMipRejector,
                   StaticGainGslsProbThresholdRejector, StaticGainGslsApproxProbThresholdRejector, StaticGainGslsMipRejector,
                   UniformGslsProbThresholdRejector, UniformGslsApproxProbThresholdRejector, UniformGslsMipRejector,
                   UniformStaticGainGslsProbThresholdRejector, UniformStaticGainGslsApproxProbThresholdRejector, UniformStaticGainGslsMipRejector)

REJECTORS = {
    'pcc-pt': PccProbThresholdRejector,
    'pcc-apt': PccApproxProbThresholdRejector,
    'pcc-mip': PccMipRejector,
    'em-pt': EmProbThresholdRejector,
    'em-apt': EmApproxProbThresholdRejector,
    'em-mip': EmMipRejector,
    'gsls-pt': GslsProbThresholdRejector,
    'gsls-apt': GslsApproxProbThresholdRejector,
    'gsls-mip': GslsMipRejector,
    'sggsls-pt': StaticGainGslsProbThresholdRejector,
    'sggsls-apt': StaticGainGslsApproxProbThresholdRejector,
    'sggsls-mip': StaticGainGslsMipRejector,
    'ugsls-pt': UniformGslsProbThresholdRejector,
    'ugsls-apt': UniformGslsApproxProbThresholdRejector,
    'ugsls-mip': UniformGslsMipRejector,
    'usggsls-pt': UniformStaticGainGslsProbThresholdRejector,
    'usggsls-apt': UniformStaticGainGslsApproxProbThresholdRejector,
    'usggsls-mip': UniformStaticGainGslsMipRejector,
}
