# -*- coding: utf-8 -*-

import logging
from typing import Callable, Optional

import numpy as np
import torch as th

from wxbtool.data.variables import code2var, codes


mean_t2m = 278.5193277994792
std_t2m = 21.219592501509624
def norm_t2m(t2m):
    return (t2m - 278.5193277994792) / 21.219592501509624


mean_tcc = 0.6740332964139107
std_tcc = 0.3626919709448507
def norm_tcc(tcc):
    return (tcc - 0.6740332964139107) / 0.3626919709448507


mean_log_tp = 0.07109370560218127
std_log_tp = 0.1847837422860926
def norm_tp(tp):
    if type(tp) is th.Tensor:
        tp = th.log(0.001 + tp) - np.log(0.001)
    else:
        tp = np.log(0.001 + tp) - np.log(0.001)
    return (tp - 0.07109370560218127) / 0.1847837422860926


mean_tisr = 1074511.0673076923
std_tisr = 1439848.7984975462
def norm_tisr(tisr):
    return (tisr - 1074511.0673076923) / 1439848.7984975462


mean_t50 = 212.440180069361
std_t50 = 10.260532486959207
def norm_t50(t50):
    return (t50 - 212.440180069361) / 10.260532486959207


mean_t250 = 222.76936692457932
std_t250 = 8.532687741643816
def norm_t250(t250):
    return (t250 - 222.76936692457932) / 8.532687741643816


mean_t500 = 252.95991789988983
std_t500 = 13.062728754466221
def norm_t500(t500):
    return (t500 - 252.95991789988983) / 13.062728754466221


mean_t600 = 261.14625666691705
std_t600 = 13.418785261563986
def norm_t600(t600):
    return (t600 - 261.14625666691705) / 13.418785261563986


mean_t700 = 267.4045856182392
std_t700 = 14.767785857645688
def norm_t700(t700):
    return (t700 - 267.4045856182392) / 14.767785857645688


mean_t850 = 274.57741292317706
std_t850 = 15.572175300640202
def norm_t850(t850):
    return (t850 - 274.57741292317706) / 15.572175300640202


mean_t925 = 277.3668893667367
std_t925 = 16.067193457642112
def norm_t925(t925):
    return (t925 - 277.3668893667367) / 16.067193457642112


mean_z50 = 199363.18309294872
std_z50 = 5882.393912540581
def norm_z50(z50):
    return (z50 - 199363.18309294872) / 5882.393912540581


mean_z250 = 101226.7467948718
std_z250 = 5537.287144094636
def norm_z250(z250):
    return (z250 - 101226.7467948718) / 5537.287144094636


mean_z500 = 54117.3952323718
std_z500 = 3353.5545664452306
def norm_z500(z500):
    return (z500 - 54117.3952323718) / 3353.5545664452306


mean_z600 = 40649.71213942308
std_z600 = 2696.302177194697
def norm_z600(z600):
    return (z600 - 40649.71213942308) / 2696.302177194697


mean_z700 = 28929.418920272437
std_z700 = 2137.0576819215225
def norm_z700(z700):
    return (z700 - 28929.418920272437) / 2137.0576819215225


mean_z850 = 13749.576822916666
std_z850 = 1471.5438146105798
def norm_z850(z850):
    return (z850 - 13749.576822916666) / 1471.5438146105798


mean_z925 = 7014.495780749198
std_z925 = 1230.0568519758604
def norm_z925(z925):
    return (z925 - 7014.495780749198) / 1230.0568519758604


mean_z1000 = 736.8600307366787
std_z1000 = 1072.7004633440063
def norm_z1000(z1000):
    return (z1000 - 736.8600307366787) / 1072.7004633440063


mean_tau = 6048.8221254006414
std_tau = 3096.4446045099244
def norm_tau(tau):
    return (tau - 6048.8221254006414) / 3096.4446045099244


mean_u50 = 5.651555599310459
std_u50 = 15.284072111757201
def norm_u50(u50):
    return (u50 - 5.651555599310459) / 15.284072111757201


mean_u250 = 13.338717974149263
std_u250 = 17.9696984120105
def norm_u250(u250):
    return (u250 - 13.338717974149263) / 17.9696984120105


mean_u500 = 6.552764354607998
std_u500 = 11.987184423423065
def norm_u500(u500):
    return (u500 - 6.552764354607998) / 11.987184423423065


mean_u600 = 4.797355407323593
std_u600 = 10.340552477523497
def norm_u600(u600):
    return (u600 - 4.797355407323593) / 10.340552477523497


mean_u700 = 3.298975400435619
std_u700 = 9.206544731461376
def norm_u700(u700):
    return (u700 - 3.298975400435619) / 9.206544731461376


mean_u850 = 1.3959463712496636
std_u850 = 8.192228835263744
def norm_u850(u850):
    return (u850 - 1.3959463712496636) / 8.192228835263744


mean_u925 = 0.5791668066611657
std_u925 = 7.954505065668797
def norm_u925(u925):
    return (u925 - 0.5791668066611657) / 7.954505065668797


mean_v50 = 0.004644189133847622
std_v50 = 7.067888921073515
def norm_v50(v50):
    return (v50 - 0.004644189133847622) / 7.067888921073515


mean_v250 = -0.030615646532402396
std_v250 = 13.388158186264183
def norm_v250(v250):
    return (v250 + 0.030615646532402396) / 13.388158186264183


mean_v500 = -0.02327536712758816
std_v500 = 9.186385511519138
def norm_v500(v500):
    return (v500 + 0.02327536712758816) / 9.186385511519138


mean_v600 = -0.030413604986209136
std_v600 = 7.805535575721749
def norm_v600(v600):
    return (v600 + 0.030413604986209136) / 7.805535575721749


mean_v700 = 0.04160793335774006
std_v700 = 6.894049310040708
def norm_v700(v700):
    return (v700 - 0.04160793335774006) / 6.894049310040708


mean_v850 = 0.16874054494576576
std_v850 = 6.288698845750149
def norm_v850(v850):
    return (v850 - 0.16874054494576576) / 6.288698845750149


mean_v925 = 0.23735194137463203
std_v925 = 6.490122512802569
def norm_v925(v925):
    return (v925 - 0.23735194137463203) / 6.490122512802569


mean_q50 = 2.665544594166045e-06
std_q50 = 3.6121240315989756e-07
def norm_q50(q50):
    return (q50 - 2.665544594166045e-06) / 3.6121240315989756e-07


mean_q250 = 5.782029126212598e-05
std_q250 = 7.4480380199925e-05
def norm_q250(q250):
    return (q250 - 5.782029126212598e-05) / 7.4480380199925e-05


mean_q500 = 0.0008543887763666228
std_q500 = 0.001079534297474708
def norm_q500(q500):
    return (q500 - 0.0008543887763666228) / 0.001079534297474708


mean_q600 = 0.0015437401389368833
std_q600 = 0.0017701706674727745
def norm_q600(q600):
    return (q600 - 0.0015437401389368833) / 0.0017701706674727745


mean_q700 = 0.002432438085237757
std_q700 = 0.002546475376073099
def norm_q700(q700):
    return (q700 - 0.002432438085237757) / 0.002546475376073099


mean_q850 = 0.004572244002841986
std_q850 = 0.004106876858978989
def norm_q850(q850):
    return (q850 - 0.004572244002841986) / 0.004106876858978989


mean_q925 = 0.006030511206541306
std_q925 = 0.005071411533793075
def norm_q925(q925):
    return (q925 - 0.006030511206541306) / 0.005071411533793075


def denorm_t2m(t2m):
    return t2m * 21.219592501509624 + 278.5193277994792


def denorm_tcc(tcc):
    return tcc * 0.3626919709448507 + 0.6740332964139107


def denorm_tp(tp):
    if type(tp) is th.Tensor:
        tp = (
            th.exp((tp * 0.1847837422860926 + 0.07109370560218127) + np.log(0.001))
            - 0.001
        )
    else:
        tp = (
            np.exp((tp * 0.1847837422860926 + 0.07109370560218127) + np.log(0.001))
            - 0.001
        )
    return tp


def denorm_tisr(tisr):
    return tisr * 1439848.7984975462 + 1074511.0673076923


def denorm_t50(t50):
    return t50 * 10.260532486959207 + 212.440180069361


def denorm_t250(t250):
    return t250 * 8.532687741643816 + 222.76936692457932


def denorm_t500(t500):
    return t500 * 13.062728754466221 + 252.95991789988983


def denorm_t600(t600):
    return t600 * 13.418785261563986 + 261.14625666691705


def denorm_t700(t700):
    return t700 * 14.767785857645688 + 267.4045856182392


def denorm_t850(t850):
    return t850 * 15.572175300640202 + 274.57741292317706


def denorm_t925(t925):
    return t925 * 16.067193457642112 + 277.3668893667367


def denorm_z50(z50):
    return z50 * 5882.393912540581 + 199363.18309294872


def denorm_z250(z250):
    return z250 * 5537.287144094636 + 101226.7467948718


def denorm_z500(z500):
    return z500 * 3353.5545664452306 + 54117.3952323718


def denorm_z600(z600):
    return z600 * 2696.302177194697 + 40649.71213942308


def denorm_z700(z700):
    return z700 * 2137.0576819215225 + 28929.418920272437


def denorm_z850(z850):
    return z850 * 1471.5438146105798 + 13749.576822916666


def denorm_z925(z925):
    return z925 * 1230.0568519758604 + 7014.495780749198


def denorm_z1000(z1000):
    return z1000 * 1072.7004633440063 + 736.8600307366787


def denorm_tau(tau):
    return tau * 3096.4446045099244 + 6048.8221254006414


def denorm_u50(u50):
    return u50 * 15.284072111757201 + 5.651555599310459


def denorm_u250(u250):
    return u250 * 17.9696984120105 + 13.338717974149263


def denorm_u500(u500):
    return u500 * 11.987184423423065 + 6.552764354607998


def denorm_u600(u600):
    return u600 * 10.340552477523497 + 4.797355407323593


def denorm_u700(u700):
    return u700 * 9.206544731461376 + 3.298975400435619


def denorm_u850(u850):
    return u850 * 8.192228835263744 + 1.3959463712496636


def denorm_u925(u925):
    return u925 * 7.954505065668797 + 0.5791668066611657


def denorm_v50(v50):
    return v50 * 7.067888921073515 + 0.004644189133847622


def denorm_v250(v250):
    return v250 * 13.388158186264183 - 0.030615646532402396


def denorm_v500(v500):
    return v500 * 9.186385511519138 - 0.02327536712758816


def denorm_v600(v600):
    return v600 * 7.805535575721749 - 0.030413604986209136


def denorm_v700(v700):
    return v700 * 6.894049310040708 + 0.04160793335774006


def denorm_v850(v850):
    return v850 * 6.288698845750149 + 0.16874054494576576


def denorm_v925(v925):
    return v925 * 6.490122512802569 + 0.23735194137463203


def denorm_q50(q50):
    return q50 * 3.6121240315989756e-07 + 2.665544594166045e-06


def denorm_q250(q250):
    return q250 * 7.4480380199925e-05 + 5.782029126212598e-05


def denorm_q500(q500):
    return q500 * 0.001079534297474708 + 0.0008543887763666228


def denorm_q600(q600):
    return q600 * 0.0017701706674727745 + 0.0015437401389368833


def denorm_q700(q700):
    return q700 * 0.002546475376073099 + 0.002432438085237757


def denorm_q850(q850):
    return q850 * 0.004106876858978989 + 0.004572244002841986


def denorm_q925(q925):
    return q925 * 0.005071411533793075 + 0.006030511206541306


def identical(x):
    return x


means = {
    "t2m": mean_t2m,
    "tcc": mean_tcc,
    "tp": mean_log_tp,
    "tisr": mean_tisr,
    "t50": mean_t50,
    "t250": mean_t250,
    "t500": mean_t500,
    "t600": mean_t600,
    "t700": mean_t700,
    "t850": mean_t850,
    "t925": mean_t925,
    "z50": mean_z50,
    "z250": mean_z250,
    "z500": mean_z500,
    "z600": mean_z600,
    "z700": mean_z700,
    "z850": mean_z850,
    "z925": mean_z925,
    "z1000": mean_z1000,
    "tau": mean_tau,
    "u50": mean_u50,
    "u250": mean_u250,
    "u500": mean_u500,
    "u600": mean_u600,
    "u700": mean_u700,
    "u850": mean_u850,
    "u925": mean_u925,
    "v50": mean_v50,
    "v250": mean_v250,
    "v500": mean_v500,
    "v600": mean_v600,
    "v700": mean_v700,
    "v850": mean_v850,
    "v925": mean_v925,
    "q50": mean_q50,
    "q250": mean_q250,
    "q500": mean_q500,
    "q600": mean_q600,
    "q700": mean_q700,
    "q850": mean_q850,
    "q925": mean_q925,
    "test": 0.5,
    "data": 0.0,
}


stds = {
    "t2m": std_t2m,
    "tcc": std_tcc,
    "tp": std_log_tp,
    "tisr": std_tisr,
    "t50": std_t50,
    "t250": std_t250,
    "t500": std_t500,
    "t600": std_t600,
    "t700": std_t700,
    "t850": std_t850,
    "t925": std_t925,
    "z50": std_z50,
    "z250": std_z250,
    "z500": std_z500,
    "z600": std_z600,
    "z700": std_z700,
    "z850": std_z850,
    "z925": std_z925,
    "z1000": std_z1000,
    "tau": std_tau,
    "u50": std_u50,
    "u250": std_u250,
    "u500": std_u500,
    "u600": std_u600,
    "u700": std_u700,
    "u850": std_u850,
    "u925": std_u925,
    "v50": std_v50,
    "v250": std_v250,
    "v500": std_v500,
    "v600": std_v600,
    "v700": std_v700,
    "v850": std_v850,
    "v925": std_v925,
    "q50": std_q50,
    "q250": std_q250,
    "q500": std_q500,
    "q600": std_q600,
    "q700": std_q700,
    "q850": std_q850,
    "q925": std_q925,
    "test": 0.5,
    "data": 1.0,
}

normalizors = {
    "t2m": norm_t2m,
    "tcc": norm_tcc,
    "tp": norm_tp,
    "tisr": norm_tisr,
    "t50": norm_t50,
    "t250": norm_t250,
    "t500": norm_t500,
    "t600": norm_t600,
    "t700": norm_t700,
    "t850": norm_t850,
    "t925": norm_t925,
    "z50": norm_z50,
    "z250": norm_z250,
    "z500": norm_z500,
    "z600": norm_z600,
    "z700": norm_z700,
    "z850": norm_z850,
    "z925": norm_z925,
    "z1000": norm_z1000,
    "tau": norm_tau,
    "u50": norm_u50,
    "u250": norm_u250,
    "u500": norm_u500,
    "u600": norm_u600,
    "u700": norm_u700,
    "u850": norm_u850,
    "u925": norm_u925,
    "v50": norm_v50,
    "v250": norm_v250,
    "v500": norm_v500,
    "v600": norm_v600,
    "v700": norm_v700,
    "v850": norm_v850,
    "v925": norm_v925,
    "q50": norm_q50,
    "q250": norm_q250,
    "q500": norm_q500,
    "q600": norm_q600,
    "q700": norm_q700,
    "q850": norm_q850,
    "q925": norm_q925,
    "test": identical,
    "data": identical,
}


denormalizors = {
    "t2m": denorm_t2m,
    "tcc": denorm_tcc,
    "tp": denorm_tp,
    "tisr": denorm_tisr,
    "t50": denorm_t50,
    "t250": denorm_t250,
    "t500": denorm_t500,
    "t600": denorm_t600,
    "t700": denorm_t700,
    "t850": denorm_t850,
    "t925": denorm_t925,
    "z50": denorm_z50,
    "z250": denorm_z250,
    "z500": denorm_z500,
    "z600": denorm_z600,
    "z700": denorm_z700,
    "z850": denorm_z850,
    "z925": denorm_z925,
    "z1000": denorm_z1000,
    "tau": denorm_tau,
    "u50": denorm_u50,
    "u250": denorm_u250,
    "u500": denorm_u500,
    "u600": denorm_u600,
    "u700": denorm_u700,
    "u850": denorm_u850,
    "u925": denorm_u925,
    "v50": denorm_v50,
    "v250": denorm_v250,
    "v500": denorm_v500,
    "v600": denorm_v600,
    "v700": denorm_v700,
    "v850": denorm_v850,
    "v925": denorm_v925,
    "q50": denorm_q50,
    "q250": denorm_q250,
    "q500": denorm_q500,
    "q600": denorm_q600,
    "q700": denorm_q700,
    "q850": denorm_q850,
    "q925": denorm_q925,
    "test": identical,
    "data": identical,
}

# Registry APIs for normalizers/denormalizers

_logger = logging.getLogger(__name__)


def _canonical_key(key: str) -> str:
    """
    Resolve a user-provided key (variable name or code) to the canonical code key
    used by normalizors/denormalizors.

    Rules:
    - If key already exists in the dicts (assumed to be a code), return it.
    - If key matches a known variable name in `codes`, map to its code.
    - If key is a known code in `code2var`, return it.
    - Otherwise, return the original key (allows advanced/custom usage).
    """
    if key in normalizors or key in denormalizors:
        return key
    if key in codes:
        return codes[key]
    if key in code2var:
        return key
    return key


def register_normalizer(key: str, fn: Callable, *, override: bool = False) -> None:
    """
    Register (or update) a normalizer function.

    - `key` may be a variable name or a code; it will be resolved to a canonical code.
    - Idempotent when the same function is already registered.
    - If a different function exists and `override=False`, raises ValueError.
      With `override=True`, overwrites and logs a WARNING.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    ckey = _canonical_key(key)

    existing = normalizors.get(ckey)
    if existing is fn:
        _logger.debug("register_normalizer: idempotent for %s", ckey)
        return
    if existing is not None and not override:
        raise ValueError(
            f"Normalizer for '{ckey}' already exists. Use override=True to replace."
        )
    if existing is not None and override:
        _logger.warning("Overriding normalizer for %s", ckey)
    normalizors[ckey] = fn


def register_denormalizer(key: str, fn: Callable, *, override: bool = False) -> None:
    """
    Register (or update) a denormalizer function.

    - `key` may be a variable name or a code; it will be resolved to a canonical code.
    - Idempotent when the same function is already registered.
    - If a different function exists and `override=False`, raises ValueError.
      With `override=True`, overwrites and logs a WARNING.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    ckey = _canonical_key(key)

    existing = denormalizors.get(ckey)
    if existing is fn:
        _logger.debug("register_denormalizer: idempotent for %s", ckey)
        return
    if existing is not None and not override:
        raise ValueError(
            f"Denormalizer for '{ckey}' already exists. Use override=True to replace."
        )
    if existing is not None and override:
        _logger.warning("Overriding denormalizer for %s", ckey)
    denormalizors[ckey] = fn


def get_normalizer(key: str) -> Optional[Callable]:
    """
    Get a normalizer by variable name or code. Returns None if not found.
    """
    ckey = _canonical_key(key)
    return normalizors.get(ckey)


def get_denormalizer(key: str) -> Optional[Callable]:
    """
    Get a denormalizer by variable name or code. Returns None if not found.
    """
    ckey = _canonical_key(key)
    return denormalizors.get(ckey)
