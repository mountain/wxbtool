# -*- coding: utf-8 -*-

import numpy as np
import torch as th


def norm_t2m(t2m):
    return (t2m - 278.5193277994792) / 21.219592501509624


def norm_tcc(tcc):
    return (tcc - 0.6740332964139107) / 0.3626919709448507


def norm_tp(tp):
    if type(tp) is th.Tensor:
        tp = th.log(0.001 + tp) - np.log(0.001)
    else:
        tp = np.log(0.001 + tp) - np.log(0.001)
    return (tp - 0.07109370560218127) / 0.1847837422860926


def norm_tisr(tisr):
    return (tisr - 1074511.0673076923) / 1439848.7984975462


def norm_t50(t50):
    return (t50 - 212.440180069361) / 10.260532486959207


def norm_t250(t250):
    return (t250 - 222.76936692457932) / 8.532687741643816


def norm_t500(t500):
    return (t500 - 252.95991789988983) / 13.062728754466221


def norm_t600(t600):
    return (t600 - 261.14625666691705) / 13.418785261563986


def norm_t700(t700):
    return (t700 - 267.4045856182392) / 14.767785857645688


def norm_t850(t850):
    return (t850 - 274.57741292317706) / 15.572175300640202


def norm_t925(t925):
    return (t925 - 277.3668893667367) / 16.067193457642112


def norm_z50(z50):
    return (z50 - 199363.18309294872) / 5882.393912540581


def norm_z250(z250):
    return (z250 - 101226.7467948718) / 5537.287144094636


def norm_z500(z500):
    return (z500 - 54117.3952323718) / 3353.5545664452306


def norm_z600(z600):
    return (z600 - 40649.71213942308) / 2696.302177194697


def norm_z700(z700):
    return (z700 - 28929.418920272437) / 2137.0576819215225


def norm_z850(z850):
    return (z850 - 13749.576822916666) / 1471.5438146105798


def norm_z925(z925):
    return (z925 - 7014.495780749198) / 1230.0568519758604


def norm_z1000(z1000):
    return (z1000 - 736.8600307366787) / 1072.7004633440063


def norm_tau(tau):
    return (tau - 6048.8221254006414) / 3096.4446045099244


def norm_u50(u50):
    return (u50 - 5.651555599310459) / 15.284072111757201


def norm_u250(u250):
    return (u250 - 13.338717974149263) / 17.9696984120105


def norm_u500(u500):
    return (u500 - 6.552764354607998) / 11.987184423423065


def norm_u600(u600):
    return (u600 - 4.797355407323593) / 10.340552477523497


def norm_u700(u700):
    return (u700 - 3.298975400435619) / 9.206544731461376


def norm_u850(u850):
    return (u850 - 1.3959463712496636) / 8.192228835263744


def norm_u925(u925):
    return (u925 - 0.5791668066611657) / 7.954505065668797


def norm_v50(v50):
    return (v50 - 0.004644189133847622) / 7.067888921073515


def norm_v250(v250):
    return (v250 + 0.030615646532402396) / 13.388158186264183


def norm_v500(v500):
    return (v500 + 0.02327536712758816) / 9.186385511519138


def norm_v600(v600):
    return (v600 + 0.030413604986209136) / 7.805535575721749


def norm_v700(v700):
    return (v700 - 0.04160793335774006) / 6.894049310040708


def norm_v850(v850):
    return (v850 - 0.16874054494576576) / 6.288698845750149


def norm_v925(v925):
    return (v925 - 0.23735194137463203) / 6.490122512802569


def norm_q50(q50):
    return (q50 - 2.665544594166045e-06) / 3.6121240315989756e-07


def norm_q250(q250):
    return (q250 - 5.782029126212598e-05) / 7.4480380199925e-05


def norm_q500(q500):
    return (q500 - 0.0008543887763666228) / 0.001079534297474708


def norm_q600(q600):
    return (q600 - 0.0015437401389368833) / 0.0017701706674727745


def norm_q700(q700):
    return (q700 - 0.002432438085237757) / 0.002546475376073099


def norm_q850(q850):
    return (q850 - 0.004572244002841986) / 0.004106876858978989


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
    "test": lambda x: x,
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
    "test": lambda x: x,
}
