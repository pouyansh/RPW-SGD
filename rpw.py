import numpy as np # type: ignore
import jpype # type: ignore
import jpype.imports # type: ignore
from jpype.types import * # type: ignore
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping # type: ignore

def RPW(A, C=None, delta=0.001, k=1, p=1):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    y = C.shape[1]
    X = np.array(A)
    Y = np.array([1/y for _ in range(y)])
    dist = np.array(C.tolist())
    nz = len(X)
    alphaa = 4.0*np.max(dist)/delta
    gtSolver = Mapping(nz, X, Y, dist, delta)
    APinfo = np.array(gtSolver.getAPinfo()) # augmenting path info

    if APinfo.size == 0:
        return 0

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]

    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = (np.cumsum(cost_AP)/(alphaa*alphaa*nz))**(1/p)
    # cumCost = np.cumsum(cost_AP)/(alphaa*alphaa*nz)

    cumCost *= 1/k
    totalCost = cumCost[-1]
    if totalCost == 0:
        normalized_cumcost = (cumCost) * 0.0
    else:
        normalized_cumcost = (cumCost)/(1.0 * totalCost)

    maxdual = APinfo_cleaned[:,4]/alphaa*1/k
    final_dual = maxdual[-1]
    if final_dual == 0:
        normalized_maxdual = maxdual * 0.0
    else:
        normalized_maxdual = maxdual/final_dual

    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    d_cost = (1 - flowProgress) - cumCost
    try:
        d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
    except:
        return 0
    d_ind_b = d_ind_a + 1
    alpha = find_intersection_point(flowProgress[d_ind_a], d_cost[d_ind_a], flowProgress[d_ind_b], d_cost[d_ind_b])
    res = 1 - alpha
    return res

def find_intersection_point(x1, y1, x2, y2):
    # x1 < x2
    # y1 > 0
    # y2 < 0
    # y = ax + b
    # find x when y = 0
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    x = -b/a
    return x