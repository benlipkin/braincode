import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


@memoize
def get_mat(fname):
    return loadmat(fname)


def parse_mat(mat, network):
    return (
        mat["data"],
        mat[network + "_tags"],
        mat["problem_content"],
        mat["problem_lang"],
        mat["problem_structure"],
    )


def formatcell(matcellarray):
    return np.array([i[0][0] for i in matcellarray])


def prep_x(data, parc):
    data = data[:, np.flatnonzero(parc)]
    for i in range(12):
        data[np.arange(i, 72, 12), :] = StandardScaler().fit_transform(
            data[np.arange(i, 72, 12), :]
        )
    return data
