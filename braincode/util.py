import numpy as np
from scipy.io import loadmat


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            value = function(*args)
            memo[args] = value
            return value

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


def init_cmat(n):
    return np.zeros((n, n))


def accuracy(cmat):
    return np.trace(cmat) / cmat.sum()
