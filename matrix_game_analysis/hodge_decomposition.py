"""
Hodge decomposition.
Re-evaluate evaluation: https://arxiv.org/pdf/1806.02643.pdf
"""

import numpy as np

def antisymmetric_verification(A):
    A_bar = A + np.transpose(A)
    if np.count_nonzero(A_bar) == 0:
        return True
    else:
        return False

def divergence(A):
    dim = np.shape(A)[0]
    ones = np.ones(dim)
    ones = np.reshape(ones, newshape=(dim, 1))
    div = 1/dim * np.dot(A, ones)
    return div

def grad(r):
    dim = np.shape(r)[0]
    ones = np.ones(dim)
    ones_row = np.reshape(ones, newshape=(1, dim))
    ones_col = np.reshape(ones, newshape=(dim, 1))
    grad = np.dot(r, ones_row) - np.dot(ones_col, np.transpose(r))
    return grad

def rot(A):
    dim = np.shape(A)[0]
    rot = np.zeros((dim, dim))
    for i in np.arange(dim):
        for j in np.arange(dim):
            rot[i,j] = A[i,j] + 1/dim * np.sum(A[j, :] - A[i, :])

    return rot

def hodge_decomposition(A):
    if not antisymmetric_verification(A):
        raise ValueError("The matrix should be antisymmetric.")
    r = divergence(A)
    gradr = grad(r)
    rotA = rot(A)
    return gradr, rotA, r

def print_decomp(A):
    gradr, rotA, r = hodge_decomposition(A)
    print("A:\n", A)
    print("r:\n", r)
    print("grad:\n", gradr)
    print("rot:\n", rotA)

RPS_numbers = np.array([[0, 3, -1, -2, 1, 1],
                        [0, 0, -1, -1, 2, 1],
                        [0, 0, 0, 3, -1, -2],
                        [0, 0, 0, 0, -1, -1],
                        [0, 0, 0, 0, 0, 3],
                        [0, 0, 0, 0, 0, 0]])

RPS_numbers = RPS_numbers - np.transpose(RPS_numbers)
print(RPS_numbers)


# A = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]) # transitive
# A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]) # cyclic
print_decomp(RPS_numbers)


