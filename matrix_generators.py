# Matrix Generators
#  - random_matrix
#  - random_tridiagonal_matrix
#  - random_upper_hessenberg_matrix_matrix
#  - random_upper_hessenberg_toeplitz
#  - random_checkerboard_matrix
#
# To Do:
#  - random_pentadiagonal_matrix
#  - random_antipentadiagonal_matrix
#  - random_antitridiagonal_matrix
#  - random_inverted_checkerboard_matrix
#  - random_vandermonde_matrix
#  - random_cauchy_matrix
#  - random_circulant_matrix
#  - random_frobenius_companion_matrix
#  - random_hankel_matrix
#  - random_frobenius_doubly_companion_matrix
#  - random_toeplitz_matrix

import numpy as np


def strip_first_dimension(func):
    """
    Used as a decorator.

    Strip the first dimension of an array if it is 1, otherwise leave array
    as is.

    Args:
        func: callable

    Returns:
        If func returns a 3-dimensional array where the first dimension is 1,
        the first dimension will be removed. i.e. if func returns an array of
        dimension (1, 12, 12), the first dimension will be stripped and an
        array of dimension (12, 12) will be returned. Alternatively, if func
        returns an array of dimension (5, 12, 12), the array will not be
        modified.
    """
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        return arr[0] if arr.shape[0] == 1 else arr
    return wrapper


def get_dtype(*args):
    """
    Given a list of arguments, the appropriate data type for the matrix
    generator is returned. If any of the input values are complex or contain
    complex values, np.complex128 is returned. Otherwise, np.float64 is
    returned.

    Args:
        *args: Supplied arguments

    Returns:
        np.complex128 if any of the arguments are complex, otherwise np.float64
    """
    return np.complex128 if np.any([np.any(np.iscomplex(arg)) for arg in args]) else np.float64
    

"""
Matrix Generators
"""

@strip_first_dimension
def random_matrix(population, n, num_matrices=1, d=None):
    """

    Args:
        population:
        n:
        num_matrices:
        d:

    Returns:

    """
    
    # Get the datatype to use for output
    dt = get_dtype(population, d)
    
    A = np.random.choice(population, size=(num_matrices, n, n)).astype(dt)
    
    if d is not None:
        for i in range(num_matrices):
            np.fill_diagonal(A[i], d)
    
    return A


@strip_first_dimension
def random_tridiagonal_matrix(population, n, num_matrices=1, d=None):
    """

    Args:
        population:
        n:
        num_matrices:
        d:

    Returns:

    """
    
    # Get the datatype to use for output
    dt = get_dtype(population, d)
    
    if d is None:
        v_main = np.random.choice(population, (num_matrices, n)).astype(dt)
        A_main = np.apply_along_axis(np.diag, 1, v_main)
    else:
        A_main = np.apply_along_axis(np.diag, 1, d*np.ones((num_matrices, n))).astype(dt)
    
    v_sub = np.random.choice(population, (num_matrices, n-1))
    A_sub = np.apply_along_axis(np.diag, 1, v_sub, k=-1)
    
    v_super = np.random.choice(population, (num_matrices, n-1))
    A_super = np.apply_along_axis(np.diag, 1, v_super, k=1)
    
    A = A_main + A_sub + A_super
    
    return A

@strip_first_dimension
def random_upper_hessenberg_matrix(population, n, s=None, d=None, num_matrices=1):
    """

    Args:
        population:
        n:
        s:
        d:
        num_matrices:

    Returns:

    """
    
    # Get the datatype to use for output
    dt = get_dtype(population, s, d)
    
    if s is None:
        v_sub = np.random.choice(population, (num_matrices, n-1)).astype(dt)
    else:
        v_sub = s*np.ones((num_matrices, n-1), dtype=dt)
    
    if d is None:
        v_diag = np.random.choice(population, (num_matrices, n)).astype(dt)
    else:
        v_diag = d*np.ones((num_matrices, n), dtype=dt)
    
    A = np.apply_along_axis(np.diag, 1, v_sub , k=-1) + np.apply_along_axis(np.diag, 1, v_diag)
    
    A += np.triu(np.random.choice(population, size=(num_matrices, n, n)), k=1)
    
    return A
    

@strip_first_dimension
def random_upper_hessenberg_toeplitz_matrix(population, n, s=None, d=None, num_matrices=1):
    """

    Args:
        population:
        n:
        s:
        d:
        num_matrices:

    Returns:

    """
    
    # Get the datatype to use for output
    dt = get_dtype(population, s, d)
    
    # Subdiagonal
    if s is None:
        v_sub = np.random.choice(population, (num_matrices,1))*np.ones((num_matrices, n-1))
    else:
        v_sub = s*np.ones((num_matrices, n-1), dtype=dt)
    
    # Main diagonal
    if d is None:
        v_diag = np.random.choice(population, (num_matrices,1))*np.ones((num_matrices, n))
    else:
        v_diag = d*np.ones((num_matrices, n), dtype=dt)
    
    A = np.apply_along_axis(np.diag, 1, v_sub, k=-1) + np.apply_along_axis(np.diag, 1, v_diag)
    
    for i in range(n-1, 0, -1):
        v = np.random.choice(population, (num_matrices,1))*np.ones((num_matrices, i))
        A += np.apply_along_axis(np.diag, 1, v, k=n-i)
    
    return A


@strip_first_dimension
def random_checkerboard_matrix(population, n, num_matrices=1):
    """
    A "checkerboard" matrix contains zeros on the main diagonal and every
    even sub/super diagonal (i.e. the 2nd, 4th, 6th, etc. sub- and
    superdiagonals).

    Args:
        population:
        n:
        num_matrices:

    Returns:

    """
    
    # Get the datatype to use for output
    dt = get_dtype(population)
    
    A = np.zeros((num_matrices, n, n), dtype=dt)
    
    n_odd = n - 2 if n % 2 == 1 else n - 1
    
    for i in range(-n_odd, n_odd+1, 2):
            v = np.random.choice(population, (num_matrices, n-abs(i)))
            A += np.apply_along_axis(np.diag, 1, v, k=i)
    
    return A
