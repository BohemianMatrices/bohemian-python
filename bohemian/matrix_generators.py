# Matrix Generators:
#  - random_matrix
#  - random_tridiagonal_matrix
#  - random_antitridiagonal_matrix
#  - random_antipentadiagonal_matrix
#  - random_upper_hessenberg_matrix
#  - random_upper_hessenberg_toeplitz_matrix
#  - random_checkerboard_matrix
#  - random_persymmetric_matrix
#  - random_symmetric_matrix
#  - random_centrosymmetric_matrix
#  - random_sylvester_matrix
#  - random_frobenius_doubly_companion_matrix
#  - random_frobenius_companion_matrix
#
# To Do:
#  - random_pentadiagonal_matrix
#  - random_inverted_checkerboard_matrix
#  - random_vandermonde_matrix
#  - random_cauchy_matrix
#  - random_circulant_matrix
#  - random_hankel_matrix
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
    return np.complex128 if np.any(
        [np.any(np.iscomplex(arg)) for arg in args]) else np.float64


def random_matrix(population, n, d=None):
    """
    Generate random matrices with entries sampled from the population.

    :param population: Population of the Bohemian family
    :param n: Matrix dimension
    :param d: Value for diagonal entries. When set to None, the diagonal
    entries are randomly sampled from the population. Defaults to None.

    :return: A function that takes a single integer n as input and returns a
    generator that will yield n matrices on each iteration. The generator can
    be called an infinite number of times.
    """

    # Get the datatype to use for output
    dt = get_dtype(population, d)

    def random_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _d = d

        while True:

            A = np.random.choice(_population, size=(batch_size, _n, _n)).astype(
                _dt)

            if _d is not None:
                for i in range(batch_size):
                    np.fill_diagonal(A[i], _d)

            yield A

    return random_matrix


def random_tridiagonal_matrix(population, n, d=None):
    """
    Generate random tridiagonal matrices with entries sampled from the
    population.

    :param population: Population of the Bohemian family
    :param n: Matrix dimension
    :param d: Value for diagonal entries. When set to None, the diagonal
    entries are randomly sampled from the population. Defaults to None.

    :return: A function that takes a single integer n as input and returns a
    generator that will yield n matrices on each iteration. The generator can
    be called an infinite number of times.
    """

    # Get the datatype to use for output
    dt = get_dtype(population, d)

    def random_tridiagonal_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _d = d

        while True:

            if _d is None:
                v_main = np.random.choice(_population, (batch_size, _n)).astype(
                    _dt)
                A_main = np.apply_along_axis(np.diag, 1, v_main)
            else:
                A_main = np.apply_along_axis(np.diag, 1, _d * np.ones(
                    (batch_size, _n))).astype(_dt)

            v_sub = np.random.choice(_population, (batch_size, _n - 1))
            A_sub = np.apply_along_axis(np.diag, 1, v_sub, k=-1)

            v_super = np.random.choice(_population, (batch_size, _n - 1))
            A_super = np.apply_along_axis(np.diag, 1, v_super, k=1)

            A = A_main + A_sub + A_super

            yield A

    return random_tridiagonal_matrix


def random_antitridiagonal_matrix(population, n, d=None):
    """
    Generate random anti-tridiagonal matrices (i.e. upsidedown tridiagonal
    matices) with entries sampled from the population.

    :param population: Population of the Bohemian family
    :param n: Matrix dimension
    :param d: Value for diagonal entries. When set to None, the diagonal
    entries are randomly sampled from the population. Defaults to None.

    :return: A function that takes a single integer n as input and returns a
    generator that will yield n matrices on each iteration. The generator can
    be called an infinite number of times.
    """

    # Get the datatype to use for output
    dt = get_dtype(population, d)

    def random_antitridiagonal_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _d = d

        while True:

            if _d is None:
                v_main = np.random.choice(_population, (batch_size, _n)).astype(
                    _dt)
                A_main = np.apply_along_axis(np.diag, 1, v_main)
            else:
                A_main = np.apply_along_axis(np.diag, 1, _d * np.ones(
                    (batch_size, _n))).astype(_dt)

            v_sub = np.random.choice(_population, (batch_size, _n - 1))
            A_sub = np.apply_along_axis(np.diag, 1, v_sub, k=-1)

            v_super = np.random.choice(_population, (batch_size, _n - 1))
            A_super = np.apply_along_axis(np.diag, 1, v_super, k=1)

            A = np.fliplr(A_main + A_sub + A_super)

            yield A

    return random_antitridiagonal_matrix


def random_antipentadiagonal_matrix(population, n, d=None):
    """
    Generate random anti-pentadiagonal matrices (i.e. upsidedown pentadiagonal
    matices) with entries sampled from the population.

    :param population: Population of the Bohemian family
    :param n: Matrix dimension
    :param d: Value for diagonal entries. When set to None, the diagonal
    entries are randomly sampled from the population. Defaults to None.

    :return: A function that takes a single integer n as input and returns a
    generator that will yield n matrices on each iteration. The generator can
    be called an infinite number of times.
    """

    # Get the datatype to use for output
    dt = get_dtype(population, d)

    def random_antipentadiagonal_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _d = d

        while True:

            if _d is None:
                v_main = np.random.choice(_population, (batch_size, _n)).astype(
                    _dt)
                A_main = np.apply_along_axis(np.diag, 1, v_main)
            else:
                A_main = np.apply_along_axis(np.diag, 1, _d * np.ones(
                    (batch_size, _n))).astype(_dt)

            v_sub = np.random.choice(_population, (batch_size, _n - 2))
            A_sub = np.apply_along_axis(np.diag, 1, v_sub, k=-2)

            v_super = np.random.choice(_population, (batch_size, _n - 2))
            A_super = np.apply_along_axis(np.diag, 1, v_super, k=2)

            A = np.fliplr(A_main + A_sub + A_super)

            yield A

    return random_antipentadiagonal_matrix


def random_upper_hessenberg_matrix(population, n, s=None, d=None):
    # Get the datatype to use for output
    dt = get_dtype(population, s, d)

    def random_upper_hessenberg_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _s = s
        _d = d

        while True:

            if _s is None:
                v_sub = np.random.choice(_population,
                                         (batch_size, _n - 1)).astype(_dt)
            else:
                v_sub = s * np.ones((batch_size, _n - 1), dtype=_dt)

            if _d is None:
                v_diag = np.random.choice(_population, (batch_size, _n)).astype(
                    _dt)
            else:
                v_diag = _d * np.ones((batch_size, _n), dtype=_dt)

            A = np.apply_along_axis(np.diag, 1, v_sub,
                                    k=-1) + np.apply_along_axis(np.diag, 1,
                                                                v_diag)

            A += np.triu(
                np.random.choice(_population, size=(batch_size, _n, _n)), k=1)

            yield A

    return random_upper_hessenberg_matrix


def random_upper_hessenberg_toeplitz_matrix(population, n, s=None, d=None):
    # Get the datatype to use for output
    dt = get_dtype(population, s, d)

    def random_upper_hessenberg_toeplitz_matrix(batch_size):

        while True:

            _dt = dt
            _population = population
            _n = n
            _s = s
            _d = d

            # Subdiagonal
            if _s is None:
                v_sub = np.random.choice(_population,
                                         (batch_size, 1)) * np.ones(
                    (batch_size, _n - 1))
            else:
                v_sub = _s * np.ones((batch_size, _n - 1), dtype=_dt)

            # Main diagonal
            if _d is None:
                v_diag = np.random.choice(_population,
                                          (batch_size, 1)) * np.ones(
                    (batch_size, _n))
            else:
                v_diag = _d * np.ones((batch_size, _n), dtype=_dt)

            A = np.apply_along_axis(np.diag, 1, v_sub,
                                    k=-1) + np.apply_along_axis(np.diag, 1,
                                                                v_diag)

            for i in range(_n - 1, 0, -1):
                v = np.random.choice(_population, (batch_size, 1)) * np.ones(
                    (batch_size, i))
                A += np.apply_along_axis(np.diag, 1, v, k=_n - i)

            yield A

    return random_upper_hessenberg_toeplitz_matrix


def random_checkerboard_matrix(population, n):
    """
    Generate random "checkerboard" matrices. A "checkerboard" matrix is a matrix
    where the main diagonal, and all even diagonals are filled with zeros. All
    other entries (i.e. all odd diagonals) are randomly populated from the
    population.

    :param population: Population of the Bohemian family
    :param n: Matrix dimension

    :return: A function that takes a single integer n as input and returns a
    generator that will yield n matrices on each iteration. The generator can
    be called an infinite number of times.
    """

    # Get the datatype to use for output
    dt = get_dtype(population)

    def random_checkerboard_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n

        while True:

            A = np.zeros((batch_size, _n, _n), dtype=_dt)

            n_odd = _n - 2 if _n % 2 == 1 else _n - 1

            for i in range(-n_odd, n_odd + 1, 2):
                v = np.random.choice(_population, (batch_size, _n - abs(i)))
                A += np.apply_along_axis(np.diag, 1, v, k=i)

            yield A

    return random_checkerboard_matrix


def random_inverted_checkerboard_matrix(population, n):
    """
    Generate random "inverted checkerboard" matrices. An "inverted
    checkerboard" matrix is a matrix where the odd diagonals (i.e. 1st
    super/sub diagonal, 3rd super/sub diagonal, etc.) are filled with zeros. All
    other entries (i.e. the main diagonal and all even diagonals) are
    randomly populated from the population.

    :param population: Population of the Bohemian family
    :param n: Matrix dimension

    :return: A function that takes a single integer n as input and returns a
    generator that will yield n matrices on each iteration. The generator can
    be called an infinite number of times.
    """

    # Get the datatype to use for output
    dt = get_dtype(population)
    
    def random_inverted_checkerboard_matrix(batch_size):
        
        _dt = dt
        _population = population
        _n = n

        while True:

            A = np.zeros((batch_size, _n, _n), dtype=_dt)

            n_even = _n - 2 if n % 2 == 0 else _n - 1

            for i in range(-n_even, n_even + 1, 2):
                v = np.random.choice(_population, (batch_size, _n - abs(i)))
                A += np.apply_along_axis(np.diag, 1, v, k=i)

            yield A

    return random_inverted_checkerboard_matrix


def random_persymmetric_matrix(population, n):
    # Get the datatype to use for output
    dt = get_dtype(population)

    def random_persymmetric_matrix(batch_size):
        _dt = dt
        _population = population
        _n = n

        while True:
            v_diag = np.random.choice(_population, (batch_size, _n)).astype(_dt)

            A = np.apply_along_axis(np.diag, 1, v_diag)

            Au = np.triu(
                np.random.choice(_population, size=(batch_size, _n, _n)), k=1)
            AuT = Au.transpose((0, 2, 1))

            A += Au + AuT

            yield np.fliplr(A)

    return random_persymmetric_matrix


def random_symmetric_matrix(population, n, d=None):
    dt = get_dtype(population)

    def random_symmetric_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _d = d

        while True:

            A = np.zeros((batch_size, _n, _n), dtype=_dt)

            for i in range(0, batch_size):

                if _d is not None:
                    l = np.random.choice(_population, size=_n * (_n - 1) // 2)
                    B = np.triu(np.ones(_n, dtype=_dt), 1)
                    B[B == 1] = l
                    B = B + np.tril(np.transpose(B), -1)
                    B = B + np.diag(np.repeat(_d, _n))
                else:
                    l = np.random.choice(_population, size=_n * (_n + 1) // 2)
                    B = np.triu(np.ones(_n, dtype=_dt))
                    B[B == 1] = l
                    B = B + np.tril(np.transpose(B), -1)

                A[i, :, :] = B

            yield A

    return random_symmetric_matrix


def random_centrosymmetric_matrix(population, n):
    dt = get_dtype(population)

    def random_centrosymmetric_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n

        while True:

            A = np.zeros((batch_size, _n, _n), dtype=_dt)
            for i in range(0, _n):
                stop_idx = _n - i - 1 if _n - i < _n / 2 else _n - i
                for j in range(0, stop_idx):
                    A[:, i, j] = np.random.choice(_population, batch_size)
                    A[:, _n - i - 1, _n - j - 1] = A[:, i, j]
            yield A

    return random_centrosymmetric_matrix


def random_sylvester_matrix(population, n):
    from scipy.linalg import toeplitz

    dt = get_dtype(population)

    def random_sylvester_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n

        while True:

            A = np.zeros((batch_size, 2 * _n - 2, 2 * _n - 2), dtype=_dt)

            for i in range(batch_size):
                p1 = np.random.choice(_population, _n)
                p2 = np.random.choice(_population, _n)

                SP = toeplitz(np.concatenate([p1, np.zeros(_n - 2)]),
                              r=np.zeros(_n - 1)).T
                SQ = toeplitz(np.concatenate([p2, np.zeros(_n - 2)]),
                              r=np.zeros(_n - 1)).T

                A[i, :, :] = np.concatenate([SP, SQ])

            yield A

    return random_sylvester_matrix


def random_frobenius_doubly_companion_matrix(population, n):
    dt = get_dtype(population)

    def random_frobenius_doubly_companion_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n

        while True:

            A = np.zeros((batch_size, _n, _n), dtype=_dt)

            for i in range(batch_size):
                B = np.diag(np.repeat(1, _n - 1), k=-1)

                p1 = np.random.choice(_population, _n)
                p2 = np.random.choice(_population, _n)

                B[0, :] = B[0, :] - p1
                B[:, -1] = B[:, -1] - p2

                A[i, :, :] = B

            yield A

    return random_frobenius_doubly_companion_matrix


def random_frobenius_companion_matrix(population, n):
    dt = get_dtype(population)

    def random_frobenius_companion_matrix(batch_size):

        while True:

            _dt = dt
            _population = population
            _n = n

            A = np.zeros((batch_size, _n, _n), dtype=_dt)

            for i in range(batch_size):
                B = np.diag(np.repeat(1, _n - 1), k=-1)

                p = np.random.choice(_population, _n)

                B[:, -1] = B[:, -1] - p

                A[i, :, :] = B

            yield A

    return random_frobenius_companion_matrix


def random_circulant_matrix(population, n):
    
    from scipy.linalg import circulant
    
    dt = get_dtype(population)
    
    def random_circulant_matrix(batch_size):
    
        _dt = dt
        _population = population
        _n = n
    
        while True:
    
            A = np.zeros((batch_size, _n, _n), dtype = _dt)
    
            for i in range(batch_size):
                
                p = np.random.choice(_population, _n)
                
                A[i, :, :] = circulant(p)
    
            yield A
    
    return random_circulant_matrix


def random_tridiagonal_skew_symmetric_matrix(population, n, d=None):
    
    # Get the datatype to use for output
    dt = get_dtype(population, d)
    
    def random_tridiagonal_skew_symmetric_matrix(batch_size):
    
        _dt = dt
        _population = population
        _n = n
        _d = d
    
        while True:
    
            if _d is None:
                v_main = np.random.choice(_population, (batch_size, _n)).astype(
                    _dt)
                A_main = np.apply_along_axis(np.diag, 1, v_main)
            else:
                A_main = np.apply_along_axis(np.diag, 1, _d * np.ones(
                    (batch_size, _n))).astype(_dt)
    
            v = np.random.choice(_population, (batch_size, _n - 1))
            A_sub = np.apply_along_axis(np.diag, 1, v, k=-1)
            A_super = np.apply_along_axis(np.diag, 1, -v, k=1)
    
            A = A_main + A_sub + A_super
    
            yield A
    
    return random_tridiagonal_skew_symmetric_matrix