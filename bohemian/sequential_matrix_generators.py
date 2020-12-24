import numpy as np


def number_to_base(n, b, N):
    if n == 0:
        return np.repeat(0, N)
    digits = np.zeros(N, dtype=np.uint64)
    i = 0
    while n:
        digits[i] = int(n % b)
        n //= b
        i += 1
    return digits

# Generator that returns matrices in order
def sequential_matrix(population, n):
    def sequential_matrix(batch_size):
        _population = np.array(population)
        _n = n
        
        # Total number of matrices in the Bohemian family
        num_matrices = len(_population) ** (_n ** 2)

        # Number of batches
        num_batches = num_matrices // batch_size + 1

        A = np.zeros((batch_size, _n, _n), dtype = np.complex128)

        # Integer count that maps to matrices in the family
        c = 0

        for i in range(num_batches):

            # Number of matrices remaining
            num_matrices_in_batch = batch_size if i != num_batches - 1 else num_matrices % batch_size

            for j in range(num_matrices_in_batch):
                # Indices of the population that define the current matrix
                idx = number_to_base(c, len(_population), _n ** 2)

                # Get entries
                p = _population[idx]

                A[j, :, :] = p.reshape((_n, _n))

                c = c + 1

            yield A[:num_matrices_in_batch, :, :]

    return sequential_matrix


# Generator that returns matrices in order
def sequential_frobenius_companion_matrix(population, n):
    def sequential_frobenius_companion_matrix(batch_size):
        _population = np.array(population)
        _n = n

        # Total number of matrices in the Bohemian family
        num_matrices = len(_population) ** _n

        # Number of batches
        num_batches = num_matrices // batch_size + 1

        A = np.zeros((batch_size, _n, _n), dtype=np.int)

        # Integer count that maps to matrices in the family
        c = 0

        for i in range(num_batches):

            # Number of matrices remaining
            num_matrices_in_batch = batch_size if i != num_batches - 1 else num_matrices % batch_size

            B = np.diag(np.repeat(1, _n - 1), k=-1)

            for j in range(num_matrices_in_batch):
                # Indices of the population that define the current matrix
                idx = number_to_base(c, len(_population), _n)

                # Get entries
                p = _population[idx]

                B[:, -1] = 0
                B[:, -1] = B[:, -1] - p

                A[j, :, :] = B

                c = c + 1

            yield A[:num_matrices_in_batch, :, :]

    return sequential_frobenius_companion_matrix
