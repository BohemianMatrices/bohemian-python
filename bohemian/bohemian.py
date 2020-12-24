import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time


class Bohemian:
    """

    """

    def __init__(self,
                 generator=None,
                 cpdb_filename=None,
                 working_dir=os.getcwd(),
                 data_dir='Data',
                 histogram_dir='Histogram',
                 images_dir='Images',
                 matrices_per_file=1000000,
                 batch_size=1000,
                 use_single_precision=True,
                 verbose=0):

        if (cpdb_filename is not None) and (generator is not None):
            raise Exception(
                'cpdb_filename and generator cannot both the provide.')
        if (cpdb_filename is None) and (generator is None):
            raise Exception('At least one of cpdb_filename and generator must '
                            'provided.')

        # Generator that will return a batch containing batch_size matrices
        if generator is not None:
            self.g = generator(batch_size)

        # CPDB filename
        self.cpdb_filename = cpdb_filename

        if cpdb_filename is not None:
            self.compute_method = 'cpdb'
        else:
            self.compute_method = 'generator'

        # Directories
        self.working_dir = working_dir
        self.data_dir = os.path.join(self.working_dir, data_dir)
        self.histogram_dir = os.path.join(self.working_dir, histogram_dir)
        self.images_dir = os.path.join(self.working_dir, images_dir)

        # Parameters for the data generation step
        self.matrices_per_file = matrices_per_file
        self.batch_size = batch_size
        self.use_single_precision = use_single_precision

        self.verbose = verbose

        # Index of the current data file
        self.current_data_file_idx = self.count_data_files()

        # Index of the current image file
        self.current_image_file_idx = self.count_image_files()

        # Index of the current histogram file
        self.current_histogram_file_idx = self.count_histogram_files()

        # Counter for the number of matrices whose eigenvalues have been
        # computed
        self.total_matrices_computed = 0

    def compute_eigenvalues(self,
                            num_files=1):

        # Create the data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if self.compute_method == 'cpdb':
            self.compute_eigenvalues_cpdb()
        elif self.compute_method == 'generator':
            self.compute_eigenvalues_generator(num_files=num_files)
        else:
            raise Exception('Incorrect compute method provided.')

    def compute_eigenvalues_cpdb(self):

        # Open the cpdb file
        f = open(self.cpdb_filename)

        # Skip header
        l = next(f)

        # Get the matrix dimension
        matrix_size = len(l.split(',')) - 2

        polys_remaining = True

        # Load the polynomials in batches of self.matrices_per_file
        while polys_remaining:

            # Array to store the eigenvalues
            L = np.zeros((self.matrices_per_file, matrix_size),
                         dtype=np.complex128)

            # Vector to store the eigenvalue counts
            C = np.zeros(self.matrices_per_file, dtype=np.uint64)

            # Load self.matrices_per_file polynomial
            for i in range(self.matrices_per_file):
                try:
                    line = next(f)
                except StopIteration:
                    polys_remaining = False
                    break

                line = line.rstrip('\n\r')
                line = line.split(',')

                # Polynomial count
                C[i] = int(line[0])

                # Coefficients of the polynomial
                p = np.array(list(map(int, line[1:])))

                # Compute the roots
                L[i, :] = np.roots(p)

            # Truncate (used if there is not exactly matrices_per_file
            # polynomials remaining, typically for the final final)
            if not polys_remaining:
                L = L[:i, :]
                C = C[:i]

            # Repeat entries of C
            C = np.repeat(C, matrix_size)

            # Flatten L
            L = L.flatten()

            # Save the data
            data_filename = self.get_data_filename(self.current_data_file_idx)
            self.save_data(L, C, data_filename)

            del L
            del C

            self.current_data_file_idx += 1

    def compute_eigenvalues_generator(self,
                                      num_files=None):

        # Number of batches of size self.batch_size to run
        # Note: If matrices_per_file is not divisible by batch_size,
        #       num_batches*batch_size (< matrices_per_file) matrices will be
        #       run per file.
        num_batches = self.matrices_per_file // self.batch_size

        # Indicator
        generator_done = False

        i = 0

        while not generator_done and i < num_files:
            i += 1

            if self.verbose > 0: print(
                'File {}'.format(self.current_data_file_idx + 1))

            if self.verbose > 2: print(
                'Computing eigenvalues of {} matrices'.format(
                    self.matrices_per_file))

            data_filename = self.get_data_filename(self.current_data_file_idx)

            compute_start_time = time.time()

            L = list()
            for _ in range(num_batches):
                
                try:
                    success = False

                    while not success:
                        
                        # Get a batch of matrices
                        A_batch = next(self.g)
                        
                        try:
                            # Compute the eigenvalues
                            L_batch = self.compute_eig_batch(A_batch,
                                                             self.use_single_precision)
                        except np.linalg.LinAlgError:
                            # Catch convergence errors and try again
                            print('Convergence error! Trying a new batch.')
                            pass
                        else:
                            success = True
                     
                    L.append(L_batch.flatten())
                    
                except StopIteration:
                    generator_done = True
                    break

            L = np.concatenate(L)

            if L.shape[0] == 0:
                break

            # Array storing the eigenvalue counts
            C = np.ones(L.shape[0], dtype=np.uint64)

            compute_time = time.time() - compute_start_time

            self.total_matrices_computed += self.matrices_per_file

            if self.verbose > 2:
                print('\tDone computing eigenvalues')
                print('\tSaving eigenvalues to {}'.format(data_filename))

            # Save the data
            self.save_data(L, C, data_filename)

            if self.verbose > 2:
                print('\tDone saving eigenvalues')

            if self.verbose > 1:
                print(
                    '\tComputation of eigenvalues of {0:d} matrices took {1:.3f} seconds'.format(
                        self.matrices_per_file, compute_time))

            # Free memory
            del L
            del C

            self.current_data_file_idx += 1

    def generate_histogram(self,
                           height=1001,
                           axis_range=((-1, 1), (-1, 1)),
                           symmetry_imag=False,
                           symmetry_real=False,
                           ignore_real=False,
                           ignore_real_tol=np.sqrt(np.finfo(float).eps)):
        """
        symmetry_imag: Symmetry across the imaginary axis. i.e. if a + bi is an
                       eigenvalue, -a + bi is also included in the image
        symmetry_real: Symmetry across the real axis. i.e. if a + bi is an
                       eigenvalue, a - bi is also included in the image
        If both symmetry_imag and symmetry_real are True, then each eigenvalue is
        counted 4 times. i.e. if a + bi is an eigenvalue, -a + bi, a - bi, and
        -a - bi are included in the image.
        """

        # Create the histogram directory
        if not os.path.exists(self.histogram_dir):
            os.makedirs(self.histogram_dir)

        width = self.get_width(height, axis_range[0][0], axis_range[0][1],
                               axis_range[1][0], axis_range[1][1])

        H = np.zeros((int(height), int(width)), dtype=np.uint32)

        # Loop through the data files
        for data_filename in self.get_data_files_generator():
            print(data_filename)
            H = H + self.one_file_histogram(data_filename, height, width,
                                            axis_range, symmetry_real,
                                            symmetry_imag, ignore_real,
                                            ignore_real_tol)

        # Name of the histogram file
        histogram_filename = self.get_histogram_filename()

        # Save
        self.save_histogram(H, histogram_filename)
        self.current_histogram_file_idx += 1

        return histogram_filename

    def plot(self,
             histogram_file=None,
             background_color=(0, 0, 0, 255),
             cm=plt.cm.hot,
             image_filename=None,
             histogram_map=lambda x: np.log(x)):

        # Create the images directory
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        H = self.load_histogram(histogram_file)

        with np.errstate(divide='ignore'):
            H = histogram_map(H.astype(np.float64))

        # Apply colormap
        norm = plt.Normalize(vmin=H[np.isfinite(H)].min(), vmax=H.max())
        colors = cm(norm(H), bytes=True)

        # Apply background color
        colors[~np.isfinite(H), :] = background_color

        # Save image
        if image_filename is None:
            image_fname = self.get_image_filename()
        else:
            image_fname = image_filename

        plt.imsave(image_fname, colors, format='png')
        self.current_image_file_idx += 1

    def count_data_files(self):
        # Count the number of files of the form BHIME_i.npz or BHIME_i.npy in
        # the data directory
        return len(glob.glob(os.path.join(self.data_dir, 'BHIME_*.npz')))

    def count_image_files(self):
        return len(glob.glob(os.path.join(self.images_dir, 'Image-*.png')))

    def count_histogram_files(self):
        return len(
            glob.glob(os.path.join(self.histogram_dir, 'Histogram-*.npz')))

    def get_data_filename(self, i):
        return os.path.join(self.data_dir, 'BHIME_{}.npz'.format(i))

    def get_data_files_generator(self):
        for i in range(self.current_data_file_idx):
            yield self.get_data_filename(i)

    def get_histogram_filename(self):

        return os.path.join(self.histogram_dir, 'Histogram-{}.npz'.format(
            self.current_histogram_file_idx + 1))

    def get_image_filename(self):
        return os.path.join(self.images_dir, 'Image-{}.png'.format(
            self.current_image_file_idx + 1))

    def load_data(self, file_name):
        return np.load(file_name)

    def save_data(self, L, C, data_filename):
        np.savez_compressed(data_filename, L=L, C=C)

    def load_histogram(self, file_name):
        return np.load(file_name)['H']

    def save_histogram(self, H, histogram_filename):
        np.savez_compressed(histogram_filename, H=H)

    def compute_eig_batch(self,
                          A_batch,
                          use_single_precision):
        """

        :param A_batch: A numpy.ndarray of dimension (M, N, N)
        :param use_single_precision: bool
        :return:
        """
        if use_single_precision:
            return np.linalg.eigvals(A_batch).astype(np.complex64)
        else:
            return np.linalg.eigvals(A_batch)

    def get_width(self, height, real_min, real_max, imag_min, imag_max):
        heightI = imag_max - imag_min
        widthI = real_max - real_min
        width = np.int64(np.floor(widthI * height / heightI))
        return width

    def one_file_histogram(self, file_name, height, width, axis_range,
                           symmetry_real, symmetry_imag, ignore_real,
                           ignore_real_tol):

        # Load eigenvalues and counts
        data = self.load_data(file_name)
        L = data['L']
        C = data['C']

        # If ignore_real is True, remove real eigenvalues.
        # An eigenvalue, lambda, is considered real if
        # abs(Im(lambda)) < ignore_real_tol
        if ignore_real:
            idx = np.abs(np.imag(L)) >= ignore_real_tol
            L = L[idx]
            C = C[idx]

        # a + bi
        H = self.histogram(L, C, bins=[height, width], axis_range=axis_range)

        if symmetry_real:
            # a - bi
            H += self.histogram(L.real - 1j * L.imag, C, bins=[height, width],
                                axis_range=axis_range)

        if symmetry_imag:
            # -a + bi
            H += self.histogram(-L.real + 1j * L.imag, C, bins=[height, width],
                                axis_range=axis_range)

        if symmetry_real and symmetry_imag:
            # -a - bi
            H += self.histogram(-L.real - 1j * L.imag, C, bins=[height, width],
                                axis_range=axis_range)

        return np.flipud(H)

    # bins = [height, width]
    # range = [[xmin, xmax], [ymin, ymax]]
    def histogram(self, L, C, bins=None, axis_range=None):

        H, _, _ = np.histogram2d(L.imag,
                                 L.real,
                                 bins=bins,
                                 range=[axis_range[1], axis_range[0]],
                                 normed=False,
                                 weights=C)
        return H.astype(np.uint64)
