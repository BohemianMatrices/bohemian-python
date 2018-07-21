import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from fast_histogram import histogram2d

# class Bohemian:
#     
#     def __init__(g, working_dir = None):
#         
#         self.g = g
#         # Counter for the number of matrices whose eigenvalues have been computed
#         self.total_matrices_computed = 0

#         self.matrix_size = 0
#     
#     def compute_eigenvalues(num_matrices = 10**6,
#                             batch_size = 'AUTO',
#                             matrices_per_file = 'AUTO'):
#         # Once the matrices_per_file option is set (i.e. once this function
#         # is called once) it can't be changed.


def compute_eig_batch(g, batch_size, use_single_precision):
        A_batch = g(num_matrices = batch_size)
        if use_single_precision:
            return np.linalg.eigvals(A_batch).astype(np.complex64)
        else:
            return np.linalg.eigvals(A_batch)


def compute_eigenvalues(g,
                        num_matrices = 10**6,
                        batch_size = 10**5,
                        matrices_per_file = 10**6,
                        file_start_idx = 0,
                        use_single_precision = True,
                        data_dir = 'Data',
                        compress_data = False,
                        verbose = 0,
                        n_jobs = 1):

    num_batches = matrices_per_file//batch_size
    num_files = num_matrices//matrices_per_file

    for file_idx in range(file_start_idx, num_files):
        
        compute_start_time = time.time()
        
        if verbose > 0: print('File {} of {}'.format(file_idx+1, num_files))
        
        if verbose > 2: print('Computing eigenvalues of {} matrices'.format(matrices_per_file))
            
        L = np.array(Parallel(n_jobs=n_jobs)(delayed(compute_eig_batch)(g, batch_size, use_single_precision) for _ in range(num_batches)))
        L = L.flatten()
        
        compute_time = time.time() - compute_start_time
        
        if verbose > 2:
            print('\tDone computing eigenvalues')
            print('\tSaving eigenvalues to {}/BHIME_{}.npz'.format(data_dir, file_idx))

        save_start_time = time.time()
        
        if compress_data:
            fname = '{}/BHIME_{}.npz'.format(data_dir, file_idx)
            np.savez_compressed(fname, L = L)
            
            #f = gzip.GzipFile('{}/BHIME_{}.npy.gz'.format(data_dir, file_idx), 'w')
            #np.save(file=f, arr=L)
            #f.close()
        else:
            np.save('{}/BHIME_{}.npy'.format(data_dir, file_idx), L)

        save_time = time.time() - save_start_time

        if verbose > 2:
            print('\tDone saving eigenvalues')
                              
        if verbose > 1:
            print('\tComputation of eigenvalues of {0:d} matrices took {1:.3f} seconds'.format(matrices_per_file, compute_time))
            print('\tSave took {0:.3f} seconds'.format(save_time))
            print('\t{0:.2f} matrices/second'.format(matrices_per_file/(compute_time+save_time)))
            # print('\t{0:.3f} eigenvalues/second'.format(L.shape//matrices_per_file)
        
        # Free memory
        del L


def one_file_histogram(file_name, compress_data, height, width, axisrange, symmetry_real, symmetry_imag):
    if compress_data:
        L = np.load(file_name)['L']
        #f = gzip.GzipFile('{}/BHIME_{}.npy.gz'.format(data_dir, file_idx), 'r')
        #L = np.load(f)
        #f.close()
    else:
        L = np.load(file_name)
    
    # a + bi
    H_file = histogram2d(L.imag, L.real, bins=[height, width], range = [axisrange[1], axisrange[0]])
    H_file = H_file.astype(np.uint32)
    H = H_file
    
    if symmetry_real:
        # a - bi
        H_file = histogram2d(-L.imag, L.real, bins=[height, width], range = [axisrange[1], axisrange[0]])
        H_file = H_file.astype(np.uint32)
        H = H + H_file
    
    if symmetry_imag:
        # -a + bi
        H_file = histogram2d(L.imag, -L.real, bins=[height, width], range = [axisrange[1], axisrange[0]])
        H_file = H_file.astype(np.uint32)
        H = H + H_file
        
    if symmetry_real and symmetry_imag:
        # -a - bi
        H_file = histogram2d(-L.imag, -L.real, bins=[height, width], range = [axisrange[1], axisrange[0]])
        H_file = H_file.astype(np.uint32)
        H = H + H_file
    
    return np.flipud(H)


def get_width(height, real_min, real_max, imag_min, imag_max):
    heightI = imag_max - imag_min
    widthI = real_max - real_min;
    width = np.int64(np.floor(widthI*height/heightI))
    return width


def compute_histogram(histogram_file = 'Histogram.npy',
                      data_dir = 'Data_Tridiagonal',
                      height = 1001,
                      axisrange = [[-1, 1], [-1, 1]],
                      compress_data = False,
                      symmetry_imag = False,
                      symmetry_real = False,
                      num_files = 1,
                      n_jobs = 1,
                      verbose = 0):
    """
    symmetry_imag: Symmetry across the imaginary axis. i.e. if a + bi is an
                   eigenvalue, -a + bi is also included in the image
    symmetry_real: Symmetry across the real axis. i.e. if a + bi is an 
                   eigenvalue, a - bi is also included in the image
    If both symmetry_imag and symmetry_real are True, then each eigenvalue is
    counted 4 times. i.e. if a + bi is an eigenvalue, -a + bi, a - bi, and
    -a - bi are included in the image.
    """
    
    width = get_width(height, axisrange[0][0], axisrange[0][1], axisrange[1][0], axisrange[1][1])
    
    H = np.zeros((int(height), int(width)), dtype = np.uint32)
    
    # File names
    if compress_data:
        file_names = ['{}/BHIME_{}.npz'.format(data_dir, i) for i in range(num_files)]
    else:
        file_names = ['{}/BHIME_{}.npy'.format(data_dir, i) for i in range(num_files)]
    
    # Serial
    for file_name in file_names:
        if verbose > 0: print(file_name)
        H = H + one_file_histogram(file_name, compress_data, height, width, axisrange, symmetry_real, symmetry_imag)
    
    # Load files in parallel
    # H = np.array(Parallel(n_jobs=n_jobs)(delayed(one_file_histogram)(file_name, compress_data, height, width, axisrange, symmetry_real, symmetry_imag) for file_name in file_names))
    # H = np.sum(H, axis=0)
    
    np.save(histogram_file, H)
    


def plot_histogram(histogram_file=None,
                   background_color=(0, 0, 0, 255),
                   cm=plt.cm.jet,
                   image_dir='Images',
                   image_name='Image_1.png',
                   histogram_map=lambda x: np.log(x+1)):
    
    H = np.load(histogram_file)
    
    H = histogram_map(H)
    
    # Apply colormap
    norm = plt.Normalize()
    colors = cm(norm(H), bytes=True)

    # Apply background color
    colors[H == 0,:] = background_color

    # Save image
    plt.imsave('{}/{}'.format(image_dir, image_name), colors, format='png')

