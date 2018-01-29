
import scipy.io as sio
import numpy as np
import scipy.sparse as sparse
import h5py

'''set of hdf5 format utils.
    Note: Open all the files in appending mode when you are writing data, 
          because often we need to write data for multiple utterances in a single .h5 file
    data -- matirx or vector to be written
    path -- location where you want to store the data
    dataset_name -- data set name to be used for the supplied data. For details https://support.hdfgroup.org/HDF5/
'''
    

def hdf5_write_data_labels(data, labels, path, prefix='train'):
    # for sparse data, writing
    ''' 
        creates two data sets one for data and other for labels. Writes the data in compressed format
    '''
    f =  h5py.File(path, 'a')
    X_dset = f.create_dataset(prefix + '_split', shape=data.shape, dtype='f', fillvalue=0, compression='gzip', compression_opts=9)
    X_dset[:] = data
    Y_dset = f.create_dataset(prefix + '_labels', labels.shape, dtype='i')
    Y_dset[:] = labels
    f.close()


def hdf5_write_compression(data, path, dataset_name):
    ''' Writes the data in compressed format '''
    f =  h5py.File(path, 'a')
    X_dset = f.create_dataset(dataset_name, shape=data.shape, dtype='f', fillvalue=0, compression='gzip', compression_opts=9)
    X_dset[:] = data
    f.close()


def hdf5_write(data, path, dataset_name):
    ''' writes the data in uncompressed format '''
    f =  h5py.File(path, 'a')
    X_dset = f.create_dataset(dataset_name, shape=data.shape, dtype='f')
    X_dset[:] = data
    f.close()

