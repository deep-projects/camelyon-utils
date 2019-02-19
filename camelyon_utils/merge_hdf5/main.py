import h5py
import os
from argparse import ArgumentParser
import numpy as np

DESCRIPTION = 'Merge single hdf5-files into one.'


def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        '--indir', action='store', type=str, metavar='SLIDEFILE',
        help='Path to dir containing all hdf5 files'
    )
    parser.add_argument(
        '--outfile', action='store', type=str, metavar='MASKFILE',
        help='Outputfile destination.'
    )
    parser.add_argument(
        '--mode', action='store', type=str, metavar='MASKFILE',
        help='whether to create single dataset (="single") or one dset for each slide ("each")'
    )

    args = parser.parse_args()
    
    
    HDF5_DIR = args.indir
    MERGED_HDF5 = args.outfile
    MODE = args.mode

    file_list = []
    for file in os.listdir(HDF5_DIR):
        if file.endswith('.hdf5'):
            file_list.append(os.path.join(HDF5_DIR, file))
    file_list.sort()
        
        
        
    if MODE == 'each':
        print('creating merged file at: ', MERGED_HDF5)
        merged_h5 = h5py.File(MERGED_HDF5, mode='w')
        print('creating group tumor')
        merged_tumor_group = merged_h5.create_group('tumor')
        print('creating group normal')
        merged_normal_group = merged_h5.create_group('normal')
        
        tumor_slide_count = 0
        normal_slide_count = 0
        for file in file_list:
            h5 = h5py.File(file, 'r', libver='latest', swmr=True)
            slide_name = h5['slide_name'][0]
            if slide_name == 'slide':
                slide_name = os.path.split(file)[1]
            print('creating subgroup: ', slide_name)
            if 'umor' in slide_name:
                slide = merged_tumor_group.create_group(slide_name)
                tumor_slide_count += 1
                pass
            elif 'ormal' in slide_name:
                slide = merged_normal_group.create_group(slide_name)
                normal_slide_count += 1
                pass
            for key in h5.keys():
                print('creating dataset: ', key)
                slide.create_dataset(name=key, shape=h5[key].shape, data=h5[key])
                pass

        merged_h5.close()
        print('tumor slides merged: ', tumor_slide_count)
        print('normal slides merged: ', normal_slide_count)
        
        
        
    if MODE == 'single':
        files_tumor = []
        files_normal = []
        for f in file_list:
            if 'umor' in f:
                files_tumor.append(f)
            if 'ormal' in f:
                files_normal.append(f)
        f_t_train = files_tumor[1::2]
        f_n_train = files_normal[1::2]
        f_t_valid = files_tumor[0::2]
        f_n_valid = files_normal[0::2]
        
        print('creating merged file at: ', MERGED_HDF5)
        merged_h5 = h5py.File(MERGED_HDF5, mode='w')
        merged_tumor_dset_train = merged_h5.create_dataset(name='tumor_train', shape=(len(f_t_train)*7500, 512, 512, 3), dtype=np.uint8)
        merged_normal_dset_train = merged_h5.create_dataset(name='normal_train', shape=(len(f_n_train)*7500, 512, 512, 3), dtype=np.uint8)
        merged_tumor_dset_val = merged_h5.create_dataset(name='tumor_valid', shape=(len(f_t_train)*7500, 512, 512, 3), dtype=np.uint8)
        merged_normal_dset_val = merged_h5.create_dataset(name='normal_valid', shape=(len(f_n_train)*7500, 512, 512, 3), dtype=np.uint8)

        for i in range(7500):
            print('run ', i, 'of ', 7500)
            new_t_train = np.ndarray((len(f_t_train),512,512,3))
            new_n_train = np.ndarray((len(f_t_train),512,512,3))
            new_t_val = np.ndarray((len(f_t_train),512,512,3))
            new_n_val = np.ndarray((len(f_t_train),512,512,3))
            
            for j in range(len(f_t_train)):
                h5 = h5py.File(f_t_train[j], 'r', libver='latest', swmr=True)
                new_t_train[i*len(f_t_train)+j] = h5['img'][0,np.random.randint(0,h5['img'].shape[1])]
            merged_tumor_dset_train[i*len(f_t_train):(i+1)*len(f_t_train)] = new_t_train
            
            for j in range(len(f_n_train)):
                h5 = h5py.File(f_n_train[j], 'r', libver='latest', swmr=True)
                new_n_train[i*len(f_n_train)+j] = h5['img'][0,np.random.randint(0,h5['img'].shape[1])]
            merged_normal_dset_train[i*len(f_n_train):(i+1)*len(f_n_train)] = new_n_train
            
            for j in range(len(f_t_valid)):
                h5 = h5py.File(f_t_valid[j], 'r', libver='latest', swmr=True)
                new_t_val[i*len(f_t_valid)+j] = h5['img'][0,np.random.randint(0,h5['img'].shape[1])]
            merged_tumor_dset_val[i*len(f_t_valid):(i+1)*len(f_t_valid)] = new_t_val
            
            for j in range(len(f_n_valid)):
                h5 = h5py.File(f_n_valid[j], 'r', libver='latest', swmr=True)
                new_n_val[i*len(f_n_valid)+j] = h5['img'][0,np.random.randint(0,h5['img'].shape[1])]
            merged_normal_dset_val[i*len(f_n_valid):(i+1)*len(f_n_valid)] = new_n_val
                                                   
        merged_h5.close()
