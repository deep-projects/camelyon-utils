import h5py
import os
from argparse import ArgumentParser
import numpy as np
from time import time

DESCRIPTION = 'Merge single hdf5-files into one.'


def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        '--indir', action='store', type=str, metavar='PATH',
        help='Directory PATH containing all HDF5 files with tiles as created by the slide-tiles subcommand.'
    )
    parser.add_argument(
        '--outdir', action='store', type=str, metavar='PATH',
        help='Directory PATH to store merged.hdf5 file inside.'
    )
    parser.add_argument(
        '--numtiles', action='store', type=str, metavar='MASKFILE',
        help='number of tiles per slide for single mode'
    )


    args = parser.parse_args()
    
    
    HDF5_DIR = args.indir
    OUTFILE_PATH = 'merged.hdf5'
    if args.outdir:
        OUTFILE_PATH = os.path.join(args.outdir, OUTFILE_PATH)
    NUM = int(args.numtiles) if args.numtiles else 3

    file_list = []
    for file in os.listdir(HDF5_DIR):
        if file.endswith('.hdf5'):
            file_list.append(os.path.join(HDF5_DIR, file))
    file_list.sort()

    start = time()

    files_tumor = []
    files_normal = []
    for f in file_list:
        if 'umor' in f:
            files_tumor.append(f)
        if 'ormal' in f:
            files_normal.append(f)
    f_t_train = files_tumor[0::2]
    f_n_train = files_normal[0::2]
    f_t_valid = files_tumor[1::2]
    f_n_valid = files_normal[1::2]

    h5_t_train = []
    h5_n_train = []
    h5_t_val = []
    h5_n_val = []
    h5_t_train_mask = []
    h5_n_train_mask = []
    h5_t_val_mask = []
    h5_n_val_mask = []
    for j in f_t_train:
        h5_t_train.append(h5py.File(j, 'r', libver='latest', swmr=True)['img'])
        h5_t_train_mask.append(h5py.File(j, 'r', libver='latest', swmr=True)['mask'])
    for j in f_n_train:
        h5_n_train.append(h5py.File(j, 'r', libver='latest', swmr=True)['img'])
        h5_n_train_mask.append(h5py.File(j, 'r', libver='latest', swmr=True)['mask'])
    for j in f_t_valid:
        h5_t_val.append(h5py.File(j, 'r', libver='latest', swmr=True)['img'])
        h5_t_val_mask.append(h5py.File(j, 'r', libver='latest', swmr=True)['mask'])
    for j in f_n_valid:
        h5_n_val.append(h5py.File(j, 'r', libver='latest', swmr=True)['img'])
        h5_n_val_mask.append(h5py.File(j, 'r', libver='latest', swmr=True)['mask'])

    print('creating merged file at: ', OUTFILE_PATH)
    merged_h5 = h5py.File(OUTFILE_PATH, mode='w')
    merged_tumor_dset_train = merged_h5.create_dataset(name='tumor_train', shape=(len(f_t_train)*NUM, 512, 512, 4), dtype=np.uint8)
    merged_normal_dset_train = merged_h5.create_dataset(name='normal_train', shape=(len(f_n_train)*NUM, 512, 512, 4), dtype=np.uint8)
    merged_tumor_dset_val = merged_h5.create_dataset(name='tumor_valid', shape=(len(f_t_valid)*NUM, 512, 512, 4), dtype=np.uint8)
    merged_normal_dset_val = merged_h5.create_dataset(name='normal_valid', shape=(len(f_n_valid)*NUM, 512, 512, 4), dtype=np.uint8)

    for i in range(NUM):
        print('run ', i, 'of ', NUM)
        new_t_train = np.ndarray((len(f_t_train),512,512,4))
        new_n_train = np.ndarray((len(f_n_train),512,512,4))
        new_t_val = np.ndarray((len(f_t_valid),512,512,4))
        new_n_val = np.ndarray((len(f_n_valid),512,512,4))

        for j in range(len(h5_t_train)):
            if h5_t_train[j].shape[1] >= 1:
                new_t_train[j,:,:,0:-1] = h5_t_train[j][0,np.random.randint(0,h5_t_train[j].shape[1])]
                new_t_train[j,:,:,-1] = h5_t_train_mask[j][0,np.random.randint(0,h5_t_train[j].shape[1])]
            else:
                new_t_train[j,:,:,0:-1] = new_t_train[np.random.randint(0,len(new_t_train)),:,:,0:-1]
                new_t_train[j,:,:,-1] = new_t_train[np.random.randint(0,len(new_t_train)),:,:,-1]
        merged_tumor_dset_train[i*len(f_t_train):(i+1)*len(f_t_train)] = new_t_train

        for j in range(len(h5_n_train)):
            if h5_n_train[j].shape[1] >= 1:
                new_n_train[j,:,:,0:-1] = h5_n_train[j][0,np.random.randint(0,h5_n_train[j].shape[1])]
                new_n_train[j,:,:,-1] = h5_n_train_mask[j][0,np.random.randint(0,h5_n_train[j].shape[1])]
            else:
                new_n_train[j,:,:,0:-1] = new_n_train[np.random.randint(0,len(new_n_train)),:,:,0:-1]
                new_n_train[j,:,:,-1] = new_n_train[np.random.randint(0,len(new_n_train)),:,:,-1]
        merged_normal_dset_train[i*len(f_n_train):(i+1)*len(f_n_train)] = new_n_train

        for j in range(len(h5_t_val)):
            if h5_t_val[j].shape[1] >= 1:
                new_t_val[j,:,:,0:-1] = h5_t_val[j][0,np.random.randint(0,h5_t_val[j].shape[1])]
                new_t_val[j,:,:,-1] = h5_t_val_mask[j][0,np.random.randint(0,h5_t_val[j].shape[1])]
            else:
                new_t_val[j,:,:,0:-1] = new_t_val[np.random.randint(0,len(new_t_val)),:,:,0:-1]
                new_t_val[j,:,:,-1] = new_t_val[np.random.randint(0,len(new_t_val)),:,:,-1]
        merged_tumor_dset_val[i*len(f_t_valid):(i+1)*len(f_t_valid)] = new_t_val

        for j in range(len(h5_n_val)):
            if h5_n_val[j].shape[1] >= 1:
                new_n_val[j,:,:,0:-1] = h5_n_val[j][0,np.random.randint(0,h5_n_val[j].shape[1])]
                new_n_val[j,:,:,-1] = h5_n_val_mask[j][0,np.random.randint(0,h5_n_val[j].shape[1])]
            else:
                new_n_val[j,:,:,0:-1] = new_n_val[np.random.randint(0,len(new_n_val)),:,:,0:-1]
                new_n_val[j,:,:,-1] = new_n_val[np.random.randint(0,len(new_n_val)),:,:,-1]
        merged_normal_dset_val[i*len(f_n_valid):(i+1)*len(f_n_valid)] = new_n_val

    merged_h5.close()

    end = time()
    print('duration (secs):', end-start)
