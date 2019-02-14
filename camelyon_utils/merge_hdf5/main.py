import h5py
import os
from argparse import ArgumentParser

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

    args = parser.parse_args()
    
    
    HDF5_DIR = args.indir
    MERGED_HDF5 = args.outfile


    file_list = []
    for file in os.listdir(HDF5_DIR):
        if file.endswith('.hdf5'):
            file_list.append(os.path.join(HDF5_DIR, file))
    
    file_list.sort()
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
