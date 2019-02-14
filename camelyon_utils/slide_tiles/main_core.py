import os
import h5py

import openslide
from camelyon_utils.slide_tiles.preprocessing import *
from camelyon_utils.slide_tiles.resolution_utilities import *


def __push_to_hsd5__(
        result,
        coordinates,
        coordinates_relative,
        mask_tiles,
        save_mask=True,
        save_coordinates=True,
        dset_img=None,
        dset_mask=None,
        dset_coord=None,
        dset_coord_relative=None
):
    # TODO: NONE MAKES NO SENSE HERE !
    result = np.array(result)
    print('Push ' + str(result.shape[1]) + ' tiles to the HDF-File.')

    shape_before = dset_img.shape[1]
    
    dset_img.resize(shape_before + result.shape[1], axis=1)
    dset_img[:, shape_before:] = result
    
    if save_mask:
        mask_tiles = np.array(mask_tiles, dtype=bool)
        shape_before = dset_mask.shape[1]
        dset_mask.resize(shape_before + mask_tiles.shape[1], axis=1)
        dset_mask[:, shape_before:] = mask_tiles
    
    if save_coordinates:
        coordinates = np.array(coordinates)
        shape_before = dset_coord.shape[1]
        dset_coord.resize(shape_before + coordinates.shape[1], axis=1)
        dset_coord[:, shape_before:] = coordinates
        
        coordinates_relative = np.array(coordinates_relative)
        shape_before = dset_coord_relative.shape[1]
        dset_coord_relative.resize(shape_before + coordinates_relative.shape[1], axis=1)
        dset_coord_relative[:, shape_before:] = coordinates_relative


def convert_openslidefile_to_hdf5(
        path_to_tif,
        mask_handler,
        path_to_output_hdf5,
        wanted_pixel_spacings,
        size, stepsize,
        num_of_epoch_to_save=100,
        maximal_number_of_epochs=None,
        save_mask=True
):
    # Open the file with openslide
    oslide = openslide.OpenSlide(path_to_tif)

    # get the pixel spacing of the scanner from this file
    ps_of_scanner = get_ps_of_openslidefile(oslide)

    mask_handler.preprocess(oslide, ps_of_scanner)
    bbox_of_mask = mask_handler.get_bbox()
    bbox_of_mask = rescale_variable(
        bbox_of_mask,
        mask_handler.get_used_pixel_spacing(),
        wanted_pixel_spacings[0],
        False
    )

    # best layers to use openslide for the wanted pixel spacings
    best_levels = [find_best_level_for_given_ps(ps_of_scanner, i) for i in wanted_pixel_spacings]
    print(best_levels)
    print(ps_of_scanner)

    # creating the HDF5-FILE
    hdf5_file = h5py.File(path_to_output_hdf5, mode='w')

    dset_slidename = hdf5_file.create_dataset(
        'slide_name',
        (1,),
        h5py.special_dtype(vlen=str)
    )
    dset_slidename[0] = os.path.split(path_to_tif)[1]
    dset_img = hdf5_file.create_dataset(
        'img',
        (len(wanted_pixel_spacings), 0, size[0], size[1], 3),
        np.uint8,
        maxshape=(len(wanted_pixel_spacings), None, size[0], size[1], 3)
    )
    dset_mask = hdf5_file.create_dataset(
        'mask',
        (len(wanted_pixel_spacings), 0, size[0], size[1]),
        np.bool,
        maxshape=(len(wanted_pixel_spacings), None, size[0], size[1])
    )
    dset_coord = hdf5_file.create_dataset(
        'coord_real',
        (len(wanted_pixel_spacings), 0, 2),
        np.uint64,
        maxshape=(len(wanted_pixel_spacings), None, 2)
    )
    dset_coord_relative = hdf5_file.create_dataset(
        'coord_relative',
        (len(wanted_pixel_spacings), 0, 2),
        np.uint64,
        maxshape=(len(wanted_pixel_spacings), None, 2)
    )

    result = [[] for _ in wanted_pixel_spacings]
    coordinates = [[] for _ in wanted_pixel_spacings]
    mask_tiles = [[] for _ in wanted_pixel_spacings]
    coordinates_relative = [[] for _ in wanted_pixel_spacings]

    # Here we iterate to all possible X, Y, Coordinates for all wanted pixel spacings.
    for x in range(int(bbox_of_mask[0])-stepsize[0], int(bbox_of_mask[2]+1)+2*stepsize[0], stepsize[0]):
    
        for y in range(int(bbox_of_mask[1])-stepsize[1], int(bbox_of_mask[3]+1)+2*stepsize[1], stepsize[1]):
            
            is_valid = None
            
            for i in range(len(wanted_pixel_spacings)):
                if (is_valid is None or is_valid) and (maximal_number_of_epochs is None or maximal_number_of_epochs > 0):
                    # calculate by the given pixel spacing.
                    pos_on_wanted_ps = np.array([x, y], dtype=np.float64)

                    if i > 0:
                        pos_on_wanted_ps *= wanted_pixel_spacings[0] / wanted_pixel_spacings[i]

                    # the position is right now the center of the image, next we calculate the upper left corner
                    pos_on_wanted_ps -= size / 2.0

                    tilemask, is_valid_current_layer = mask_handler.get_tilemask(
                        np.copy(pos_on_wanted_ps),
                        size,
                        wanted_pixel_spacings[i]
                    )

                    # only the first pixel spacing can use this
                    if is_valid is None:
                        is_valid = is_valid_current_layer

                    if is_valid:
                        
                        pos_on_l0 = np.copy(pos_on_wanted_ps)
                        pos_on_l0 = rescale_variable(pos_on_l0, wanted_pixel_spacings[i], ps_of_scanner[0], True)

                        size_to_optimal_layer = np.copy(size)
                        size_to_optimal_layer = rescale_variable(
                            size_to_optimal_layer,
                            wanted_pixel_spacings[i],
                            ps_of_scanner[best_levels[i]],
                            True
                        )
                         
                        # print(i,size_to_optimal_layer)
                        img = np.array(
                            oslide.read_region(pos_on_l0, best_levels[i], size_to_optimal_layer),
                            dtype=float
                        )[:, :, :3]
                        img = resize(img, size, anti_aliasing=False, mode='reflect')
                        img = np.round(img)
                        img = np.array(img, dtype=np.uint8)
                        
                        result[i].append(img)
                        
                        if save_mask:
                            mask_tiles[i].append(tilemask)    
                        
                        coordinates[i].append(pos_on_l0)
                        coordinates_relative[i].append(np.array(pos_on_wanted_ps))

            if len(result[0]) >= num_of_epoch_to_save:
                # TODO: TRUE;TRUE-VALUES MAKE NO SENSE!
                __push_to_hsd5__(
                    result,
                    coordinates,
                    coordinates_relative,
                    mask_tiles,
                    True,
                    True,
                    dset_img,
                    dset_mask,
                    dset_coord,
                    dset_coord_relative
                )
                result = [[] for _ in wanted_pixel_spacings]
                coordinates = [[] for _ in wanted_pixel_spacings]
                mask_tiles = [[] for _ in wanted_pixel_spacings]
                coordinates_relative = [[] for _ in wanted_pixel_spacings]
                
                # JUST FOR TESTING!
                if maximal_number_of_epochs is not None:
                    maximal_number_of_epochs -= 1
                
    if len(result[0]) > 0:
        # TODO: TRUE;TRUE-VALUES MAKE NO SENSE!
        __push_to_hsd5__(
            result,
            coordinates,
            coordinates_relative,
            mask_tiles,
            True,
            True,
            dset_img,
            dset_mask,
            dset_coord,
            dset_coord_relative
        )

    hdf5_file.close()
