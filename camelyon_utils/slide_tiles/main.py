from argparse import ArgumentParser
from camelyon_utils.slide_tiles.main_core import *

DESCRIPTION = 'Create image tiles from a single tif slide.'


def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        'slide', action='store', type=str, metavar='SLIDEFILE',
        help='Path to SLIDEFILE in  OpenSlide tif format.'
    )
    parser.add_argument(
        '--mask', action='store', type=str, metavar='MASKFILE',
        help='Path to MASKFILE in CAMELYON XML format.'
    )
    parser.add_argument(
        '--second-scanner', action='store_true',
        help='Slides of the second scanner have different zoom levels, which will be set if this parameter is true.'
    )

    args = parser.parse_args()

    # PS OF SCANNER 2:
    # [0.00022727 0.00045454 0.00090909 0.00181818 0.00363636 0.00727273 0.0145455  0.0290909  0.0581818 ]

    # wanted_pixel_spacings = [0.00025, 0.0005, 0.001]
    wanted_pixel_spacings = [0.00025]
    if args.second_scanner:
        # wanted_pixel_spacings = [0.00022727, 0.00045454, 0.00090909]
        wanted_pixel_spacings = [0.00022727]

    size = np.array((512, 512))
    stepsize = np.array((512, 512))

    path_to_output_hdf5 = 'tiles.hdf5'
    num_of_epoch_to_save = 100
    level_to_create_otsu = 4
    
    maximal_number_of_epochs = None

    slide_path = os.path.expanduser(args.slide)

    if args.mask is None:
        handler = BinaryMaskHandler(level_to_create_the_otsu_mask=level_to_create_otsu)
    else:
        handler = XMLMaskHandler(args.mask)

    convert_openslidefile_to_hdf5(
        slide_path,
        handler,
        path_to_output_hdf5,
        wanted_pixel_spacings,
        size,
        stepsize,
        maximal_number_of_epochs=maximal_number_of_epochs,
        num_of_epoch_to_save=num_of_epoch_to_save
    )
