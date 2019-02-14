from collections import OrderedDict

from camelyon_utils.version import VERSION
from camelyon_utils.cli_modes import cli_modes

from camelyon_utils.slide_tiles.main import main as slide_tiles_main
from camelyon_utils.slide_tiles.main import DESCRIPTION as SLIDE_TILES_DESCRIPTION

from camelyon_utils.merge_hdf5.main import main as merge_hdf5_main
from camelyon_utils.merge_hdf5.main import DESCRIPTION as MERGE_HDF5_DESCRIPTION


SCRIPT_NAME = 'camelyon-utils'
TITLE = 'tools'
DESCRIPTION = 'CAMELYON Utils Copyright (C) 2019  Jonas Annuscheit. This software is distributed under the AGPL-3.0 ' \
              'LICENSE.'
MODES = OrderedDict([
    ('slide-tiles', {'main': slide_tiles_main, 'description': SLIDE_TILES_DESCRIPTION}),
    ('merge-hdf5', {'main': merge_hdf5_main, 'description': MERGE_HDF5_DESCRIPTION})
    
])


def main():
    cli_modes(SCRIPT_NAME, TITLE, DESCRIPTION, MODES, VERSION)
