# This file contains all helpers to use the pixel spacing information of the openslide tif image.

import xml.etree.ElementTree as ET
import numpy as np


def get_ps_from_openslidefile_version_image_description(oslide):
    """This function reads the pixel spacing each layers from the open slide properties.
    In this function the 'tiff.ImageDescription' is used for this. This version
    is used in the Google Drive version of CAMELYON 16.
    
    Args:
        oslide: This is a openslide file, which is used to read the pixel spacing information.
                You can open a openslide file with openslide.OpenSlide('path/to/openslide.tif').

    Returns:
        A array that have the information of the pixel spacing information for each layer. For example:

                array([0.00022727, 0.00045454, 0.00090909, 0.00181818, 0.00363636,
                        0.00727273, 0.0145455 , 0.0290909 , 0.0581818 ])

    """

    tree = ET.ElementTree(
        ET.fromstring(oslide.properties['tiff.ImageDescription'])
    )
    root = tree.getroot()

    pixel_spaces = np.zeros([oslide.level_count], dtype=np.float64)
    
    for test in root.iter('DataObject'):
        if test.attrib['ObjectType'] == 'PixelDataRepresentation':
            ps = 0.0
            index = -1
            for test2 in test.iter():
                try:
                    if test2.attrib['Name'] == 'PIIM_PIXEL_DATA_REPRESENTATION_NUMBER':
                        index = int(test2.text)
                    elif test2.attrib['Name'] == 'DICOM_PIXEL_SPACING':
                        ps = test2.text.split('"')[1]
                except Exception:
                    pass
            pixel_spaces[index] = ps
    return pixel_spaces


def get_ps_from_openslidefile_version_philips(oslide):
    """This function reads the pixel spacing each layers from the open slide properties.
    In this function the 'philips.PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE['+str(layer_id)+'].DICOM_PIXEL_SPACING' is used for this. This version
    is used in the GigaScience Database of CAMELYON 16.
    
    Args:
        oslide: This is a openslide file, which is used to read the pixel spacing information.
                You can open a openslide file with openslide.OpenSlide('path/to/openslide.tif').

    Returns:
        A array that have the information of the pixel spacing information for each layer. For example:

                array([0.00022727, 0.00045454, 0.00090909, 0.00181818, 0.00363636,
                        0.00727273, 0.0145455 , 0.0290909 , 0.0581818 ])

    """
    pixel_spacing = []
    
    for i in range(oslide.level_count):
        ps = oslide.properties['philips.PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE['+str(i)+'].DICOM_PIXEL_SPACING']
        ps = ps.split('"')[1]
        pixel_spacing.append(float(ps))
        
    return np.array(pixel_spacing)


def get_ps_of_openslidefile(oslide):
    """This function reads the pixel spacing each layers from the open slide properties.
    This function handles both formats for the GigaScience Database as the Google Drive version of the CAMELYON16 dataset.
    
    Args:
        oslide: This is a openslide file, which is used to read the pixel spacing information.
                You can open a openslide file with openslide.OpenSlide('path/to/openslide.tif').

    Returns:
        A array that have the information of the pixel spacing information for each layer. For example:

                array([0.00022727, 0.00045454, 0.00090909, 0.00181818, 0.00363636,
                        0.00727273, 0.0145455 , 0.0290909 , 0.0581818 ])

    """

    if 'philips.PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE[0].DICOM_PIXEL_SPACING' in oslide.properties.keys():
        result = get_ps_from_openslidefile_version_philips(oslide)
        return result
    elif 'tiff.ImageDescription' in oslide.properties.keys():
        result = get_ps_from_openslidefile_version_image_description(oslide)
        # for i in range(1,len(result)):
        #   result[i] = result[0] * float(oslide.properties['openslide.level['+str(i)+'].downsample'])
        return result
    else:
        raise ValueError(
            'This openslide files needs to set following properties "philips.PIIM_PIXEL_DATA_REPRESENTATION_\
            SEQUENCE[0].DICOM_PIXEL_SPACING" for each layer. See here ... to get more information about this error. \
            You are welcome to write a issue to: jonas.annuscheit@htw-berlin.de'
        )


def rescale_variable(variable_to_rescale, initial_ps, indented_ps, round_result = False):
    """This function recalculate the values of an input array from a given pixel spacing to a new pixel spacing.

    Args:
        variable_to_rescale: This value is a single value or an array of integers or floats that are in the initial \
        pixel spacing and need to rescale to the indented pixel spacing.
        initial_ps: A single value that describes the initial pixel spacing that the input value[s] have. Take a look \
        at the function get_ps_of_openslidefile() to get more information.
        indented_ps: The pixel spacing to which the input value[s] should get rescaled. Take a look at the function \
        get_ps_of_openslidefile() to get more information.
        round_result: If True than you will get a rounded result as np.int64. If not you will get a np.float64 as \
        result.

    Returns:
        The rescaled version of input value[s].

    """
    variable_to_rescale = np.float64(variable_to_rescale)
    initial_ps = np.float64(initial_ps)
    indented_ps = np.float64(indented_ps)

    result = variable_to_rescale * (initial_ps / indented_ps)

    if round_result:
        result = np.round(result)
        result = np.int64(result)
        
    return result


def find_best_level_for_given_ps(ps_openslidefile, wanted_ps, use_always_higher_resolution = True):
    """This function gives you the best level to use for your wanted pixel spacing.

    Args:
        ps_openslidefile: The pixel spacing information for all the openslidefile.
        wanted_ps: The wanted pixel spacing for which you want find the best level.
        use_always_higher_resolution: Currently not implemented!

    Returns: The best level for reading the tile to get so close as possible to your wanted pixel spacing
        

    """
    # check for warnings:
    if ps_openslidefile[0] * 0.5 > wanted_ps:
        print(
            'Warning: You want get the best layer for the pixel spacing that is {}. That is a two times higher \
            resolution as the highest resolution from the openslidefile (with ps: {}).'.format(
                wanted_ps,
                ps_openslidefile[0]
            )
        )
    elif ps_openslidefile[-1] * 2 < wanted_ps:
        print(
            'Warning: You want get the best layer for the pixel spacing that is {}. That is a two times lower \
            resolution as the lowest resolution from the openslidefile (with ps: {}).'.format(
                wanted_ps,
                ps_openslidefile[0]
            )
        )

    # compare all pixel spacing information from the openslide with your wanted pixel spacing
    larger_ps = wanted_ps < ps_openslidefile

    # if no ps of the scanner is larger than the wanted, take the last level
    if np.max(larger_ps) == 0: # if no ps of the scanner is larger than the wanted, take the last level
        best_level = len(ps_openslidefile) - 1

    # the wanted pixel spacing is smaller than one or more pixel spacings from the openslidefile
    else:
        best_level = np.argmax(larger_ps) - 1

        # if we want zoom more in as the first layer, we need to take the first
        if best_level < 0:
            best_level = 0
    
    return best_level
