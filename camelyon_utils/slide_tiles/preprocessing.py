from camelyon_utils.slide_tiles.resolution_utilities import *
from skimage.filters import threshold_otsu
from skimage.transform import resize
from PIL import Image, ImageDraw


def is_valid_tilemask(threshold=0.5, percent_of_ignoring_boarder=0.5):
    """This function returns you a function that can check if a tilemask is a valid mask.
       This is done by a threshold.
    
    Args:
        threshold: the threshold must be larger than 0. and smaller or equal than 1..
                   I. e.  if we want that 60% or more of the tile to check are in the mask we need to set this value to 0.6
        percent_of_ignoring_boarder: The precent that should be irgnored for the checking if this is a real mask. This should a value equal or larger than 0 and must be smaller than 1.

    Returns:
        Returns a function that can check the mask
    """

    # If this value is larger or equal to one than return a function that gives always True, so every mask will be a valid mask.
    if percent_of_ignoring_boarder >= 1.:
        return lambda tilemask: True

    if percent_of_ignoring_boarder < 0:
        percent_of_ignoring_boarder = 0
    percent_of_ignoring_boarder /= 2.0

    if threshold > 1:
        threshold = 1

    def is_valid_tilemask_percent_threshold(tilemask):
        """This function can be used to check if a mask is a valid mask and can be saved.
        
        Args:
            tilemask: A 2d-boolean-numpy array that should contain the mask that needs to be checked

        Returns:
            Returns a boolean that is true if its a valid mask.
        """

        ignorepixels_axis0 = int(tilemask.shape[0] * percent_of_ignoring_boarder)
        ignorepixels_axis1 = int(tilemask.shape[1] * percent_of_ignoring_boarder)

        # make sure that minimum of one pixel get checked
        if ignorepixels_axis0 * 2 >= tilemask.shape[0]:
            ignorepixels_axis0 = tilemask.shape[0] // 2
            # if the shape is a even number we take the middle 2 pixels as minimum.
            if tilemask.shape[0] % 2 == 0:
                ignorepixels_axis0 -= 1
            
        if ignorepixels_axis1 * 2 >= tilemask.shape[1]:
            ignorepixels_axis1 = tilemask.shape[1] // 2
            # if the shape is a even number we take the middle 2 pixels as minimum.
            if tilemask.shape[1] % 2 == 0:
                ignorepixels_axis1 -= 1

        # crop the tile to get only the tile that needs to be checked.
        tilemask_to_check = tilemask[ignorepixels_axis0:tilemask.shape[0]-ignorepixels_axis0,ignorepixels_axis1:tilemask.shape[1]-ignorepixels_axis1]
                
        relation = tilemask_to_check[tilemask_to_check == True].sum() / tilemask_to_check.size

        if relation < threshold:
            return False
        else:
            return True

    return is_valid_tilemask_percent_threshold


class MaskHandler:
    """This is a abstract class that describes how a mask handler need to be build.
        A mask handler handles creates or load a mask and know how to get to get a tile from
        the mask at a given position and given pixel spacing.
    
    Args:
        lambda_check_if_tilemask_is_valid: A function that is used to check if a tile is valid.

    """
    def __init__(self, lambda_check_if_tilemask_is_valid):
        """The init function.
        
        Args:
            lambda_check_if_tilemask_is_valid: A function that is used to check if a tile is valid.

        """
        self.lambda_check_if_tilemask_is_valid = lambda_check_if_tilemask_is_valid
        self.used_pixel_spacing = None

    def preprocess(self, oslide, ps_from_openslidefile):
        """ This function is used to make the preprocessing steps to get the mask.
        
        Args:
            oslide: The open openslide file, that is used to read properties.
            ps_from_openslidefile: the pixel spacing of the openslide file.

        """
        pass
    
    def get_tilemask(self, pos, tile_shape, pixel_spacing):
        """This function gets you the mask to a tile.

        Args:
            pos: The position that you want to get back
            tile_shape: the size of the tile
            pixel_spacing: the pixel spacing information

        Returns:
            tilemask: the mask for this position and tile shape.
            is_valid: the lambda function was used to check if this tilemask is a valid.            
        """
        pass

    def get_bbox(self):
        """Get the bounding box that was calculated in the preprocessing step.

        Returns:
            The bounding box
        """
        pass

    def get_used_pixel_spacing(self):
        """This function returns the used pixel spacing.

        Returns:
            the used pixel spacing.
        """
        return self.used_pixel_spacing


class XMLMaskHandler(MaskHandler):
    """ A mask handler handles creates or load a mask and know how to get to get a tile from
        the mask at a given position and given pixel spacing.
        This mask handler use a xml file that is given to read the mask. The XML structure must be in a ASAP format
        that is also used in the camelyon challenge.
    
    Args:
        lambda_check_if_tilemask_is_valid: A function that is used to check if a tile is valid.

    """
    def __init__(self, path_to_xmlfile, lambda_check_if_tilemask_is_valid=is_valid_tilemask()):
        """The init function
        
        Args:
            path_to_xmlfile: The path to the xml file that is used. This must be in the ASAP format that is also used in the camelyon challenge.
            lambda_check_if_tilemask_is_valid: A function that is used to check if a tile is valid.
        """
        super().__init__(lambda_check_if_tilemask_is_valid=lambda_check_if_tilemask_is_valid)
        self.path_to_xmlfile = path_to_xmlfile
        self.parse_single_xml_mask(self.path_to_xmlfile)

    def parse_single_xml_mask(self, input_file):
        """This function parse a single ASAP xml file that is also used to describe the tumor mask in the camelyon 16 / 17 challenge.
        
        Args:
            input_file: The path to the xml file.

        """
        tree = ET.parse(input_file)
        root = tree.getroot()
        total_coords= []
        group_ids = []
        
        for annotation in root.iter('Annotation'):
            
            groupid = int(annotation.attrib['PartOfGroup'] == '_0')
            group_ids.append(groupid)
            
            for coordinates in annotation.iter('Coordinates'):
                
                coords = []
                
                for coord in coordinates.iter('Coordinate'):

                    pos = np.array([coord.get('X'), coord.get('Y')], dtype=np.float64)
                    coords.append(pos)

                if len(coords) >= 3:
                    total_coords.append(np.array(coords,dtype=np.float64))

        self.total_coords = total_coords
        self.group_ids = group_ids

    def preprocess(self, oslide, ps_from_openslidefile):
        """This function preprocess the xml file and set the bbox that need to be checked.
        
        Args:
            oslide: The open openslide file, that is used to read properties.
            ps_from_openslidefile: the pixel spacing of the openslide file.

        """
        self.used_pixel_spacing = ps_from_openslidefile[0]

        # TODO: the bbox should be created from the XML file and not only taken the full slide. With that we could speed up the reading process.
        self.bbox = [0, 0, oslide.properties['openslide.level[0].width'], oslide.properties['openslide.level[0].height']]
        self.bbox = np.array(self.bbox, dtype=np.int64)

    def get_bbox(self):
        """Get the bounding box that was calculated in the preprocessing step.

        Returns:
            The bounding box
        """
        return self.bbox

    def get_tilemask(self, pos, tile_shape, pixel_spacing):
        """This function gets you the mask to a tile.

        Args:
            pos: The position that you want to get back
            tile_shape: the size of the tile
            pixel_spacing: the pixel spacing information

        Returns:
            tilemask: the mask for this position and tile shape.
            is_valid: the lambda function was used to check if this tilemask is a valid.            
        """
        pos = pos + 0.5
        pos = np.array(pos, dtype=np.int64)

        mask_tile = Image.new('L', list(tile_shape), 0)

        for c3, i in zip(self.total_coords, range(len(self.total_coords))):
            c2 = np.copy(c3)
            
            c2 = rescale_variable(c2, self.used_pixel_spacing, pixel_spacing, True)

            c2 -= pos  # - np.int64((size / 2)+0.5) TODO
            c = [(j[0], j[1]) for j in list(c2)]
            
            ImageDraw.Draw(mask_tile).polygon(c, outline=self.group_ids[i], fill=self.group_ids[i])
           
        mask_tile = np.array(mask_tile, dtype=bool)

        return mask_tile, self.lambda_check_if_tilemask_is_valid(mask_tile)


class BinaryMaskHandler(MaskHandler):
    def __init__(
            self,
            level_to_create_the_otsu_mask=4,
            lambda_check_if_tilemask_is_valid=is_valid_tilemask()
    ):
        """ A mask handler handles creates or load a mask and know how to get to get a tile from
            the mask at a given position and given pixel spacing.
            This mask handler use the otsu algorithm to find tissue and use this as mask.
        
        Args:
            level_to_create_the_otsu_mask: the level that should be used to create the otsu mask. 0 is the layer with the highest resolution, but
                                   you need a lot of memory and cpu power to work with this. A good value to work is layedr 4.
            lambda_check_if_tilemask_is_valid: A function that is used to check if a tile is valid.

        """
        super().__init__(lambda_check_if_tilemask_is_valid = lambda_check_if_tilemask_is_valid)
        self.level_to_create_the_otsu_mask = level_to_create_the_otsu_mask

    def preprocess(self, oslide, ps_from_openslidefile):
        """This function runs the tissue segmentation that use the otsu algorithm. See the function 'tissue_segmentation' for more information.
        
        Args:
            oslide: The open openslide file, that is used to read properties.
            ps_from_openslidefile: the pixel spacing of the openslide file.

        """
        self.tissue_segmentation(oslide)
        self.used_pixel_spacing = ps_from_openslidefile[self.level_to_create_the_otsu_mask]

    def tissue_segmentation(self, oslide):
        """A simple version of the otsu algorithm to get a mask of the tissue. This version is optimized for the HE staining, it use the following
        transformation to get a gray image from the color image: Red + Blue - 2*Green. On the gray image the otsu algorithmus is used.
        
        Args:
            oslide: This is a openslide file, which is used to read the pixel spacing information.
                    You can open a openslide file with openslide.OpenSlide('path/to/openslide.tif').
            level: The level to get the slide and do the tissue segmentation. This should be not on a to high resolution, because
                    the full slide get readed into the memory.

        Returns:
            A boolean mask that has the same size like the original slide at this level, that indicates where the algorithm finds tissue segmentation.

        """

        # Get the slide on the level
        shape = oslide.level_dimensions[self.level_to_create_the_otsu_mask]
        img = np.array(oslide.read_region((0, 0), self.level_to_create_the_otsu_mask, shape), np.float32)
       
        # convert image to gray image
        gray_image = img[:, :, 0] + img[:, :, 2] - 2.0 * img[:, :, 1]
        
        # use the otsu algorithm and create the mask
        otsu = threshold_otsu(gray_image[np.isfinite(gray_image)], nbins=256)

        self.mask = gray_image > otsu

    def get_bbox(self):
        """This function gets the bounding box for a boolean mask.
        
        Args:
            mask: The 2d-mask to find the bounding box.

        Returns:
            The start and end coordinates of the bounding box. For Example:

            mask = [[0,0,0,0,0],
                    [0,1,1,1,0],
                    [0,0,1,0,0],
                    [0,0,0,0,0],
                   ]
            would give you the result
                (x_start, y_start, x_end, y_end) = (1, 1, 3, 2)
        """
        x_start = np.argmax(self.mask.sum(axis=0) > 0)
        y_start = np.argmax(self.mask.sum(axis=1) > 0)
        x_end = self.mask.shape[1] - np.argmax(self.mask.sum(axis=0)[::-1] > 0) - 1
        y_end = self.mask.shape[0] - np.argmax(self.mask.sum(axis=1)[::-1] > 0) - 1
        return np.array((x_start, y_start, x_end, y_end))

    def get_tilemask(self, pos, tile_shape, pixel_spacing):
        """This function gets you the mask to a tile.

        Args:
            pos: The position that you want to get back
            tile_shape: the size of the tile
            pixel_spacing: the pixel spacing information

        Returns:
            tilemask: the mask for this position and tile shape.
            is_valid: the lambda function was used to check if this tilemask is a valid.            
        """
        size = np.copy(tile_shape)

        pos = rescale_variable(pos, pixel_spacing, self.used_pixel_spacing, True)
        tile_shape = rescale_variable(tile_shape, pixel_spacing, self.used_pixel_spacing, True)

        tilemask = np.zeros(tile_shape)

        # x and y Positions
        y, x = pos

        # The width of the image
        y_w, x_w = tile_shape
        
        # check for the boarder cases
        left = 0 if x >= 0 else x * -1
        upper = 0 if y >= 0 else y * -1
        right = 0 if x + x_w < self.mask.shape[0] else x+x_w - (self.mask.shape[0]-1)
        down = 0 if y + y_w < self.mask.shape[1] else y+y_w - (self.mask.shape[1]-1)

        if x + x_w - right > 0 and y + y_w - down > 0 and tilemask.shape[0]-right-left > 0 and tilemask.shape[1]-down-upper > 0:
            tilemask[left:tilemask.shape[0]-right, upper:tilemask.shape[1]-down] = self.mask[x+left: x + x_w - right, y+upper : y + y_w - down]

        tilemask = np.array(tilemask,dtype=float)
        tilemask = np.array(resize(tilemask, size, anti_aliasing=False, mode='reflect') >= 0.5, dtype=bool)

        is_valid = self.lambda_check_if_tilemask_is_valid(tilemask)

        return tilemask, is_valid
