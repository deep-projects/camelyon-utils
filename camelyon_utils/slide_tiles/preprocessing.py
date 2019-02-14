from camelyon_utils.slide_tiles.resolution_utilities import *
from skimage.filters import threshold_otsu
from skimage.transform import resize
from PIL import Image, ImageDraw


def is_valid_tilemask_simple_threshold(tilemask):
    threshold = 0.1
    
    relation = tilemask[tilemask == True].sum() / tilemask.size

    if relation < threshold:
        return False
    else:
        return True


class MaskHandler:
    def __init__(self, lambda_check_if_tilemask_is_valid):
        self.lambda_check_if_tilemask_is_valid = lambda_check_if_tilemask_is_valid
        self.used_pixel_spacing = None

    def preprocess(self, oslide, ps_from_openslidefile):
        pass
    
    def get_tilemask(self, pos, tile_shape, pixel_spacing):
        pass

    def get_bbox(self):
        pass

    def get_used_pixel_spacing(self):
        return self.used_pixel_spacing


class XMLMaskHandler(MaskHandler):
    def __init__(self, path_to_xmlfile, lambda_check_if_tilemask_is_valid=is_valid_tilemask_simple_threshold):
        super().__init__(lambda_check_if_tilemask_is_valid=lambda_check_if_tilemask_is_valid)
        self.path_to_xmlfile = path_to_xmlfile
        self.parse_single_xml_mask(self.path_to_xmlfile)

    def parse_single_xml_mask(self, input_file):
        """
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
        self.used_pixel_spacing = ps_from_openslidefile[0]

        # TODO: the bbox should be created from the XML file and not only taken the full slide.
        self.bbox = [0, 0, oslide.properties['openslide.level[0].width'], oslide.properties['openslide.level[0].height']]
        self.bbox = np.array(self.bbox, dtype=np.int64)

    def get_bbox(self):
        return self.bbox

    def get_tilemask(self, pos, tile_shape, pixel_spacing):
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
            level_to_create_the_otsu_mask=5,
            lambda_check_if_tilemask_is_valid=is_valid_tilemask_simple_threshold
    ):
        super().__init__(lambda_check_if_tilemask_is_valid = lambda_check_if_tilemask_is_valid)
        self.level_to_create_the_otsu_mask = level_to_create_the_otsu_mask

    def preprocess(self, oslide, ps_from_openslidefile):
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
        """
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
