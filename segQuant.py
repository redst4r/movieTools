"""
for segmentation and quantification of the objects
"""
import pandas as pd
import skimage.measure
from collections import namedtuple
import numpy as np
from imageNormalizer import MSch_Normalizer

segmentedObject = namedtuple('segObject', "relX relY timepoint position area coords")

class SegmentationReader(object):
    """
    base class for implementations of a segmenation reader

    """

    def __init__(self, ):
        """Constructor for SegmentationReader"""


class SegmentationReaderFelix(SegmentationReader):
    """
    using felix segmented brightfield images
    """

    def __init__(self, movie, SEG_WL='w00', fileExtension='png'):
        """Constructor for SegmentationReaderFelix(SegmentationReader)"""
        super().__init__()
        self.movie = movie
        self.SEG_WL = SEG_WL
        self.fileExtension = fileExtension

    def iterate_segmented_objects(self, position, timepoint):
        """
        returns a generator over the segmented objects in this image
        :param position:
        :param timepoint:
        :return: iterator over tuples (x,y,time,position,area, coordinates)
        """
        try:
            segImg = self.movie.load_segmentation_image(position, timepoint, self.SEG_WL, extension=self.fileExtension)
        except FileNotFoundError as e:
            # return emtpy dataframe if no segmetnation available #TODO unit test this
            print("skipping this position because no segmentation image found: %s"%str(e))
            raise NotImplementedError('some issue with yield here')
            yield None

        STATS = skimage.measure.regionprops(skimage.measure.label(segImg))  # different than matlab: regionprops doesnt take the raw image but the labeled one

        minsize, eccfilter = 25, 0.9  # filter out some dirt
        for obj in filter(lambda x: x.area>minsize and x.eccentricity < eccfilter, STATS):

            x, y = obj.centroid
            t = timepoint
            p = position
            area = obj.area
            coordinates = obj.coords
            #wl = np.sum(fluorImg[obj.coords[:,0], obj.coords[:,1]])  # coords is a n X 2 array with pixel coordinates

            yield segmentedObject(relX=x, relY=y, timepoint=t, position=p, area=area, coords=coordinates)


class FluorescenceQuantifier(object):

    def __init__(self, movie):
        self.movie = movie
        self.normalizer = MSch_Normalizer()  # nice trick, so that every call to quantify can use cached results
    def quantify(self, segObject, WL):
        """
        given the segmented object, get its fluorescence
        :param segObject:
        :param WL:
        :return:
        """
        try:
            img = self.movie.loadimage( segObject.position, segObject.timepoint, WL, 'png' , normalizer=self.normalizer)
        except FileNotFoundError as e:
            # print("skipping this position because no fluorescence image found: %s"%str(e))
            return None

        sum_int = np.sum(img[segObject.coords[:,0], segObject.coords[:,1]])

        return sum_int




