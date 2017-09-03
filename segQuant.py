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
    using Felix Buggenthins segmented brightfield images
    see
    "An automatic method for robust and fast cell detection in bright field images from high-throughput microscopy"
    http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-297
    """

    def __init__(self, the_movie, SEG_WL='w00', fileExtension='png'):
        """Constructor for SegmentationReaderFelix(SegmentationReader)"""
        super().__init__()
        self.movie = the_movie
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
            # yield None  # DONT EVEN RETURN NONE just terminate the generator
            return   # funny: even a return "BLA" wouldnt give a return value, python detects its an generator and only return yield statements

        STATS = skimage.measure.regionprops(skimage.measure.label(segImg))  # different than matlab: regionprops doesnt take the raw image but the labeled one

        minsize, eccfilter = 25, 0.9  # filter out some dirt
        for obj in filter(lambda x: x.area>minsize and x.eccentricity < eccfilter, STATS):

            y, x = obj.centroid  # centroid is (row, column) !! row is usually the y coordinate. also note that (0,0) is in the upper left corner!
            t = timepoint
            p = position
            area = obj.area
            coordinates = obj.coords

            yield segmentedObject(relX=x, relY=y, timepoint=t, position=p, area=area, coords=coordinates)


class FluorescenceQuantifier(object):

    def __init__(self, the_movie, normalizer):
        self.movie = the_movie

        # nice trick, so that every call to quantify can use cached results
        self.normalizer = normalizer if normalizer is not None else MSch_Normalizer()

    def quantify(self, segObject, WL):
        """
        given the segmented object, get its fluorescence
        :param segObject:
        :param WL:
        :return:
        """
        img = self.movie.loadimage(segObject.position, segObject.timepoint,
                                   WL, 'png', normalizer=self.normalizer)

        if img is None:  # file does not exist
            return None
        else:
            sum_int = np.sum(img[segObject.coords[:,0], segObject.coords[:,1]])
            return sum_int
