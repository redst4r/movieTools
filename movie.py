import tttTools
import pandas as pd
import skimage.measure
from collections import namedtuple
from imageNormalizer import MSch_Normalizer, NoNormalizer, __bit_normalize__
import numpy as np
import os
import re


class Movie(object):
    """an entire time lapse experiment """

    def __init__(self, movieID, verbose=False):
        """Constructor for Movie"""
        self.movieID = movieID
        self.TTTDIR = tttTools.getTTTDir()
        self.verbose = verbose

    def get_all_positions(self):
        """
        determines all available positions (parsing the folders names)
        :return: dict  int->abs.folderName
        """
        movieID = self.movieID
        movieFolder = os.path.join(self.TTTDIR, movieID)

        positions = {}  # a dict of int-> absolute positionFilename
        pattern = re.compile(r"%s_p(\d{4})$" % (movieID))
        for f in os.listdir(movieFolder):
            m = pattern.match(f)
            if m:
                positions[(int(m.group(1)))] = os.path.join(movieFolder,f)

        return positions

    def get_all_images(self):
        """
        retrieves all images from the movie, across all timepoints, positions
        :return: dict with (position, timepoint, WL) -> filename of the image
        """
        positions = self.get_all_positions()
        images = {}  # dict with (position, time, WL) -> filename
        for p, pDir in positions.items():
            # get the images present
            pattern = re.compile(r"%s_p%.04d_t(\d{5})_z001_w(\d+).png$"%(self.movieID,p))
            for f in os.listdir(pDir):
                m =pattern.match(f)
                if m:
                    t, wl = int(m.group(1)), int(m.group(2))
                    images[(p,t,wl)] = os.path.join(pDir, f)
        return images

    def createPositionFoldername(self, position):
        """
        returns the folder corresponding to requested position
        :param position: int
        :return: str
        """
        return '{dir}/{movie}/{movie}_p{pos:04d}/'.format(dir=self.TTTDIR, movie=self.movieID, pos=position)

    def createTTTfilename(self, position, timepoint, WL, extension, SEGFLAG=False):
        """
        creates the name of the image file, correspoding to position and timepoint
        :param position:
        :param timepoint:
        :param WL:
        :param extension: file extension: {png, jpg}
        :param SEGFLAG: do we want the segmentation image or just the bright/fluor
        :return:
        """
        if not SEGFLAG:
            filename = '{dir}/{movie}/{movie}_p{pos:04d}/{movie}_p{pos:04d}_t{time:05d}_z001_{WL}.{ext}'.format(
                    dir=self.TTTDIR, movie=self.movieID, pos=position, time=timepoint,WL=WL, ext=extension)
        else:
            filename = '{dir}/{movie}/segmentation/{movie}_p{pos:04d}/{movie}_p{pos:04d}_t{time:05d}_z001_{WL}.{ext}'.format(
                    dir=self.TTTDIR, movie=self.movieID, pos=position, time=timepoint,WL=WL, ext=extension)

        return filename

    def createTATexpfilename(self):
        """
        returns the filename of the TATxml file
        :return: str
        """
        return '{dir}/{movie}/{movie}_TATexp.xml'.format(dir=self.TTTDIR, movie=self.movieID)

    def loadimage(self, position, timepoint, WL, extension, normalizer):
        """
        loads an image of the position/timepoint using the given normalization
        :param position:
        :param timepoint:
        :param WL:
        :param extension: usually 'png' or 'jpg'
        :param normalizer: instance of a imageNormalizer.ImageNormalizer
        :return: np.array of the normalized image
        """
        filename = self.createTTTfilename(position, timepoint, WL, extension, SEGFLAG=False)
        if self.verbose:
            print("loading: %s" % filename)

        if os.path.isfile(filename):
            return normalizer.normalize(filename)
        else:
            if self.verbose:
                print("file %s does not exist, returning None" % filename)
            return None

    def load_segmentation_image(self, position, timepoint, WL, extension):
        """
        loads the segmentation of the requested timepoint and position
        :param position:
        :param timepoint:
        :param WL:
        :param extension:
        :return: seg image, np.array
        """
        fname = self.createTTTfilename(position, timepoint, WL, extension,SEGFLAG=True)
        if self.verbose:
            print("loading segmentation: %s" % fname)

        # create the NoNormalizer on the fly, which just loads the plain image
        img_SEG = NoNormalizer().normalize(fname) #  # this loads the image as it is, ie no bit conversion!
        img_SEG =  __bit_normalize__(img_SEG)  # explicitly do the bit conversion, such that 255 -> 1

        if not np.all(np.logical_or(img_SEG == True, img_SEG == False)):  # make sure its a binary mask
            raise ValueError('segmentation image is not binary!')

        return img_SEG

        # TODO exception if no ssegmentaion is available

    def get_segmented_objects(self, position, timepoint, FluorWL, SEG_WL):
        """
        Inputs:
        :param position:          position of interest
        :param timepoint          the timepoint we're interested in
        :param FluorWL            what wavelength to quantify
        :param SEG_WL             extension of the segmenation images (e.g 'w01.png')
         Outputs:
           objects_df                 - pandas.Dataframe containing all the segmented objects with their properties
        """
        raise Exception('method is deprecated and cannot be used any more. Use segQuant.SegmentationReader')
        extension = 'png'

        try:
            segImg = self.load_segmentation_image(position, timepoint, SEG_WL, extension=extension)
        except FileNotFoundError as e:
            # return emtpy dataframe if no segmetnation available
            print("skipping this position because no segmentation image found: %s"%str(e))
            return pd.DataFrame()

        # TODO: using michiSch normalization always
        fluorImg = self.loadimage(position, timepoint, FluorWL, extension=extension, normalizer=MSch_Normalizer())

        STATS = skimage.measure.regionprops(skimage.measure.label(segImg))  # different than matlab: regionprops doesnt take the raw image but the labeled one

        segObjects = []
        segmentedObject = namedtuple('segObject', "relX relY time position area wavelength")
        minsize, eccfilter = 25, 0.9  # filter out some dirt
        for obj in filter(lambda x: x.area>minsize and x.eccentricity < eccfilter, STATS):

            y, x = obj.centroid # DONE centroid is (row, column) !! row is usually the y coordinate. also ntoe that (0,0) is in the upper left corner!
            t = timepoint
            p = position
            area = obj.area
            wl = np.sum(fluorImg[obj.coords[:,0], obj.coords[:,1]])  # coords is a n X 2 array with pixel coordinates

            segObjects.append(segmentedObject(relX=x, relY=y, time=t, position=p, area=area, wavelength=wl))

        # TODO: absolute coordinates
        # # get absolute coordinates
        # tatfile = tttTools.createTATexpfilename(movieID)
        #
        # # need to "transpose" to get the vectors of relX ...
        # relXV, relYV, timeV, positionV, _, _ = list(zip(*segObjects))  # WARNING: carful here with the order on the left: has to be the same as in def of namedtuple
        # absX, absY = tttTools.mapCoordinates_relative2absolute(relXV,relYV,positionV,tatfile)
        # objects_struct.absX = absX;
        # objects_struct.absY = absY;

        objects_df = pd.DataFrame(segObjects)

        return objects_df
