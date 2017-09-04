import movieTools.tttTools as tttTools
from movieTools.imageNormalizer import MSch_Normalizer, NoNormalizer, _bit_normalize, SLIC_Normalizer
import numpy as np
import os
import re
from movieTools.segQuant import SegmentationReaderFelix, FluorescenceQuantifier


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

    def get_timepoints(self, position):
        "returns a list of timepoints (sorted) of images available at a position"
        all_images_dict= self.get_all_images()
        timepoints = [time for (pos, time, wl), fname in all_images_dict.items() if pos == position]
        timepoints = np.sort(timepoints).tolist()
        return timepoints

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
        img_SEG =  _bit_normalize(img_SEG)  # explicitly do the bit conversion, such that 255 -> 1

        if not np.all(np.logical_or(img_SEG == True, img_SEG == False)):  # make sure its a binary mask
            raise ValueError('segmentation image is not binary!')

        return img_SEG

        # TODO exception if no ssegmentaion is available

    def get_segmented_objects(self, position, timepoint, FluorWL, SEG_WL):
        raise Exception('method is deprecated and cannot be used any more. Use segQuant.SegmentationReader')

    def get_segmented_objects_from_images(self, timepoint, position, patchsize=29, zoomfactor=1):
        """
        retrieves the segmented objects from the images on the disk
        also qunatifies them in w01, w02, w03
        also returns image patches (28x28) and wether these patches had to be padded to get 28x28

        :param timepoint:
        :param position:
        :param patchsize: how big of patches to extract. however, its one to large here! 29 will yield patches of 28x28
        :param zoomfactor: how much to zoom out, e.g. =1 will yield the original imagepatch (patchsize**2),
                           setting it to 0.5 will extract an imagepatch of twice the size
                           but scale it down to patchsize x patchsize
        :return:
        """

        seg_reader = SegmentationReaderFelix(self)

        # we use both MSch and SLIC for fluorescence quantification
        fluor_MSch = FluorescenceQuantifier(self, MSch_Normalizer())
        fluor_SLIC = FluorescenceQuantifier(self, SLIC_Normalizer())

        BFnormalizer = SLIC_Normalizer()  # when loading the brightfield patches

        object_dicts = []
        image_patches = []
        image_patches_padded = []
        for counter, segObject in enumerate(seg_reader.iterate_segmented_objects(position=position, timepoint=timepoint)):
            w1, w2, w3 = [fluor_MSch.quantify(segObject, WL) for WL in
                          ['w01', 'w02', 'w03']]  # will contain None if no FL image present
            w1_SLIC, w2_SLIC, w3_SLIC = [fluor_SLIC.quantify(segObject, WL) for WL in ['w01', 'w02', 'w03']]

            object_dicts.append({'x': segObject.relX, 'y': segObject.relY,
                                 'position': segObject.position,
                                 'timepoint': segObject.timepoint,
                                 'area': segObject.area,
                                 'w01': w1, 'w02': w2, 'w03': w3,
                                 'w01_SLIC': w1_SLIC, 'w02_SLIC': w2_SLIC, 'w03_SLIC': w3_SLIC,
                                 'h5counter': counter,
                                 'uniqueKey': 'p%d_t%d_%d' % (segObject.position, segObject.timepoint, counter)})

            thePatch, wasPadded = self.get_image_patch(segObject.position, segObject.timepoint,
                                                       int(segObject.relX), int(segObject.relY),
                                                       wavelength='w00', extension='png',
                                                       patchsize_x=patchsize, patchsize_y=patchsize,
                                                       normalizer = BFnormalizer, padValue = np.nan
                                                       )
            image_patches.append(thePatch)
            image_patches_padded.append(wasPadded)

        return object_dicts, image_patches, image_patches_padded

    def get_image_patch(self, position, timepoint, x, y, wavelength, extension, patchsize_x, patchsize_y, normalizer, padValue):
        """
        get a single image patch, basically a wrapper to tttTools.get_image_patch
        returns a tuple, the patch and a boolean whether the patch is padded
        """
        fname = self.createTTTfilename(position, timepoint, wavelength, extension)
        thePatch, wasPadded = tttTools.get_image_patch(fname, x=int(x), y=int(y),  # due to caching, the actual image is loaded only once, not for each call of get_image_patch
                                                       patchsize_x=patchsize_x, patchsize_y=patchsize_y,
                                                       normalizer=normalizer, padValue=padValue)
        return thePatch, wasPadded