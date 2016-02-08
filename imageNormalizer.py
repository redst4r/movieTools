import skimage.io
from scipy.io import loadmat # MATLAB file loading
from functools import lru_cache
import tttTools
import os
import re
import numpy as np


def __load_raw_image__(filename):
    """
    loads an image file from the disk and returns it as np.array. THIS is the prefered method of loading stuff
    as there are some issues with e.g. scipy.misc.imread!!
    :param filename: the image file to be loaded
    :return: np.array of the image
    """

    # img = scipy.misc.imread(filename) # WARNING: THIS ONE IS BAD, something strange happens when we load binary images: its non-detemrinsitc, smthimes the loaded image is crap!!
    return skimage.io.imread(filename)

def __bit_normalize__(filename):
    "loads an image, does bit conversion. supposed to put the values into [0,1]"

    img = __load_raw_image__(filename)

    if img.dtype == np.uint8:
        img = img / (2 ** 8 - 1)

    elif img.dtype == np.uint16:
        img = img / (2 ** 16 - 1)
    elif img.dtype == np.int32: # also seems to correspond to 16 bit (comapring matlabs output): seems to be -2**31 to 2**31, maybe the background/gain can be negative!
        assert not np.any(img<0)
        img = img/ (2 ** 16 - 1)
    elif img.dtype == np.bool: # proably a segmentation image
        pass # just leave it boolean
    else:
        raise NotImplementedError('unknown bitdept')

    return img


class ImageNormalizer(object):
    """
    base class to apply image normalization. inherit from this one if you want to implement one yourself

    main method is "normalize" which just takes a filename and returns an np.array

    note that these classes dont calculate/estimate the image normalization
    but just apply precalculated results
    """
    
    def __init__(self, ):
        """Constructor for ImageNormalizer"""

    def normalize(self, filename):
        """
        apply normalization to the image given as filename and return a numpy.array of the normalized image
        :param filename: the path to the image file
        :return:
        """
        raise NotImplementedError("ImageNormalizer subclasses have to implement this method")

class NoNormalizer(ImageNormalizer):
    """
    does not do any normalization at all, just load the image
    handy if we dont want any normalization, but the normal workflow
    """
    def __init__(self):
        super().__init__()

    @lru_cache(maxsize=100)
    def normalize(self, filename):
        return __load_raw_image__(filename)


class SLIC_Normalizer(ImageNormalizer):
    """
    method by Tinying Peng
    """
    
    def __init__(self, ):
        """Constructor for SLIC_Normalizer"""
        super().__init__()

    def _SLIC_load_precomputed_bgs(self, filename):
        """
        loads the precomputed darkfield, flatfield, baseline from the SLIC normalization
        assumes the bg-files are located in movieDir/background_SLIC/

        :param filename: the image we want to normalize (used to parse timepoint/position/movie/ etc)
        :return: a dict with 3 fields: darkfield, flatfield, b
        """

        # the entire data needed is stored in a matfile located under /background_SLIC/
        # however, we have to figure out which one
        directory, movieID, position, timepoint, wavelength, extension = tttTools.parseTTTfilename(filename)

        bgFolder = "%s/../background_SLIC" % directory  # as directory is not the movieDir, but the one where the file is located
        pattern = re.compile(r"%s_p%.04d_(\d+)-(\d+)_w%s.%s_SLIC.mat" % (movieID, position, wavelength, extension))

        candidates = []
        for f in os.listdir(bgFolder):
            m = pattern.match(f)
            if m:  # it the pattern matches to position/wavlength, check if the matfile is in the right timeinterval
                minT, maxT = int(m.group(1)), int(m.group(2))
                if minT <= timepoint <= maxT:
                    candidates.append(f)

        if len(candidates) == 1:
            matfile = os.path.join(bgFolder, candidates[0])
        elif len(candidates) == 0:
            raise ValueError('no background correction found for movie %s, pos %d, time %d,  WL%s' %(movieID, position, timepoint, wavelength ))

        else:  #multiple ones corresponding to the timepoint
            raise ValueError('MULTIPLE background correction found for movie %s, pos %d, time %d,  WL%s' %(movieID, position, timepoint, wavelength ))

        return loadmat(matfile) # its a dict. fields are 'flatfield','darkfield','fi_base', 'timepoint'

    @lru_cache(maxsize=100)
    def normalize(self,filename):
        """
        #TODO: current problem (on the SLIC side) is that it cannot be com[puted for the
        entire time of the position (huge MEM). hence i compute it for different intervals

        :param filename: the path to the image file
        :return:
        """

        matDict = self._SLIC_load_precomputed_bgs(filename)
        I = __load_raw_image__(filename)  # dont use __bit_normalize here, the SLIC is calculated on the [0, 255] range, hence it has to be applied to this range

        imgCorrect = (I-matDict['darkfield'])/matDict['flatfield']  # (I - DF)/FF
        return imgCorrect


class Felix_Normalizer(ImageNormalizer):
    """
    using Felix's method which just subtracts the time-average image.
    usually applied to brightfield images
    """
    def __init__(self, ):
        """Constructor for SLIC_Normalizer"""
        super().__init__()

    @lru_cache(maxsize=100)
    def normalize(self, filename):
        """
        :param filename: the path to the image file
        :return:
        """

        directory, movieID, position, timepoint, wavelength, extension = tttTools.parseTTTfilename(filename)
        bgfilename = '%s/../background_projected/%s_p%04d/%s_p%04d_w%s_projbg.png' %(directory,movieID,position,movieID,position,wavelength)

        bg = __bit_normalize__(bgfilename)
        I = __bit_normalize__(filename)

        I = I/bg
        I = I-np.min(I)
        img = I/np.max(I)

        # felix's proposed inter-picture normailisation (the one i dont get)
        # subtract from the corrected "img" the difference
        # mean(img)-mean(bg)
        img = img- (np.mean(img)-np.mean(bg))

        return img


class MSch_Normalizer(ImageNormalizer):
    """
    along Michi Schwarzsfischers paper
    """

    def __init__(self, ):
        """Constructor for MSch_Normalizer"""
        super().__init__()

    @lru_cache(maxsize=100)
    def normalize(self, filename):

        img = __bit_normalize__(filename)
        directory, movieID, position, timepoint, wavelength, extension = tttTools.parseTTTfilename(filename)
        BGpos_folder = "%s/../background/%s_p%.4d/" % (directory, movieID, position)
        bgname = "%s/%s_p%.4d_t%.5d_z001_w%s.png" % (BGpos_folder, movieID, position, timepoint, wavelength)
        gainname = '%s/gain_w%s.png' % (BGpos_folder, wavelength)
        offsetname = '%s/offset_w%s.png'% (BGpos_folder, wavelength)

        if os.path.exists(bgname):
            background = __bit_normalize__(bgname)
            if os.path.exists(gainname):
                gain = __bit_normalize__(gainname)*255
            else:
                raise Exception('Warning: Gain/offset not found %s using old normalization\n' % gainname)
                #gain = None

            img = self.__normalize_image(img, background, gain=gain)

        else:
            raise Exception('Warning: Background not found %s no normalization will be applied.\n' % bgname)

        img[np.isnan(img)]=0
        img[np.isinf(img)]=0
        return img

    def __normalize_image(self, img, background, gain=None):

        if gain is not None:
            normed = ((img - background) / gain)
        else:
            raise Exception("deprecated stuff: normalization without gain")
            bg = np.median(img.flatten())
            normed = ((img / background) - 1) * bg

        return normed