from os import listdir
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io
from scipy.io import loadmat # MATLAB file loading
from functools import lru_cache


def parseTTTfilename(filename):
    "returns (directory, movieID, position, timepoint, wavelength). dir is the entire folder where the img is located (not the reference folder from TTT/movie)"
    if filename[-3:] == 'png':
        imagenamePattern = '(.+/)?(\w+)_p(\d{4})_t(\d{5})_z001_w(0?\d)\.png'
        extension = 'png'
    else:
        imagenamePattern = '(.+/)?(\w+)_p(\d{4})_t(\d{5})_z001_w(0?\d)\.jpg'
        extension = 'jpg'

    p = re.compile(imagenamePattern)
    m = p.match(filename)

    # TODO assert number of groups
    directory, movieID, position, timepoint, wavelength = [m.group(i) for i in range(1,6)]
    return directory, movieID, int(position), int(timepoint), wavelength, extension


def getTTTDir():
    import getpass
    username = getpass.getuser()
    if username in ['gpu-devuser01','gpu_devuser01']:
        TTTDIR = '/home/%s/MSt/TTT/'%username
    else:
        TTTDIR = '/storage/icbTTTdata/TTT/'
    return TTTDIR


def __normalize_image(img, background, gain=None):

    if gain is not None:
        normed = ((img - background) / gain)
    else:
        bg = np.median(img.flatten())
        normed = ((img / background) - 1) * bg

    return normed


def __load_image(filename):
    "loads an image, does bit conversion. THis one should always be used for the quantitative stuff"

    # img = scipy.misc.imread(filename) # WARNING: THIS ONE IS BAD, something strange happens when we load binary images: its non-detemrinsitc, smthimes the loaded image is crap!!
    img = skimage.io.imread(filename)

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
        raise NotImplementedError('unknwon bitdept')

    return img

@lru_cache(maxsize=100)
def loadimage(filename, normalize, BF_norm='felix'):
    img = __load_image(filename)

    directory, movieID, position, timepoint, wavelength, extension = parseTTTfilename(filename)

    if normalize:

        # for brightfield, different normalization
        if wavelength == '00':
            if BF_norm == 'felix':
                print('Felixs normalization')
                img = loadimage_felix_bg(filename)
            elif BF_norm == 'SLIC':
                print('SLIC normalization')
                img = loadimage_SLIC_bg(filename)
        else:
            BGpos_folder = "%s/../background/%s_p%.4d/" % (directory, movieID, position)
            bgname = "%s/%s_p%.4d_t%.5d_z001_w%s.png" % (BGpos_folder, movieID, position, timepoint, wavelength)
            gainname = '%s/gain_w%s.png' % (BGpos_folder, wavelength)
            offsetname = '%s/offset_w%s.png'% (BGpos_folder, wavelength)

            if os.path.exists(bgname):
                background = __load_image(bgname)
                if os.path.exists(gainname):
                    gain = __load_image(gainname)*255
                    img = __normalize_image(img, background, gain)

                else:
                    print('Warning: Gain/offset not found %s using old normalization\n' % gainname)
                    img = __normalize_image(img, background, gain=None)

            else:
                print('Warning: Background not found %s no normalization will be applied.\n' % bgname)

    img[np.isnan(img)]=0
    img[np.isinf(img)]=0
    return img

def loadimage_felix_bg(filename):
    """
    for brightfield we can use felix's special normalizatiton procedure
    :param filename:
    :return:
    """

    directory, movieID, position, timepoint, wavelength, extension = parseTTTfilename(filename)
    bgfilename = '%s/../background_projected/%s_p%04d/%s_p%04d_w%s_projbg.png' %(directory,movieID,position,movieID,position,wavelength)
#             I_org = imread(filename);
#             bg = imread(bgfilename);
#             I = im2double(I_org);
#             bg = im2double(bg);

    bg = __load_image(bgfilename)
    I = __load_image(filename)

    I = I/bg;
    I = I-np.min(I);
    img = I/np.max(I);

    # felix's proposed inter-picture normailisation (the one i dont get)
    # subtract from the corrected "img" the difference
    # mean(img)-mean(bg)

    img = img- (np.mean(img)-np.mean(bg))

    return img

def loadimage_SLIC_bg(filename):
    """
    using SLIC by Tinying to normalize the images

    current problem (on the SLIC side) is that it cannot be com[puted for the
    entire time of the position (huge MEM). hence i compute it for different intervals

    :param filename:
    :return:
    """

    _SLIC_load_precomputed_bgs(filename)
    I = skimage.io.imread(filename)  # dont use __loadimage here, the SLIC is calculated on the [0, 255] range, hence it has to be applied to this range

    imgCorrect = (I-matDict['darkfield'])/matDict['flatfield']  # (I - DF)/FF
    return imgCorrect

def _SLIC_load_precomputed_bgs(filename):
    """
    loads the precomputed darkfield, flatfield, baseline from the SLIC normalization
    assumes the bg-files are located in movieDir/background_SLIC/

    :param filename: the image we want to normalize (used to parse timepoint/position/movie/ etc)
    :return: a dict with 3 fields: darkfield, flatfield, b
    """
    # the entire data needed is stored in a matfile located under /background_SLIC/
    # however, we have to figure out which one
    directory, movieID, position, timepoint, wavelength, extension = parseTTTfilename(filename)

    bgFolder = "%s/../background_SLIC" % directory  # as directory is not the movieDir, but the one where the file is located
    pattern = re.compile(r"%s_p%.04d_(\d+)-(\d+)_w%s.%s_SLIC.mat" % (movieID, position, wavelength, extension))

    candidates = []
    for f in listdir(bgFolder):
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


def mapCoordinates_relative2absolute(relxVector, relyVector, positionVector, xmlfilename):
    """
    % mapCoordinates_relative2absolute - maps relative coordinates (intra position) to global coordinates
    % typically we look at cells within a position and use their "relative" (X,Y) coords
    % when looking across positions, we have to use coordinates that are valid globally
    % using the info about the absolute coordinates of the positions themselves (TATxml),
    % we transform the realtive to absolute coordinates
    %
    % Syntax:  [absX, absY]=mapCoordinates_relative2absolute(relXvector,relYvector,positionVector,xmlfilename)
    %
    % Inputs:
    %    relXvector     - vector of relative x coordinates (all 3 have to be the same length)
    %    relYvector     - vector of relative y coordinates
    %    positionVector - vector of positions
    %    xmlfilename    - full path to the tat.xml file
    %
    % Outputs:
    %    absX - result of transforming relX into absolute coordinates
    %    absY - result of transforming relY into absolute coordinates
    """
    assert len(relxVector)==len(relyVector) and len(relxVector) == len(positionVector), 'first second and thrid args must have same length'
    offsetStruct = create_struct_from_tat_xml(xmlfilename)
    dummy,ix = __ismember__(positionVector,offsetStruct.position)
    assert np.all(dummy)

    offsetX, offsetY = offsetStruct.offsetX(ix), offsetStruct.offsetY(ix)

    absX = relxVector + offsetX
    absY = relyVector + offsetY
    return absX, absY


from bs4 import BeautifulSoup
def create_struct_from_tat_xml(xmlfilename):
    with open(xmlfilename, 'r') as f:
        bs = BeautifulSoup(f, 'xml.parser')
    """
    %parse the XML
    fac=xml_reader(fileread(xmlfilename));


    offsetStruct.position = zeros(1,length(fac.positions));
    offsetStruct.offsetX = zeros(1,length(fac.positions));
    offsetStruct.offsetY = zeros(1,length(fac.positions));
    for i = 1:length(fac.positions)

        offsetStruct.offsetX(i) =  str2num(fac.offsetx{i}.PosInfoDimension{1}.ATTRIBUTE.posX);
        offsetStruct.offsetY(i) =  str2num(fac.offsety{i}.PosInfoDimension{1}.ATTRIBUTE.posY);
        offsetStruct.position(i) = fac.positions(i);
    end
    """
    raise NotImplementedError() # TODO: implement xml parsing

def __ismember__(A,B):
    "from https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function"
    B_unique_sorted, B_idx = np.unique(B, return_index=True)
    B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)
    return B_unique_sorted[B_in_A_bool], B_idx[B_in_A_bool]

# def __ismember2__(a, b):
#     bind = {}
#     for i, elt in enumerate(b):
#         if elt not in bind:
#             bind[elt] = i
#     return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value


def get_image_patch(imgFile, normalize, x, y, patchsize_x, patchsize_y, BF_norm='felix'):
    """
    isolates a patch at the given position in the image

    :param imgFile: an image file (str)
    :param normalize: bool wheter to apply background normalization
    :param x: center of the patch in x ()
    :param y: center of the patch in y
    :param patchsize_x: total witdth of the patch
    :param patchsize_y: total hieght of the patch
    :return: np.array: patchsize_x, patchsize_x
    """
    assert patchsize_x%2 == 1 and patchsize_y%2 == 1, "only odd patchsize supported" # TODO relax to even patchsize
    img = loadimage(imgFile, normalize, BF_norm)

    X_WINDOWSIZE_HALF = int((patchsize_x-1)/2)
    Y_WINDOWSIZE_HALF = int((patchsize_y-1)/2)
    x_box = range(max(1,x-X_WINDOWSIZE_HALF) , min(x+X_WINDOWSIZE_HALF,img.shape[0]))  # TODO THIS is still a mess with rows/col vs x,y
    y_box = range(max(1,y-Y_WINDOWSIZE_HALF) , min(y+Y_WINDOWSIZE_HALF,img.shape[1]))

    return img[np.ix_(x_box, y_box)] # confusing as rows are actually the y coordinate when indexing...


if __name__ == '__main__':
    imgname = '/storage/icbTTTdata/TTT/140206PH8/140206PH8_p0045/140206PH8_p0045_t05373_z001_w01.png'
    q = loadimage(imgname, normalize=True, BF_norm=BF_norm)
    plt.imshow(q)
    plt.show()



    imgname = '/storage/icbTTTdata/TTT/140206PH8/140206PH8_p0020/140206PH8_p0020_t02500_z001_w00.png'
    q = loadimage_SLIC_bg(imgname)
    w = loadimage_felix_bg(imgname)

    I = __load_image(imgname)