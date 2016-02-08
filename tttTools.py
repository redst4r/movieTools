from os import listdir
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io
from scipy.io import loadmat # MATLAB file loading
from functools import lru_cache
from imageNormalizer import NoNormalizer

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


def get_image_patch(imgFile, x, y, patchsize_x, patchsize_y, normalizer=None):
    """
    isolates a patch at the given position in the image

    :param imgFile: an image file (str)
    :param x: center of the patch in x ()
    :param y: center of the patch in y
    :param patchsize_x: total witdth of the patch
    :param patchsize_y: total hieght of the patch
    :param normalizer: instance of ImageNormalizer. if set to "None", it defaults to 'NoNormalizer' which will do no normalization
    :return: np.array: patchsize_x, patchsize_x
    """
    if normalizer is None:
        normalizer = NoNormalizer()

    assert patchsize_x%2 == 1 and patchsize_y%2 == 1, "only odd patchsize supported" # TODO relax to even patchsize
    img = normalizer.normalize(imgFile)

    X_WINDOWSIZE_HALF = int((patchsize_x-1)/2)
    Y_WINDOWSIZE_HALF = int((patchsize_y-1)/2)
    x_box = range(max(1,x-X_WINDOWSIZE_HALF) , min(x+X_WINDOWSIZE_HALF,img.shape[0]))  # TODO THIS is still a mess with rows/col vs x,y
    y_box = range(max(1,y-Y_WINDOWSIZE_HALF) , min(y+Y_WINDOWSIZE_HALF,img.shape[1]))

    return img[np.ix_(x_box, y_box)] # confusing as rows are actually the y coordinate when indexing...
