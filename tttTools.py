import re
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.misc

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
    img = scipy.misc.imread(filename)

    if img.dtype == np.uint8:
        img = img / (2 ** 8 - 1)

    elif img.dtype == np.uint16:
        img = img / (2 ** 16 - 1)
    elif img.dtype == np.int32: # also seems to correspond to 16 bit (comapring matlabs output)
        img = img/ (2 ** 16 - 1)
    else:
        raise NotImplementedError('unknwon bitdept')

    return img


def loadimage(filename, normalize):
    img = __load_image(filename)

    directory, movieID, position, timepoint, wavelength, extension = parseTTTfilename(filename)

    if normalize:

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


def get_image_patch(imgFile, normalize, x, y, patchsize_x, patchsize_y):
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
    img = loadimage(imgFile, normalize)

    X_WINDOWSIZE_HALF = (patchsize_x-1)/2
    Y_WINDOWSIZE_HALF = (patchsize_y-1)/2
    x_box = range(max(1,x-X_WINDOWSIZE_HALF) , min(x+X_WINDOWSIZE_HALF,img.shape[1]))
    y_box = range(max(1,y-Y_WINDOWSIZE_HALF) , min(y+Y_WINDOWSIZE_HALF,img.shape[0]))

    return img[y_box, x_box] # confusing as rows are actually the y coordinate when indexing...


if __name__ == '__main__':
    imgname = '/storage/icbTTTdata/TTT/140206PH8/140206PH8_p0045/140206PH8_p0045_t05373_z001_w01.png'
    q = loadimage(imgname, normalize=True)
    plt.imshow(q)
    plt.show()