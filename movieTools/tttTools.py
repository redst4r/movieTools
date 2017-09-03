import re
import numpy as np
import movieTools.config as config
import movieTools.imageNormalizer


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
    return config.TTTDIR



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


def create_struct_from_tat_xml(xmlfilename):
    from bs4 import BeautifulSoup
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


def get_image_patch(imgFile, x, y, patchsize_x, patchsize_y, zoomfactor=1, normalizer=None, padValue=None):
    """
    isolates a patch at the given position in the image

    :param imgFile: an image file (str)
    :param x: center of the patch in x (the is COLUMS!!)
    :param y: center of the patch in y (that is ROWS!!)
    :param patchsize_x: total witdth of the patch
    :param patchsize_y: total hieght of the patch
    :param zoomfactor: how much to zoom out of the image (giving the same requested patchsize), i.e 0.5 normal patchsize but covering twice the view
    :param normalizer: instance of ImageNormalizer. if set to "None", it defaults to 'NoNormalizer' which will do no normalization
    :param padValue: if not None, fill up the imagePatch with the value in case we're at the border
    :return: np.array: patchsize_x, patchsize_x
    """

    patchsize_x -= 1
    patchsize_y -= 1

    if normalizer is None:
        normalizer = movieTools.imageNormalizer.NoNormalizer()

    assert patchsize_x%2 == 0 and patchsize_y%2 == 0, "only odd patchsize supported" # TODO relax to even patchsize
    img = normalizer.normalize(imgFile)
    img_final, wasPadded = _extract_patch(img, x, y, patchsize_x, patchsize_y, zoomfactor, padValue)

    return img_final, wasPadded

def _extract_patch(img, x, y, patchsize_x, patchsize_y, zoomfactor, padValue):

    assert x<= img.shape[1], 'out of the image %d/%d' %(x,img.shape[1])
    assert y<= img.shape[0], 'out of the image %d/%d' %(y,img.shape[0])


    assert zoomfactor == 1 or (1/zoomfactor) % 2 == 0,  'zoomfactor has to be  1/(2^n)'

    X_WINDOWSIZE_HALF = int((1/zoomfactor) * ((patchsize_x)/2))  # these are always even before multiplication
    Y_WINDOWSIZE_HALF = int((1/zoomfactor) * ((patchsize_y)/2))

    # x_box = range(max(1,x-X_WINDOWSIZE_HALF) , min(x+X_WINDOWSIZE_HALF,img.shape[0]))  # TODO THIS is still a mess with rows/col vs x,y
    # y_box = range(max(1,y-Y_WINDOWSIZE_HALF) , min(y+Y_WINDOWSIZE_HALF,img.shape[1]))
    x_box, y_box, xpad, ypad = __helper_boxsize(x,y ,X_WINDOWSIZE_HALF, Y_WINDOWSIZE_HALF, img.shape)

    x_box = range(*x_box)
    y_box = range(*y_box)

    img_unpadded = img[np.ix_(y_box, x_box)]  # confusing as rows are actually the y coordinate when indexing...

    if padValue is not None:
        img_final = np.pad(img_unpadded, pad_width=(ypad, xpad), mode='constant', constant_values=padValue)

        wasPadded = any([xpad[0] !=0, xpad[1] !=0, ypad[0] !=0, ypad[1] !=0] )
        assert img_final.shape == (patchsize_x * (1/zoomfactor), (1/zoomfactor)*patchsize_y) # check that the image gets larger as expected
    else:
        img_final = img_unpadded
        wasPadded = False

    # get the respective scaling
    f = int(1/zoomfactor)
    img_final = img_final[::f, ::f]

    if padValue is not None:
        assert img_final.shape == (patchsize_x, patchsize_y), 'something messed up the requested image shape' #TODO -1 ugly
    else:
        raise  NotImplementedError('no proper teting implemented for non padding')

    return img_final, wasPadded

def __helper_boxsize(x,y ,X_WINDOWSIZE_HALF, Y_WINDOWSIZE_HALF, imgShape ):

    if x-X_WINDOWSIZE_HALF < 1:
        xpad_left = 1+np.abs(x-X_WINDOWSIZE_HALF)
        xstart = 1
    else:
        xstart = x-X_WINDOWSIZE_HALF
        xpad_left = 0

    if x+X_WINDOWSIZE_HALF > imgShape[1]:
        xpad_right = x+X_WINDOWSIZE_HALF - imgShape[1]
        xend = imgShape[1]
    else:
        xpad_right = 0
        xend = x+X_WINDOWSIZE_HALF

    ## y

    if y-Y_WINDOWSIZE_HALF < 1:
        ypad_top = 1+np.abs(y-Y_WINDOWSIZE_HALF)
        ystart = 1
    else:
        ystart = y-Y_WINDOWSIZE_HALF
        ypad_top = 0

    if y+Y_WINDOWSIZE_HALF > imgShape[0]:
        ypad_bottom = y+Y_WINDOWSIZE_HALF - imgShape[0]
        yend = imgShape[0]
    else:
        ypad_bottom = 0
        yend = y+Y_WINDOWSIZE_HALF

    assert all([xstart>0, xend>0, ystart>0,yend>0 ])
    assert all([xpad_left>=0, xpad_right>=0, ypad_top>=0,ypad_bottom>=0 ])

    return (xstart, xend),(ystart, yend),  (xpad_left, xpad_right), (ypad_top, ypad_bottom)