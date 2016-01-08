import re
import numpy as np
import os
import matplotlib.pyplot as plt
import tttTools
import skimage.measure
from collections import namedtuple
from movie import Movie
import pandas as pd

segmentedObject = namedtuple('segObject', "relX relY time position area wavelength")


def readSegmentation_single_position(movieID,position,timepoint,FluorWL,SEG_WL):
    """
    Inputs:
       movieID                    - string of the movie
       position                   - position of interest
       timepoint                  - the timepoint we're interested in
       FluorWL                    - what wavelength to quantify
       SEG_WL                     - extension of the segmenation images (e.g 'w01.png')
     Outputs:
       objects_df                 - pandas.Dataframe containing all the segmented objects with their properties
    """

    movie = Movie(movieID)

    # load segmentation
    extension = 'png'
    segImg = movie.load_segmentation_image(position, timepoint, SEG_WL, extension=extension)
    fluorImg = movie.loadimage(position, timepoint, FluorWL, extension=extension, normalize=True)

    STATS = skimage.measure.regionprops(skimage.measure.label(segImg))  # different than matlab: regionprops doesnt take the raw image but the labeled one

    segObjects = []

    minsize, eccfilter = 25, 0.9  # filter out some dirt
    for obj in filter(lambda x: x.area>minsize and x.eccentricity < eccfilter, STATS):

        x, y = obj.centroid
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