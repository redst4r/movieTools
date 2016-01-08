import tttTools
from segmentation import readSegmentation_single_position

class Movie(object):
    """an entire time lapse experiment """

    def __init__(self, movieID):
        """Constructor for Movie"""
        self.movieID = movieID
        self.TTTDIR = tttTools.getTTTDir()

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
            filename = '%s/%s/%s_p%.4d/%s_p%.4d_t%.5d_z001_%s.%s' % (self.TTTDIR, self.movieID, self.movieID, position, self.movieID, position, timepoint,WL, extension)
            filename2 = '{dir}/{movie}/{movie}_p{pos:.4d}/{movie}_p{pos:.4d}_t{time:.5d}_z001_{WL}.{ext}'.format(
                    dir=self.TTTDIR, movie=self.movieID, pos=position, time=timepoint,WL=WL, ext=extension)
            assert filename == filename2 #TODO replace with second
        else:
            filename = '%s/%s/segmentation/%s_p%.4d/%s_p%.4d_t%.5d_z001_%s.%s' % (self.TTTDIR, self.movieID, self.movieID,position,self.movieID,position,timepoint,WL, extension)
            filename2 = '{dir}/{movie}/segmentation/{movie}_p{pos:.4d}/{movie}_p{pos:.4d}_t{time:.5d}_z001_{WL}.{ext}'.format(
                    dir=self.TTTDIR, movie=self.movieID, pos=position, time=timepoint,WL=WL, ext=extension)
            assert filename == filename2 #TODO replace with second

        return filename

    def createTATexpfilename(self):
        """
        returns the filename of the TATxml file
        :return: str
        """
        return '{dir}/{movie}/{movie}_TATexp.xml'.format(dir=self.TTTDIR, movie=self.movieID)

    def loadimage(self, position, timepoint, WL, extension, normalize):
        filename = self.createTTTfilename(position, timepoint, WL, extension, SEGFLAG=False)
        return tttTools.loadimage(filename, normalize)


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
        return tttTools.loadimage(fname, normalize=False)

        # TODO exception if no ssegmentaion is available

    def get_segmented_objects(self, position, timepoint, quantifyWavelength, SEG_WL):
        """
        Wrapper around segmentation.readSegmentation_singlePosition
        :param position:
        :param timepoint:
        :param quantifyWavelength: {w01.png}
        :param SEG_WL: segmenation wavlength {w00.png}
        :return:
        """
        return readSegmentation_single_position(self.movieID, position, timepoint, quantifyWavelength, SEG_WL)

