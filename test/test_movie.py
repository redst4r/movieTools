import sys
sys.path.append('..')
from movie import Movie
import numpy as np
import imageNormalizer
from mock import patch, MagicMock
import pytest
import os

def test_createTTTFilename():
    """

    """
    m = Movie('140206PH8')
    f = m.createTTTfilename(1,1,'w02','png')
    assert isinstance(f, str), 'return no string'

    fseg = m.createTTTfilename(1,1,'w00','png',SEGFLAG=True)
    assert isinstance(fseg, str), 'return no string'


def test_createTATexpfilename():
    m = Movie('140206PH8')
    assert isinstance(m.createTATexpfilename(), str), 'no string returned'


@patch('movie.os.path.isfile')
@patch('imageNormalizer.NoNormalizer') # note that this mocks the instance of NoNormalizer created here!
def test_loadimage(mock_norm, mock_isfile):
    "loadimage just utilizes the Normalizer to load an image. no own action!"
    mock_norm.normalize.return_value = np.ones((1040,1388))
    mock_isfile.return_value = True

    m = Movie('140206PH8')
    img_BF = m.loadimage(1,100,'w00','png', normalizer=mock_norm)
    assert mock_norm.normalize.called, 'normalization wasnt called'
    assert img_BF.shape == (1040, 1388), 'wrong output format'

    # same just for FL wavelength
    mock_norm.normalize.reset_mock()
    img_FL = m.loadimage(1,1,'w01','png', normalizer=mock_norm)
    assert img_FL.shape == (1040, 1388)
    assert mock_norm.normalize.called, 'normalization wasnt called'

@patch('movie.os.path.isfile')
@patch('imageNormalizer.NoNormalizer')
def test_loadimage_return_none_if_no_file(mock_norm, mock_isfile):
    mock_norm.normalize.return_value = np.ones((1040,1388))
    mock_isfile.return_value = False

    m = Movie('140206PH8')
    img = m.loadimage(1,100,'w00','png', normalizer=mock_norm)
    assert not mock_norm.normalize.called, 'should not call noramlize if image doesnt exist; jsut return None'
    assert img is None


@patch('movie.NoNormalizer')  # mocking the NoNormalizer inside the package
def test_load_image_segmentation(mock_norm):
    # TODO this mocking doesnt seem to be very pythonic
    mock_norm.return_value = mock_norm  # the constructor called when doing NoNormalizer() just returns the same mock again
    mock_norm.normalize.return_value = np.ones((1040,1388), dtype='uint8')*255

    m = Movie('140206PH8')
    img_SEG = m.load_segmentation_image(1,1,'w00','png')
    assert img_SEG.shape == (1040, 1388)
    assert mock_norm.normalize.called, 'no call to NoNormalizer.normalize'


@patch('movie.NoNormalizer')  # mocking the NoNormalizer inside the package
def test_load_image_segmentation_binarize(mock_norm):
    # TODO this mocking doesnt seem to be very pythonic
    mock_norm.return_value = mock_norm  # the constructor called when doing NoNormalizer() just returns the same mock again
    fakeSegImg =  np.ones((1040,1388), dtype='uint8')*255  # usually we dont get 0/1 images
    mock_norm.normalize.return_value =  fakeSegImg

    m = Movie('140206PH8')
    img_SEG = m.load_segmentation_image(1,1,'w00','png')
    assert img_SEG.shape == (1040, 1388)
    assert mock_norm.normalize.called, 'no call to NoNormalizer.normalize'
    assert np.all(np.logical_or(img_SEG == True, img_SEG == False))

@patch('movie.NoNormalizer')  # mocking the NoNormalizer inside the package
def test_load_image_segmentation_exception_if_not_binary(mock_norm):
    """ must throw an exception if the loaded mask is not binary"""

    # TODO this mocking doesnt seem to be very pythonic
    mock_norm.return_value = mock_norm  # the constructor called when doing NoNormalizer() just returns the same mock again

    fakeSegImg = np.ones((1040,1388), dtype='uint8')*255
    fakeSegImg[0,0] = 128   # a non binary mask
    mock_norm.normalize.return_value =  fakeSegImg

    m = Movie('140206PH8')
    with pytest.raises(ValueError):
        m.load_segmentation_image(1,1,'w00','png')

    assert mock_norm.normalize.called, 'no call to NoNormalizer.normalize'

"removed as get_segmented_objects is deprecated"
# def test_get_segmented_objects_return_empty_if_no_segmentation():
#     m = Movie('140206PH8')
#
#     with patch('movie.Movie.load_segmentation_image', new=MagicMock(side_effect=FileNotFoundError(), autospec=True)):
#         df = m.get_segmented_objects(1,1000,'w00','w00')
#         assert len(df) == 0
#
#
# @patch('movie.NoNormalizer', autospec=True)
# @patch('movie.MSch_Normalizer', autospec=True)
# def test_get_segmented_objects_return_pdDataFrame(mock_MSchNorm, mock_NoNorm):
#     m = Movie('140206PH8')
#
#     mock_MSchNorm.return_value = mock_MSchNorm # construcvtor
#     mock_MSchNorm.normalize.return_value = np.ones((1040,1388)) # TODO this mocking doesnt seem to be very pythonic
#
#     mock_NoNorm.return_value = mock_NoNorm # construcvtor
#     mock_NoNorm.normalize.return_value = np.ones((1040,1388)) # TODO this mocking doesnt seem to be very pythonic
#
#
#     df = m.get_segmented_objects(1,1000,'w00','w00')
#
#     for m in [mock_NoNorm, mock_MSchNorm]:
#         assert m.normalize.called
#
#     assert isinstance(df, pd.DataFrame)
#     print(df.head())

@patch('movie.os.listdir')
def test_get_all_positions(mock_listdir):
    "check the parsing of the foldernames"

    m = Movie('140206PH8')
    movieFolder = os.path.join(m.TTTDIR, m.movieID)

    trueFolders = ['140206PH8_p0001', '140206PH8_p0002', '140206PH8_p0003']
    fakeFoldes =  ['120502PH7_p0001', '140206PH8_p002', '140206PH8_p0003.old']
    mock_listdir.return_value = trueFolders + fakeFoldes

    pos = m.get_all_positions()

    assert mock_listdir.called
    assert isinstance(pos, dict), 'has to return a dict (int)->folderName'

    #slightly complicated, as the returned values are as absolute Path
    trueReturn = {i+1: os.path.join(movieFolder,j) for i,j in enumerate(trueFolders)}
    assert pos == trueReturn , 'did not return the true positions'


@patch('movie.Movie.get_all_positions')
@patch('movie.os.listdir')
def test_all_images(mock_listdir, mock_getpos):

    m = Movie('140206PH8')

    # some fake returns of position directories
    fakeFolder = '/someFolder/firstPositionDir/'
    mock_getpos.return_value = {1:fakeFolder}

    # some fake directory lisstings
    fakedir_content = ['140206PH8_p0001_t00001_z001_w01.png', '140206PH8_p0001_t00001_z001_w01.png_meta.xml']
    mock_listdir.return_value = fakedir_content

    result = m.get_all_images()
    assert mock_getpos.called
    mock_listdir.assert_called_with('/someFolder/firstPositionDir/')

    assert list(result.values()) == [fakeFolder + fakedir_content[0]], 'probably returned the xml file too!'

def test_createPositionFoldername():
    assert isinstance(Movie('140206PH8').createPositionFoldername(1), str)

def test_get_all_images():
    pass



if __name__ == '__main__':
    test_createTTTFilename()
    test_createTATexpfilename()
    test_loadimage()
    test_load_image_segmentation()
    test_get_segmented_objects()
