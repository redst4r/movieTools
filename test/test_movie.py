import sys
sys.path.append('..')
from movie import Movie
import numpy as np
import pandas as pd
import imageNormalizer
from unittest.mock import patch, MagicMock
import pytest


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


@patch('imageNormalizer.NoNormalizer', autospec=True) # note that this mocks the instance of NoNormalizer created here!
def test_loadimage(mock_norm):
    "loadimage just utilizes the Normalizer to load an image. no own action!"
    mock_norm.normalize.return_value = np.ones((1040,1388))

    m = Movie('140206PH8')
    img_BF = m.loadimage(1,100,'w00','png', normalizer=mock_norm)
    assert mock_norm.normalize.called, 'normalization wasnt called'
    assert img_BF.shape == (1040, 1388), 'wrong output format'

    # same just for FL wavelength
    mock_norm.normalize.reset_mock()
    img_FL = m.loadimage(1,1,'w01','png', normalizer=mock_norm)
    assert img_FL.shape == (1040, 1388)
    assert mock_norm.normalize.called, 'normalization wasnt called'


@patch('movie.NoNormalizer', autospec=True)  # mocking the NoNormalizer inside the package
def test_load_image_segmentation(mock_norm):
    # TODO this mocking doesnt seem to be very pythonic
    mock_norm.return_value = mock_norm  # the constructor called when doing NoNormalizer() just returns the same mock again
    mock_norm.normalize.return_value = np.ones((1040,1388))

    m = Movie('140206PH8')
    img_SEG = m.load_segmentation_image(1,1,'w00','png')
    assert img_SEG.shape == (1040, 1388)
    assert mock_norm.normalize.called, 'no call to NoNormalizer.normalize'


@patch('movie.NoNormalizer', autospec=True)  # mocking the NoNormalizer inside the package
def test_load_image_segmentation_exception_if_not_binary(mock_norm):
    """ must throw an exception if the loaded mask is not binary"""

    # TODO this mocking doesnt seem to be very pythonic
    mock_norm.return_value = mock_norm  # the constructor called when doing NoNormalizer() just returns the same mock again
    mock_norm.normalize.return_value = np.ones((1040,1388))*0.1  # a non binary mask

    m = Movie('140206PH8')
    with pytest.raises(ValueError):
        m.load_segmentation_image(1,1,'w00','png')

    assert mock_norm.normalize.called, 'no call to NoNormalizer.normalize'


def test_get_segmented_objects_return_empty_if_no_segmentation():
    m = Movie('140206PH8')

    with patch('movie.Movie.load_segmentation_image', new=MagicMock(side_effect=FileNotFoundError(), autospec=True)):
        df = m.get_segmented_objects(1,1000,'w00','w00')
        assert len(df) == 0


@patch('movie.NoNormalizer', autospec=True)
@patch('movie.MSch_Normalizer', autospec=True)
def test_get_segmented_objects_return_pdDataFrame(mock_MSchNorm, mock_NoNorm):
    m = Movie('140206PH8')

    mock_MSchNorm.return_value = mock_MSchNorm # construcvtor
    mock_MSchNorm.normalize.return_value = np.ones((1040,1388)) # TODO this mocking doesnt seem to be very pythonic

    mock_NoNorm.return_value = mock_NoNorm # construcvtor
    mock_NoNorm.normalize.return_value = np.ones((1040,1388)) # TODO this mocking doesnt seem to be very pythonic


    df = m.get_segmented_objects(1,1000,'w00','w00')

    for m in [mock_NoNorm, mock_MSchNorm]:
        assert m.normalize.called

    assert isinstance(df, pd.DataFrame)
    print(df.head())



if __name__ == '__main__':
    test_createTTTFilename()
    test_createTATexpfilename()
    test_loadimage()
    test_load_image_segmentation()
    test_get_segmented_objects()