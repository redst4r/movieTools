from imageNormalizer import NoNormalizer, Felix_Normalizer, MSch_Normalizer, SLIC_Normalizer
import imageNormalizer
import numpy as np
from unittest.mock import patch
import pytest


def get_random_image(size=(1040,1388), type='uint8'):
    "fake image used for mocking"
    return np.random.randint(0,256, size=size).astype(type) # pngs contain 255 as max value


def test___load_raw_image__returnType():
    # only test where the libray call is not mocked
    q = imageNormalizer.__load_raw_image__('test/testImage/sample_digit.png')
    assert q.dtype == 'uint8'


@patch('skimage.io.imread')
def test___bit_normalize__(mock_imread):

    fakeImg = get_random_image()
    mock_imread.return_value = fakeImg

    q = imageNormalizer.__bit_normalize__('someFile.png')
    assert q.max() <= 1

@patch('skimage.io.imread')
def test___bit_normalize__exception(mock_imread):
    "has to raise exception if some crayz image type"
    fakeImg = get_random_image(type='float32')
    mock_imread.return_value = fakeImg

    with pytest.raises(NotImplementedError):
        imageNormalizer.__bit_normalize__('someFile.png')

"------------------------------------------------------------------------"
@patch('skimage.io.imread')
def test_NoNormalizer(mock_imread):

    # mock out the imread call to return a standrad image
    fakeImg = get_random_image()
    mock_imread.return_value = fakeImg

    n = NoNormalizer()
    imgname = 'someImage.png'
    q = n.normalize(imgname)

    mock_imread.assert_called_with(imgname)

    assert np.all(q==fakeImg), "NoNormalizer must leave the image unchanged"
    assert q.shape == (1040, 1388), "size must remain unchanged"
    assert q.dtype == 'uint8', 'must return uint8 array'


"------------------------------------------------------------------------"

# def test_SLIC_normalizer__load_precomputed_bgs():
#     pass

@patch('skimage.io.imread')
def test_Felix_Normalizer___load_bg_for__(mock_imread):
    "check the the projected background file is loaded from the correct location and that image size matches"
    fakeImg = get_random_image()
    mock_imread.return_value = fakeImg

    imgName = '/somepath/140206PH8/140206PH8_p0045/140206PH8_p0045_t05373_z001_w01.png'
                         #
    bg = Felix_Normalizer().__load_bg_for__(imgName)
    mock_imread.assert_called_with('/somepath/140206PH8/140206PH8_p0045//../background_projected/140206PH8_p0045/140206PH8_p0045_w01_projbg.png')

    assert bg.shape == (1040,1388)
    assert bg.dtype == 'float64'


@patch('imageNormalizer.Felix_Normalizer.__load_bg_for__')  # to mok the bg_loader
@patch('skimage.io.imread')                                 # to mock the foreground loader
def test_Felix_Normalizer_normalize(mock_imread, mock_loadBG):

    # create to fake images, first call to the mock gets the background, the second the forground
    fakeImg_bg = get_random_image(type='float32')
    mock_loadBG.return_value = fakeImg_bg

    fakeImg_foreground = get_random_image(type='uint8')  # forground has to be  a 8bit image
    mock_imread.return_value = fakeImg_foreground

    filename = 'someImg.png'
    q = Felix_Normalizer().normalize(filename)

    mock_loadBG.assert_called_with(filename)
    mock_imread.assert_called_with(filename)

    assert q.shape == (1040, 1388)
    assert q.dtype == 'float64'
