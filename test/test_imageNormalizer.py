from imageNormalizer import NoNormalizer, Felix_Normalizer, MSch_Normalizer, SLIC_Normalizer
import tttTools
import numpy as np
import skimage.io

def test_NoNormalizer():

    n = NoNormalizer()

    imgname = '/storage/icbTTTdata/TTT/140206PH8/140206PH8_p0045/140206PH8_p0045_t05373_z001_w01.png'
    q = n.normalize(imgname)
    w = skimage.io.imread(imgname) # load it myself
    assert np.all(q==w), "NoNormalizer must leave the image unchanged"


    assert q.shape == (1040, 1388), "size must remain unchanged"


# def test_SLIC_normalizer():
#     pass