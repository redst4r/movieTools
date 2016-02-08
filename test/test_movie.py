import sys
sys.path.append('..')
from movie import Movie
import numpy as np
import pandas as pd
import imageNormalizer
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


def test_loadimage():
    m = Movie('140206PH8')

    img_BF = m.loadimage(1,100,'w00','png', normalizer=imageNormalizer.NoNormalizer()) # TODO mock NoNormalizer
    assert img_BF.shape == (1040, 1388)

    img_FL = m.loadimage(1,1,'w01','png', normalizer=imageNormalizer.NoNormalizer())  # TODO mock NoNormalizer
    assert img_FL.shape == (1040, 1388)

def test_load_image_segmentation():
    m = Movie('140206PH8')
    img_SEG = m.load_segmentation_image(1,1,'w00','png')
    assert img_SEG.shape == (1040, 1388)
    assert np.all(np.logical_or(img_SEG == True, img_SEG == False))  # make sure its a binary mask

def test_get_segmented_objects():
    m = Movie('140206PH8')

    df = m.get_segmented_objects(1,1000,'w00','w00')
    assert isinstance(df, pd.DataFrame)
    print(df.head())

    df2 = m.get_segmented_objects(43,2612,'w01','w00')
    print(df2.head(n=100))

if __name__ == '__main__':
    test_createTTTFilename()
    test_createTATexpfilename()
    test_loadimage()
    test_load_image_segmentation()
    test_get_segmented_objects()