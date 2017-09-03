from tttTools import get_image_patch
from mock import patch
import numpy as np
import imageNormalizer


@patch('imageNormalizer.NoNormalizer') # note that this mocks the instance of NoNormalizer created here!
def test_get_image_patch_zoom_returnsize(mock_norm):

    mock_norm.normalize.return_value = np.ones((1040,1388))

    patchsize = 29
    zoom = 0.25
    q, _ = get_image_patch(imgFile='bbla', x=500, y=300, patchsize_x=patchsize, patchsize_y=patchsize, zoomfactor=zoom, normalizer=mock_norm, padValue=np.nan)

    assert q.shape==(patchsize-1, patchsize-1)