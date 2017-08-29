"""
stuff to handle tracked data. not for actual tracking!
"""
from scipy.io import loadmat  # MATLAB file loading
import numpy as np
import pandas as pd
import progressbar
import pickle

def load_felix_inspected_tracks(filename, the_key='newTrees140206'):
    """takes a file that was exported (in matlab) from the corrected (@FB) .mat files (which themselves are exports from the InspecTGUI)
    and turns it into a list of trees. a single tree is just a dict (corresponding to the matlab structs that are used all the time for the trees)
    """
    treeArray = loadmat(filename, squeeze_me=True)[the_key]

    # wll its f*cked up to get this into python
    # 1) treeArray is a array over the trees.
    # 2) each tree is a kind of dict, but each dict entry is an array(array([some numbers]))

    theTrees = []
    for tree in treeArray:
        treeDict = {field: tree[field].item() for field in tree.dtype.names}  # item() to get it out of this array(array())
        theTrees.append(treeDict)

    return theTrees


def trees_to_dataframe(trees):
    """
    takes alist of trees (all dcitionaries) and creates a big dataframe, adding a column treeID
    also removes empty trees
    """
    for t in trees:
        assert isinstance(t, dict)

    # add the treeID field so that we know which entries belong together
    for i, t in enumerate(trees):

        tshape = t['cellNr'].shape if isinstance(t['cellNr'], np.ndarray) else  1
        t['treeID'] = i*np.ones(tshape,dtype='int16')

        # kick out attributes that we cannot store in the Dataframe
        for k in list(t.keys()):  # list() necessary to make a persistent copy of the keys, as the keys of the dict change
            if not isinstance(t[k], np.ndarray) or t[k].shape != tshape:  # not numeric attribute or some thats not as long as the entire tree
                del t[k]

    trees = [t for t in trees if t != {} and t['cellNr'].size!=0] # kcik out the empty ones
    df_trees = pd.concat([pd.DataFrame(t) for t in trees], ignore_index=True)

    return df_trees


def link_pointsets(pointset1, pointset2, distanceCutoff):
    """
    links two sets of points using LAP such that the assignment minimizes the euclidean norm
    :param pointset1: np.array Nx2
    :param pointset2: np.array Mx2
    :param distanceCutoff: float, specifying the maximal distance allowed for a link
    :return: (ind1,ind2)  telling that ind1[i] linnks to ind2[i]
    """
    assert pointset1.shape[1] == 2 and pointset2.shape[1] == 2, 'sets have to be 2dimensional (X,Y)'

    # link them via Hungarian linker
    # import sklearn.utils.linear_assignment_
    import scipy.optimize, scipy.spatial.distance
    C = scipy.spatial.distance.cdist(pointset1, pointset2, 'euclidean')
    ind1, ind2 = scipy.optimize.linear_sum_assignment(C)  # the object at ind1[i] corrsponds to the object at ind2[i]

    # only keep assignments below the threshold.
    # TODO this afterwards filtering is suboptimal. better set distances > threshold to Inf before calling LAP
    # TODO: setting them to inf before LAP crashes the linear_sum_assignment (it doenst converge..)
    distances = C[ind1, ind2]
    ix_goodAssign = distances < distanceCutoff
    ind1, ind2 = ind1[ix_goodAssign], ind2[ix_goodAssign]

    return ind1, ind2


def map_tracks_and_segmented_objects(df_seg, df_trees, distanceCutoff=np.inf):
    """
    relates segmented objects and manually tracked objects
    :param df_seg:    pd.Dataframe containing the segemented obejects across the movie (see the deeplearning project where it is created)
    :param df_trees:  pd.Dataframe created via trees_to_dataframe() containing the tracked trees
    :return:
    """
    commonPositions = np.intersect1d( df_trees.position.unique(),
                                      df_seg.position.unique() )

    index_pair_list = []  # list of np.arrays : first column is the index in seg, second is the index in trees
    for p in commonPositions:
        print(p)
        commonTime = np.intersect1d( df_trees.timepoint[df_trees.position==p].unique(),  # timepoints in that position that are shared
                                     df_seg.timepoint[df_seg.position==p].unique())

        progbar = progressbar.ProgressBar()
        for t in progbar(commonTime):
            # get the x,y coordinates of the present objects
            XY_seg = df_seg.query('position==@p & timepoint==@t')[['X','Y']]
            XY_trees = df_trees.query('position==@p & timepoint==@t')[['X','Y']]

            seg_ind, tree_ind= link_pointsets(XY_seg, XY_trees, distanceCutoff)
            # plotTracking(XY_seg, XY_trees, zip(seg_ind, tree_ind))

            # get the correspionding indexes, those are global for the entire dataframe irrespective of the time/posuition subindexing
            iSeg = XY_seg.iloc[seg_ind].index.values
            iTrees = XY_trees.iloc[tree_ind].index.values

            index_pair_list.extend(zip(iSeg, iTrees))

    ix_seg, ix_tree = zip(*index_pair_list)
    return list(ix_seg), list(ix_tree)  # they are tuples before

    # df_seg.loc[list(ix_seg),'link'] = list(ix_tree)
    # df_trees.loc[list(ix_tree),'link'] = list(ix_seg)
    #
    # join the trees onto the segmentation struct
    # q = pd.merge(df_seg, df_trees, right_index=True, left_on='link')


def plotTracking(centroid1, centroid2, trackingPairs, col1='r', col2='b', linecolor='k'):
    """
    graphical display of the linking
    if the linking says that an object appeared or
    disappeared (an element of trackingPairs len(centrioid1,2))
    """
    import matplotlib.pyplot as plt
    # plt.figure()
    for centr, col in zip([centroid1, centroid2], [col1, col2]):
        plt.scatter(centr[:,0], centr[:,1], s=40,c=col)

    for (p1, p2) in trackingPairs:
        # linked a empty to an empty
        if p1 >= len(centroid1) and p2 >= len(centroid2):
            pass
        # a new object appeared in the second frame
        elif p1 >= len(centroid1):
            plt.scatter(centroid2[p2,0], centroid2[p2,1],
                        marker='d', s=40, c=col1)
        elif p2 >= len(centroid2):  # a object disappeared
            plt.scatter(centroid1[p1,0], centroid1[p1,1],
                        marker='x', s=40, c=col2)
        else:  # genuine linking
            x1, y1 = centroid1[p1,:]
            x2, y2 = centroid2[p2,:]
            plt.plot([x1, x2],
                     [y1, y2],   color=linecolor, linestyle='-')


if __name__ == '__main__':

    filename = '140206results_felix_curated.mat'
    trees = load_felix_inspected_tracks(filename)
    df_trees = trees_to_dataframe(trees)
    df_trees.info()


    # with open('../../full_dataset_BIGDATA.pkl', 'rb') as f:
    #     BIGDATA = pickle.load(f)

    with open('../../../tmp_BIGDATA.pkl', 'rb') as f:
        BIGDATA = pickle.load(f)

    df_seg=BIGDATA
    df_seg.rename(columns={'x': 'Y', 'y': 'X'}, inplace=True)

    ix_seg, ix_tree = map_tracks_and_segmented_objects(BIGDATA, df_trees)

    with open('../../../mapping.pkl','wb') as f:
        pickle.dump((ix_seg, ix_tree),f)


    with open('../AE_timeLapse/mapping.pkl','rb') as f:
         ix_seg, ix_tree = pickle.load(f)
