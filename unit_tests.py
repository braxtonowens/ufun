# File that contains unit tests for ufun.py

import pytest
import numpy as np
from src.ufun_cbo27.ufun import kernel_pca_kmeans, cur

def test_pca_kmeans():
    '''
    Tests pca_kmeans function
    '''
    matrix = np.array([[0,0,.5],[1,0,1],[0,1,.5],[.5,1,1]])
    my_cluster = kernel_pca_kmeans(matrix,2,2)
    print(my_cluster)
    assert np.allclose(np.shape(my_cluster),(2,2))
    

def test_cur():
    '''
    Tests column_select function
    '''
    matrix = np.array([[0,0,.5,.6],[1,0,1,8],[0,1,.5,.9],[.5,1,1,2]])
    my_column = cur(matrix,3,.001)
    assert np.allclose(np.shape(my_column),(4,3))