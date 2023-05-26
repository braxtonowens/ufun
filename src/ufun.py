"""
ufun Package
------------------------------
This simple python package that contains a few functions that are useful for my research.
"""

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import det, qr
import math
from scipy.linalg import svd
from scipy.stats import rv_discrete
import time


### Functions to Fix Matrix size

def kernel_pca_kmeans(matrix,n_components,n_clusters):
	'''
	Perform Kernel PCA and KMeans clustering on a matrix
	PCA reduces the dimensionality of the matrix to n_components and KMeans clusters the reduced matrix into n_clusters
	The idea behind this is that lets say you have a grain boundary with N atoms, then you represent the data using SOAP.
	Thus your Nx3 matrix is now a NxM matrix where M is the number of SOAP features.
	You can then use this function to reduce the dimensionality of the matrix to a NxCp matrix where Cp is the number of components you want to reduce the matrix to.
	You can then cluster the reduced matrix into a ClxCp matrix where Cl is the number of clusters you want to cluster the reduced matrix into.
	Now you have a bunch of cluster centers that now represent the grain boundary.

	Parameters:
		matrix (array): matrix to be reduced
		n_components (int): number of components to reduce matrix to
		n_clusters (int): number of clusters to cluster reduced matrix into

	Returns:
		cluster_centers (array): array of cluster centers
	
	Example:
		.. code-block:: python

			matrix = np.array([[0,0],[1,0],[0,1],[1,1]])
			kernel_pca_kmeans(matrix,2,2)
	'''
	start_pca = time.time()
	transformer = KernelPCA(n_components=n_components, kernel='linear')
	reduced_matrix = transformer.fit_transform(matrix)
	end_pca = time.time() - start_pca
	print("Time to perform PCA: ", end_pca)
	start_kmeans = time.time()
	kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(reduced_matrix)
	end_kmeans = time.time() - start_kmeans
	print("Time to perform KMeans: ", end_kmeans)
	return_val = kmeans.cluster_centers_
	del(transformer)
	del(reduced_matrix)
	del(kmeans)
	return(return_val)


def pca_kmeans(matrix,n_components,n_clusters):
	'''
	Perform PCA and KMeans clustering on a matrix

	Parameters:
		matrix (array): matrix to be reduced
		n_components (int): number of components to reduce matrix to
		n_clusters (int): number of clusters to cluster reduced matrix into

	Returns:
		cluster_centers (array): array of cluster centers

	Example:
		.. code-block:: python

			matrix = np.array([[0,0],[1,0],[0,1],[1,1]])
			pca_kmeans(matrix,2,2)
	'''
	transformer = PCA(n_components=n_components)
	reduced_matrix = transformer.fit_transform(matrix)
	kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(reduced_matrix)
	return(kmeans.cluster_centers_)

def gram_calculate_volume(simplex):
    # Subtract the first vertex from all other vertices
    simplex = simplex[1:] - simplex[0]
    # Compute the Gram matrix
    gram_matrix = np.dot(simplex, simplex.T)
    # Calculate the volume
    return np.sqrt(np.abs(det(gram_matrix))) / math.factorial(simplex.shape[0]-1)

def find_largest_simplex(matrix):
	'''
	Find the largest simplex in a set of points

	Parameters:
		matrix (array): array of points to find the largest simplex in

	Returns:
		order_indicies (array): array of indicies of the points in the largest simplex
	
	Example:
		.. code-block:: python

			matrix = np.array([[0,0],[1,0],[0,1],[1,1]])
			find_largest_simplex(matrix)
	'''
	matrix = np.asarray(matrix)
	n = len(matrix)
	# Start by identifying the two environments characterized by the largest distance
	distances = squareform(pdist(matrix, 'euclidean'))
	x0, x1 = np.unravel_index(np.argmax(distances), distances.shape)
	simplex = [x0, x1]
	simplex_matrix = matrix[simplex]
	oreder_indicies = []
	while True:
		max_volume = 0
		for i in range(n):
			if i in simplex:
				continue
			temp_simplex = np.vstack([simplex_matrix, matrix[i]])
        	# Calculate the volume
			volume = gram_calculate_volume(temp_simplex)
			if volume > max_volume:
				max_volume = volume
				max_volume_index = i
		simplex_matrix = np.vstack([simplex_matrix, matrix[max_volume_index]])
		order_indicies.append(max_volume_index)
	return order_indicies

def get_simplex_projection(row, simplex):
	# Solves ax=b to find the projection of col onto the simplex
	A = simplex.T
	b = row.T
	x = np.linalg.solve(A, b)
	return x

def cur(A, k, epsilon):
	#this function follows the CUR decomposition algorithm from the paper CUR matrix decompositions for improved data analysis

	#Compute the top k right singular vectors of A
	U, S, V = svd(A, full_matrices=False)

	#Compute the normalized statistical leverage scores
	probabilities = []
	for svec in V:
		pi = np.sum(svec[:k]**2) / k
		probabilities.append(pi)
	#Keep the jth column of A with probability min(1,c*pi[j]) where c = O(k*ln(k)/epsilon^2)
	c = k * np.log(k) / epsilon**2
	probj = np.minimum(1, c*pi)

	#Return the matrix C consisting of the selected columns of A

	return probabilities