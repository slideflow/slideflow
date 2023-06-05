from __future__ import division

from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist


def calculate_centroid(
    act: Dict[str, np.ndarray]
) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """Calcultes slide-level centroid indices for a provided activations dict.

    Args:
        activations (dict): Dict mapping slide names to ndarray of activations
            across tiles, of shape (n_tiles, n_features)

    Returns:
        A tuple containing

            dict: Dict mapping slides to index of tile nearest to centroid

            dict: Dict mapping slides to activations of tile nearest to centroid
    """

    optimal_indices = {}
    centroid_activations = {}
    for slide in act:
        if not len(act[slide]):
            continue
        km = KMeans(n_clusters=1, n_init=10).fit(act[slide])
        closest, _ = pairwise_distances_argmin_min(
            km.cluster_centers_,
            act[slide]
        )
        closest_index = closest[0]
        closest_activations = act[slide][closest_index]
        optimal_indices.update({slide: closest_index})
        centroid_activations.update({slide: closest_activations})
    return optimal_indices, centroid_activations


def get_centroid_index(arr: np.ndarray) -> int:
    """Calculate index nearest to centroid from a given 2D input array."""
    km = KMeans(n_clusters=1, n_init=10).fit(arr)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, arr)
    return closest[0]


def normalize_layout(
    layout: np.ndarray,
    min_percentile: int = 1,
    max_percentile: int = 99,
    relative_margin: float = 0.1
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Removes outliers and scales layout to between [0,1].

    Args:
        layout (np.ndarray): 2D array containing data to be scaled.
        min_percentile (int, optional): Percentile for scaling. Defaults to 1.
        max_percentile (int, optional): Percentile for scaling. Defaults to 99.
        relative_margin (float, optional): Add an additional margin (fraction
            of total plot width). Defaults to 0.1.

    Returns:
        np.ndarray: layout array, re-scaled and clipped.

        tuple(float, float): Range in original space covered by this layout.

        tuple(float, float): Clipping values (min, max) used for this layout
    """

    # Compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))
    # Add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)
    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)
    # embed within [0,1] along both axes
    _min = clipped.min(axis=0)
    _max = clipped.max(axis=0)
    clipped -= _min
    clipped /= (_max - _min)
    return clipped, (_min, _max), (mins, maxs)

def normalize(
    array: np.ndarray,
    norm_range: Tuple[np.ndarray, np.ndarray],
    norm_clip: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Normalize and clip an array."""
    _min, _max = norm_range
    mins, maxs = norm_clip
    clipped = np.clip(array, mins, maxs)
    clipped -= _min
    clipped /= (_max - _min)
    return clipped

def denormalize(
    array: np.ndarray,
    norm_range: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """De-normalize an array."""
    _min, _max = norm_range
    transformed = array * (_max - _min)
    transformed += _min
    return transformed

def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
    
    This method was written by Francis Song and can be found here:            
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/                                                                             
    """
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals[evals > 0]

class SAMMON:
    """Class for using Sammon dimensionality reduction.
    
    """
    def __init__(self, sammom_kwargs = None, reducer = None) -> None:
        self.sammon_kwargs = None
        self.reducer = None

    def fit_transform(self, 
               array: np.ndarray,
               n: int = 2,
               display: int = 2,
               inputdist: str = 'raw',
               maxhalves: int = 20,
               maxiter: int = 500,
               tolfun: int = 1e-9,
               init: str = 'default') -> Tuple[np.ndarray, int]: 
        """Perform Sammon mapping on dataset x
            y = fit_transform(x) applies the Sammon nonlinear mapping procedure on
            multivariate data x, where each row represents a pattern and each column
            represents a feature.  On completion, y contains the corresponding
            co-ordinates of each point on the map.  By default, a two-dimensional
            map is created.  Note if x contains any duplicated rows, SAMMON will
            fail (ungracefully). 
            
            [y,E] = fit_transform(x) also returns the value of the cost function in E (i.e.
            the stress of the mapping).
            
            An N-dimensional output map is generated by y = fit_transform(x,n).
        
        Args:
            array (np.ndarray): Array to transform with Sammon
            n (optional, int): Number of dimensions to fit the Sammon projection
                Default is 2. 
            display (int, optional): 0 to 2. 0 least verbose, 2 max verbose.
            inputdist (str, optional):{'raw','distance'} if set to 'distance', X is 
                interpreted as a matrix of pairwise distances. 
            maxhalves (int, optional): maximum number of step halvings
            maxiter (int, optional): maximum number of iterations
            tolfun (int, optional): relative tolerance on objective function
            init (str, optional): {'pca', 'cmdscale', random', 'default'}
                    default is 'pca' if input is 'raw', 
                    'msdcale' if input is 'distance'

        Returns:
            [y, E] (Tuple[np.ndarray, int]): y is a n-dim array and E is the error
        File        : sammon.py
        Date        : 18 April 2014
        Authors     : Tom J. Pollard (tom.pollard.11@ucl.ac.uk)
                    : Ported from MATLAB implementation by 
                    Gavin C. Cawley and Nicola L. C. Talbot

        Description : Simple python implementation of Sammon's non-linear
                    mapping algorithm [1].

        References  : [1] Sammon, John W. Jr., "A Nonlinear Mapping for Data
                    Structure Analysis", IEEE Transactions on Computers,
                    vol. C-18, no. 5, pp 401-409, May 1969.

        Copyright   : (c) Dr Gavin C. Cawley, November 2007.

        This program is free software; you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation; either version 2 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to the Free Software
        Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

        """

        # Create distance matrix unless given by parameters
        if inputdist == 'distance':
            D = array
            if init == 'default':
                init = 'cmdscale'
        else:
            D = cdist(array, array)
            if init == 'default':
                init = 'pca'

        if inputdist == 'distance' and init == 'pca':
            raise ValueError("Cannot use init == 'pca' when inputdist == 'distance'")

        if np.count_nonzero(np.diagonal(D)) > 0:
            raise ValueError("The diagonal of the dissimilarity matrix must be zero")

        # Remaining initialisation
        N = array.shape[0]
        scale = 0.5 / D.sum()
        D = D + np.eye(N)     

        if np.count_nonzero(D<=0) > 0:
            raise ValueError("Off-diagonal dissimilarities must be strictly positive")   

        Dinv = 1 / D
        if init == 'pca':
            [UU,DD,_] = np.linalg.svd(array)
            y = UU[:,:n]*DD[:n] 
        elif init == 'cmdscale':
            y,e = cmdscale(D)
            y = y[:,:n]
        else:
            y = np.random.normal(0.0,1.0,[N,n])
        one = np.ones([N,n])
        d = cdist(y,y) + np.eye(N)
        dinv = 1. / d
        delta = D-d 
        E = ((delta**2)*Dinv).sum() 

        # Get on with it
        for i in range(maxiter):

            # Compute gradient, Hessian and search direction (note it is actually
            # 1/4 of the gradient and Hessian, but the step size is just the ratio
            # of the gradient and the diagonal of the Hessian so it doesn't
            # matter).
            delta = dinv - Dinv
            deltaone = np.dot(delta,one)
            g = np.dot(delta,y) - (y * deltaone)
            dinv3 = dinv ** 3
            y2 = y ** 2
            H = np.dot(dinv3,y2) - deltaone - np.dot(2,y) * np.dot(dinv3,y) + y2 * np.dot(dinv3,one)
            s = -g.flatten(order='F') / np.abs(H.flatten(order='F'))
            y_old    = y

            # Use step-halving procedure to ensure progress is made
            for j in range(maxhalves):
                s_reshape = np.reshape(s, (-1,n),order='F')
                y = y_old + s_reshape
                d = cdist(y, y) + np.eye(N)
                dinv = 1 / d
                delta = D - d
                E_new = ((delta**2)*Dinv).sum()
                if E_new < E:
                    break
                else:
                    s = 0.5*s

            # Bomb out if too many halving steps are required
            if j == maxhalves-1:
                print('Warning: maxhalves exceeded. Sammon mapping may not converge...')

            # Evaluate termination criterion
            if abs((E - E_new) / E) < tolfun:
                if display:
                    print('TolFun exceeded: Optimisation terminated')
                break

            # Report progress
            E = E_new
            if display > 1:
                print('epoch = %d : E = %12.10f'% (i+1, E * scale))

        if i == maxiter-1:
            print('Warning: maxiter exceeded. Sammon mapping may not have converged...')

        # Fiddle stress to match the original Sammon paper
        E = E * scale
        
        return [y,E]