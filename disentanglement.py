# Copyright (c) 2018, 
#
# authors: Luca Celotti
# during their PhD at Universite' de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import matplotlib.pyplot as plt
import time

import seaborn as sns
import sklearn.cluster as cluster
from sklearn.metrics.cluster import v_measure_score


#%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


def generate_3_overlapping_gaussians(samples=500):
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    
    x0, y0 = np.random.multivariate_normal(mean, cov, samples).T
    x1, y1 = np.random.multivariate_normal(mean, cov, samples).T
    x2, y2 = np.random.multivariate_normal(mean, cov, samples).T

    return [x0, y0], [x1, y1], [x2, y2]


def generate_3_non_overlapping_gaussians(samples=500):
    
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    
    mean = [10, 0]
    x0, y0 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 10]
    x1, y1 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 0]
    x2, y2 = np.random.multivariate_normal(mean, cov, samples).T

    return [x0, y0], [x1, y1], [x2, y2]


def generate_3_almost_overlapping_gaussians(samples=500):

    cov = [[1, 0], [0, 1]]  # diagonal covariance
    
    mean = [10, 0]
    x0, y0 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 2]
    x1, y1 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 0]
    x2, y2 = np.random.multivariate_normal(mean, cov, samples).T

    return [x0, y0], [x1, y1], [x2, y2]



def measure_disentanglement():
    pass



def main():
    n_samples = 100
    gaussians_types = [generate_3_overlapping_gaussians, 
                       generate_3_almost_overlapping_gaussians, 
                       generate_3_non_overlapping_gaussians
                       ]
    
    n_studies = len(gaussians_types)
    
    
    # algorithm used for classification
    algorithm = cluster.KMeans
    
    f, axarr = plt.subplots(2, n_studies, sharex=True, sharey=True)
    f.suptitle('Clusters found by {}\n and v measure'.format(str(algorithm.__name__)), fontsize=12)
    for i, g_type in enumerate(gaussians_types):
        [x0, y0], [x1, y1], [x2, y2] = g_type(n_samples)
        gauss0 = np.stack((x0,y0), axis=-1)
        gauss1 = np.stack((x1,y1), axis=-1)
        gauss2 = np.stack((x2,y2), axis=-1)
        
        axarr[0,i].plot(gauss0[:,0], gauss0[:,1], 'x')
        axarr[0,i].plot(gauss1[:,0], gauss1[:,1], 'x')
        axarr[0,i].plot(gauss2[:,0], gauss2[:,1], 'x')
            
        #axarr[i, 0].axis('equal')
        #plt.show()

        trueLabels = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)
        data = np.concatenate((gauss0, gauss1, gauss2), axis=0)

        
        clusteringLabels = algorithm(n_clusters=3).fit_predict(data)
        palette = sns.color_palette('deep', np.unique(clusteringLabels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in clusteringLabels]
        axarr[1,i].scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        
        
        
        
        #print(clusteringLabels)
        #print(trueLabels)
        measure = v_measure_score(clusteringLabels, trueLabels)
        print(measure)
        axarr[0,i].set_title(round(measure,2), fontsize=12)
        
    rows = ['True\nLabels', '{}\nLabels'.format(str(algorithm.__name__))]
    
    for ax, row in zip(axarr[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')

    f.savefig("disentangled.pdf", bbox_inches='tight')
    
    
    
    
if __name__ == '__main__':
    main()