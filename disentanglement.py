#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:14:01 2019

@author: perfect
"""

import numpy as np
import matplotlib.pyplot as plt





def generate_3_overlapping_gaussians(samples=500):
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    
    x0, y0 = np.random.multivariate_normal(mean, cov, samples).T
    x1, y1 = np.random.multivariate_normal(mean, cov, samples).T
    x2, y2 = np.random.multivariate_normal(mean, cov, samples).T
    
    
    plt.plot(x0, y0, 'x')
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'x')
    
    
    plt.axis('equal')
    plt.show()

    return [x0, y0], [x1, y1], [x2, y2]


def generate_3_non_overlapping_gaussians(samples=500):
    
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    
    mean = [10, 0]
    x0, y0 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 10]
    x1, y1 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 0]
    x2, y2 = np.random.multivariate_normal(mean, cov, samples).T
    
    
    plt.plot(x0, y0, 'x')
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'x')
    
    
    plt.axis('equal')
    plt.show()

    return [x0, y0], [x1, y1], [x2, y2]


def generate_3_almost_overlapping_gaussians(samples=500):

    cov = [[1, 0], [0, 1]]  # diagonal covariance
    
    mean = [10, 0]
    x0, y0 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 2]
    x1, y1 = np.random.multivariate_normal(mean, cov, samples).T
    
    mean = [0, 0]
    x2, y2 = np.random.multivariate_normal(mean, cov, samples).T
    
    
    plt.plot(x0, y0, 'x')
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'x')
    
    
    plt.axis('equal')
    plt.show()

    return [x0, y0], [x1, y1], [x2, y2]



def measure_disentanglement():
    pass



def main():
    gaussians_types = [generate_3_overlapping_gaussians, 
                       generate_3_almost_overlapping_gaussians, 
                       generate_3_non_overlapping_gaussians]
    
    for g_type in gaussians_types:
        [x0, y0], [x1, y1], [x2, y2] = g_type(100)
        gauss0 = np.stack((x0,y0), axis=-1)
        gauss1 = np.stack((x1,y1), axis=-1)
        gauss2 = np.stack((x2,y2), axis=-1)
        
        print(gauss0.shape)



    
    
if __name__ == '__main__':
    main()