# Deep sparse representation-based classification
![overview](data/MBarchitecture.pdf)

## Abstract
Recent advances in acquisition and display technologies have led to an enormous amount of visual data, which requires appropriate storage and management tools. One of the fundamental needs is the design of efficient image classification and recognition solutions. In this paper, we propose a wavelet neural network approach for sparse representation-based object classification. The proposed approach aims to exploit the advantages of sparse coding, multi-scale wavelet representation as well as neural networks. More precisely, a wavelet transform is firstly applied to the image datasets. The generated approximation and detail wavelet subbands are then fed into a multi-branch neural network architecture. The latter allows us to produce multiple sparse codes that are efficiently combined during the classification stage. Extensive experiments, carried out on various types of standard object datasets, have shown the efficiency of the proposed methods compared to the existing sparse coding and deep learning-based methods.

## Citation

Please use the following to refer to this work:

<pre><code>
@ARTICLE{MB-WNN-SRC, 
author={Tan-Sy Nguyen, Marie Luong, Mounir Kaaniche, Long H. Ngo, Azeddine Beghdadi}, 
title={A novel multi-branch wavelet neural network for sparse representation based object classificatio}, 
year={2022}, 
</code></pre>


## Setup:
### Dependencies:
Tensorflow, numpy, scipy, random, argparse.
### Data preprocessing:

Save the data in a `.mat` file that includes verctorized features in a `1024xN` matrix with the name `features` and labels in a vector with the name `Label`.

A sample preprocessed dataset is available in: `data/UMD-AA01.mat` 

### Note:
To keep the regularization parameters valid, please make sure that the preprocessing stage is done correctly. Also, for large datasets since the batch size will be larger, the learning rate (or the maximum number of iterations) may need to be adapted accordingly. 







