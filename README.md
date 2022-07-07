# Deep sparse representation-based classification
![overview](https://user-images.githubusercontent.com/18729506/57503787-b905dd00-72bf-11e9-968f-a66010572b26.png)

## Abstract
Recent advances in acquisition and display technologies have led to an enormous amount of visual data, which requires appropriate storage and management tools. One of the fundamental needs is the design of efficient image classification and recognition solutions. In this paper, we propose a wavelet neural network approach for sparse representation-based object classification. The proposed approach aims to exploit the advantages of sparse coding, multi-scale wavelet representation as well as neural networks. More precisely, a wavelet transform is firstly applied to the image datasets. The generated approximation and detail wavelet subbands are then fed into a multi-branch neural network architecture. The latter allows us to produce multiple sparse codes that are efficiently combined during the classification stage. Extensive experiments, carried out on various types of standard object datasets, have shown the efficiency of the proposed methods compared to the existing sparse coding and deep learning-based methods.

## Citation

Please use the following to refer to this work:

<pre><code>
@ARTICLE{dsrc, 
author={M. {Abavisani} and V. M. {Patel}}, 
journal={IEEE Signal Processing Letters}, 
title={Deep Sparse Representation-Based Classification}, 
year={2019}, 
volume={26}, 
number={6}, 
pages={948-952}, 
doi={10.1109/LSP.2019.2913022}, 
ISSN={1070-9908}, 
month={June},}
</code></pre>

M. Abavisani and V. M. Patel, "Deep Sparse Representation-Based Classification," in IEEE Signal Processing Letters, vol. 26, no. 6, pp. 948-952, June 2019.


## Setup:
### Dependencies:
Tensorflow, numpy, scipy, random, argparse.
### Data preprocessing:

Save the data in a `.mat` file that includes verctorized features in a `1024xN` matrix with the name `features` and labels in a vector with the name `Label`.

A sample preprocessed dataset is available in: `data/UMD-AA01.mat` 

### Note:
To keep the regularization parameters valid, please make sure that the preprocessing stage is done correctly. Also, for large datasets since the batch size will be larger, the learning rate (or the maximum number of iterations) may need to be adapted accordingly. 







