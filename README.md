# Decoding and encoding mental representations of objects in the brain

## Background

Neuroimaging, ways of understanding how the brain produces images, produces sets of data that are high-dimensional and complicated. Ways of interpreting this data provides the means for understanding how the brain encodes and decodes images. In this context, encoding refers to predicting the imaging data given external variables, such as stimuli descriptors and decoding refers to learning a model that predicts behavioral or phenotypic variables from fMRI data. With the way these models can be learned and predicted, supervised machine learning methods can be used to decode images to relate brain images to behavioral or clinical observations. Sci-kit learn can be used for this analysis in making predictions that can be cross-validated. 

## What is this?

I re-create the methods of Miyawaki et al. (2008) in inferring visual stimulus from brain activity. In the experiment of Miyawaki et al. (2008) several series of 10×10 binary images are presented to two subjects while activity on the visual cortex is recorded. In the original paper, the training set is composed of random images (where black and white pixels are balanced) while the testing set is composed of structured images containing geometric shapes (square, cross…) and letters. I will use the training set with cross-validation to get scores on unknown data. I can examine decoding (the reconstruction of visual stimuli from fMRI) and encoding (prediction of fMRI data from descriptors of visual stimuli). This would let me look at the relation between stimuli pixels and brains voxels from both angles. The approach uses a support vector classifier and logistic ridge regression in the prediction function in both the decoding and the encoding. 

This repository uses a modified version of the code from: https://github.com/AlexandreAbraham/frontiers2013

## Usage

`python decode.py` for decoding

`python encode.py` for encoding

## Requirements

Nilearn (>0.4.1)

Python (>3.0)

Matplotlib (>=2.2.0)

Numpy 

## References

Miyawaki, Y., Uchida, H., Yamashita, O., Sato, M.-A., Morito, Y., Tanabe, H. C., et al. (2008). Visual image reconstruction from human brain activity using a combination of multiscale local image decoders. Neuron 60, 915–929. doi: 10.1016/j.neuron.2008.11.004
