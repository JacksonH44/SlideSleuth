# SlideSleuth

## Description
SlideSleuth is a tool to analyze large whole slide image (WSI) datasets of lung adenocarcinoma (LUAD) via feature extraction and unsupervised learning. Specifically, SlideSleuth uses each slide image as input to a variational autoencoder (VAE), then clusters made by the VAE are analyzed. Within the clusters, we aim to identify biomarkers/cancer drivers for LUAD. 

The tool includes pipelines that prepare WSI datasets for both a supervised classifier and a variational autoencoder.

Currently, the tool is still in active development. As of right now, only the data pipeline has been built. The development languages are Python and R, and bash. Pipelining and development are done with the help of Tensorflow, Openslide, and the R package Bioconductor. Containerization is done with Apptainer (formerly Singularity).

## Table of Contents
* [Installation Instructions](#install)
* [Use Instructions](#use)
* [Credits](#credits)
* [Extra Points](#extra)

## Installation Instructons <a name="install"></a>

### Requirements
* [NumPy 1.13+](https://numpy.org/)
* [NVIDIA GPU](https://www.nvidia.com/en-us/)
* [Openslide Python](https://openslide.org/api/python/)
* [Openslide](https://openslide.org/)
* [pandas](https://pandas.pydata.org/)
* [PIL](https://pillow.readthedocs.io/en/stable/)
* [Python 3.7+](https://www.python.org/downloads/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SciPy](https://scipy.org/)
* [Ubuntu](https://ubuntu.com/)

## Use Instructions <a name="use"></a>

## Credits <a name="credits"></a>

## Extra Points <a name="extra"></a>