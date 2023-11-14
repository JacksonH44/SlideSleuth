# SlideSleuth

## Description
SlideSleuth is a tool to analyze large whole slide image (WSI) datasets of lung adenocarcinoma (LUAD) via feature extraction and unsupervised learning. Specifically, SlideSleuth uses each slide image as input to a variational autoencoder (VAE), then clusters made by the VAE are analyzed. Within the clusters, we aim to identify biomarkers/cancer drivers for LUAD. 

The tool includes pipelines that prepare WSI datasets for both a supervised classifier and a variational autoencoder.

Currently, the tool is still in active development. As of right now, only the data pipeline has been built. The development languages are Python and R, and bash. Pipelining and development are done with the help of Tensorflow, Openslide, and the R package Bioconductor. Containerization is done with Apptainer (formerly Singularity).

## Table of Contents
* [Installation and Setup Instructions](#install)
* [Use Instructions](#use)
* [Data Availability](#data-availability)
* [Contact](#contact)

## Installation Instructons <a name="install"></a>

### Requirements
* [NumPy 1.17+](https://numpy.org/)
* [NVIDIA GPU](https://www.nvidia.com/en-us/)
* [Openslide Python](https://openslide.org/api/python/)
* [Openslide](https://openslide.org/)
* [pandas](https://pandas.pydata.org/)
* [PIL](https://pillow.readthedocs.io/en/stable/)
* [Python 3.7+](https://www.python.org/downloads/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SciPy](https://scipy.org/)
* [Ubuntu](https://ubuntu.com/)

### Setup Instructions
#### Compute Canada
Run the command `./setup.sh`, followed by the command `source ENV/bin/activate` in the same directory. This will install all necessary dependencies.

## Use Instructions <a name="use"></a>
The project source code is divided into 4 main sections: [features](#features) - code to tile images and sort the images into , [data](#data) - code to perform model-specific post-processing on the data, [models](#models) - code to train and test supervised and unsupervised models on the data, and [visualization](#visualization) - code to visualize the trained model performance.

### Features <a name="features"></a>
Assuming the tiled images are in `folder`, execution of the script `src/data/cvae_data_pipeline.sh` with `folder` as the `DIR_PATH` global variable will reorganize the data into a format that is readable by Tensorflow's data pipeline APIs. Similar to the data step, this step has been done by Jackson already for the UHN dataset (it is a little bit time consuming). 

### Data <a name="data"></a>
Assuming the use case of the UHN private dataset that this project was developed with, the script `src/features/tile_uhn_binary.sh` should be run to make tiles from raw slide images. In the case of the UHN dataset, this has been done by Jackson already and may save some time if you contact him about transferring the data (assuming you have permission to view the data).

### Models <a name="models"></a>
Once the data is processed, the convolutional variational autoencoder can be trained by running `src/models/train_cvae.sh` with the desired `DIR_PATH`, `SAVE_PATH`, and `FIG_PATH` global variables. 

### Visualization <a name="visualization"></a>
Once the model is trained, the autoencoder can reconstruct a sample of images by calling `src/visualization/analyze_cvae.sh`. 

## Data Availability <a name="data-availability"></a>
The dataset used for the current iteration of this tool is a private dataset from UHN. Please contact the authors for inquiries regarding data availability. Other test datasets were used during development, mainly the TCGA-BRCA and TCGA-PAAD projects from [GDC](https://portal.gdc.cancer.gov/).

## Contact <a name="contact"></a>
Please contact <j2howe@uwaterloo.ca> for any questions, concerns, bug fixes, or further clarifications. 