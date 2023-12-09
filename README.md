# Satellite Imagery to Map Translation with Pix2Pix Architecture

## Overview

This is the code repository based on the paper _Satellite Imagery to Map Translation with Pix2Pix Architecture_.

This was part of the CS 548-12 course taught by Dr. Michael J. Reale at SUNY Polytechnic Institute in Fall 2023.

Contributors: Ryan Primus

In the code, I used a Pix2Pix implementation with PyTorch to translate satellite imagery to maps. Because it was based on Pix2Pix, I used a U-Net Generator and a binary CNN (Convolutional Neural Network) Discriminator for my model.

## Dataset
Download the 'map.tar.gz' dataset from UC Berkeley: [Maps Dataset](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

Place this data in the directory you are working in to access the files.

## Models
Download the trained models from this Google Drive: [Trained Models](PUT THIS HERE)

Open the models in the folder you are working in.

## Software Dependencies
Install [Miniconda](https://repo.anaconda.com/miniconda/) (specific to the device you are running the program on) to have access to all imports used.
The model is built through PyTorch, which comes with Anaconda when installed. 

## Running the Project
The primary files used in the project are:

baseline_train.py -> trains the model

baseline_eval.py -> evaluates the model

You only have to run baseline_eval if you downloaded the trained models listed before.

```
cd $project_location
./baseline_eval.py
```

If you didn't download the trained models, you'll have to run baseline_train, then baseline_eval.
