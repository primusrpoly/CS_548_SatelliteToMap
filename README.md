# Satellite Imagery to Map Translation with Pix2Pix Architecture

## Overview

This is the code repository based on the paper _Satellite Imagery to Map Translation with Pix2Pix Architecture_.

This was part of the CS 548-12 course taught by Dr. Michael J. Reale at SUNY Polytechnic Institute in Fall 2023.

Contributors: Ryan Primus

In the code, I used a Pix2Pix implementation with PyTorch to translate satellite imagery to maps. Because it was based off Pix2Pix, I used a U-Net Generator and a binary CNN (Convolutional Neural Network) Discriminator for my model.

## Dataset
Download the dataset from this Google Drive: [Maps Dataset](https://drive.google.com/file/d/1s5a2UeJR4H_KJ-nV4NmRMkBHr3zn20Tf/view)

Place this data directly into the folder you are working in to access the files.

## Models
Download the trained models from this Google Drive: [Trained Models](PUT THIS HERE)

Open the models in the folder you are working in.

## Software Dependencies
Install [Miniconda](https://repo.anaconda.com/miniconda/) (specific to the device you are running the program on) to have access to all imports used.

## Running the Project
The primary files used in the project are:

baseline_train.py -> this trains the model

baseline_eval.py -> this evaluates the model

You only have to run baseline_eval if you downloaded the trained models listed before.

```
cd $project_location
./baseline_eval.py
```

If you didn't download the trained models, you'll have to run baseline_train, then baseline_eval.
