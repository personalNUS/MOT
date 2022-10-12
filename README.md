# MOT

Repo to explore and experiment with MOT for DarkNUS

# Object Detection Model

## Training

The Object Detection Model was trained in Google Colab, using crowdhuman data from Roboflow.
The notebook can be found [here](https://colab.research.google.com/drive/1-7aaSzQcf3iA7Rfu_nVhmwwEmEdZWmiR#scrollTo=1QP1FCSLb7ct)

# Tracking Algorithm

There will be two tracking algorithms:
1. DeepSort
2. Tracktor++

## Feature Extraction

TBA

## Tracking

TBA

# Integration

The different modules will be integrated together in `app.py`. 
- Each Tracking Algorithm should be created in its own folder, i.e. `DeepSort` and/or `Tracktor`
- The Object Detection Model is within the `model` folder