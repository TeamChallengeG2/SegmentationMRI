# Team Challenge - Image Analysis

## Group 2

* Romy Buijs
* Lisa Cornelissen
* Dimo Devetzis
* Daniel Le
* Jiaxin Zhang

## Project 
[Link to paper]()

## Requirements

* Python == 3.10.9 (may work with other versions)
* PyTorch == 2.0.1+cu117

## Usage
WIP

## Folder Structure
VolumeSegmentationTC2/
├───config.json - holds configuration for training, validation and testing
├───main.py - main script to run training
├───train.py - class for training
│
├───data/ - folder for DICOM data files wrapped in .zip
│
├───dataloader/ - code concerning dataloading 
│   └── datasets.py - class for loading and preprocessing of data
│
├───logger/ - logger for training
│   └── logger.py
│
├───model/ - model used for training
│   └── UNet.py
│
├───saved/ - generated log files
│
└───utils/ - utility methods 
    └── utils.py

