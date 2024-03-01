# Team Challenge - Medical Image Analysis

This repository contains a PyTorch implementation used for the Team Challenge project 2023-2024, hosted by University of Technology Eindhoven and University Utrecht. The task is to perform voxel-wise semantic segmentation of the *spine* and the *chest* with Magnetic Resonance Images (MRI). The predicted segmentations are used to quantify the spinal length and chest volume, adhering to our definitions. Our method applies the 3D U-net on provided MRI data from the ScoliStorm project (UMC Utrecht). The workflow and usage of our method is described below. 
## Group 2

* Romy Buijs
* Lisa Cornelissen
* Dimo Devetzis
* Daniel Le
* Jiaxin Zhang

## Table of contents
* [Description](#team-challenge---medical-image-analysis)
* [Dependencies](#dependencies)
* [Folder Structure](#folder-structure)
* [Dataset and annotation process]()
* [Workflow](#workflow)
    * [Config file](#config-file)
    * [Dataset and annotation]()
    * [Data preprocessing and augmentation]()
    * [Data splitting]()
    * [3D U-net architecture]()
    * [Training]()
        * [Model weights]()
    * [Model output]()
    * [Model testing]()
        * [Qualitative results]()
        * [Quantitative results]()
    * [Postprocessing]()
        * [Chest volume]()
        * [Spinal length]()


## Dependencies

* Python==3.10.9 (may work with other versions)
* matplotlib==3.8.2
* numpy==1.25.2
* pandas==2.2.1
* pynrrd==1.0.0
* scipy==1.12.0
* slicerio==0.1.8
* torch==2.0.1+cu117
* torchvision==0.15.2+cu117
* tqdm==4.66.2

`pip install -r requirements.txt`

## Folder Structure
```
SegmentationMRI/
├───config.json - holds configuration preprocessing, training, validation and testing
├───main.py - main script to run training, validation and testing
├───train.py - class for training (, validation and testing)
│
├───data/ - folder for .nrrd and .seg.nrrd data files 
|   ├── EBSx/ - contains scans children 8 - 10
|   └── Volunteer X/ - contains scans volunteers
│
├───dataloader/ - code concerning dataloading 
│   └── datasets.py - class for loading and preprocessing of data
│
├───logger/ - logger for training
│   └── logger.py
│
├───model/ - model used for training
│   └── UNet3D.py
│
├───saved/ - log files and plots for visualization
│
└───utils/ - utility methods 
    ├── transforms.py - class for geometric transformations
    └── utils.py - contains visualization and config loading methods
```

## Workflow

### Config file 
The config file is in `.json` file format and contains parameters used for data preprocessing, training and testing.

```
{
    "dataloader": {
        "data_dir": "data/",                // path to .nrrd directory
        "normalize": false,                 // boolean value for normalization
        "extension": ".nrrd",               // extension of data files
        "LP_dimension": 160,                // dimension after resample in LP
        "S_dimension": 16,                  // dimension after resample in S
        "rotation_angle": 10                // rotation angle, set to 0 for no aug
    },
        
    "trainer": {    
        "batch_size": 1,                    
        "device": "cuda",                   
        "epochs": 100,
        "lr": 0.0005,                       // learning rate
        "loss_fn": "CrossEntropyLoss"       // loss function used for training
    },

    "tester": {
        "batch_size": 1,
        "device": "cuda"
    }
}
```

