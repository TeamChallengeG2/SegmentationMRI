# Team Challenge - Medical Image Analysis

This repository contains a PyTorch implementation used for the Team Challenge project 2023-2024, hosted by University of Technology Eindhoven and University Utrecht. The objective is to quantify the chest volume and/or spinal length in MR images. To this end, we perform voxel-wise semantic segmentation of the *spine* and the *chest*, adhering to our definitions. Our method applies the 3D U-net on provided MRI data from the ScoliStorm project (UMC Utrecht). The predicted segmentations are then used to quantify the volume and length. The workflow and usage of our method is described below. 
## Group 2

* Romy Buijs
* Lisa Cornelissen
* Dimo Devetzis
* Daniel Le
* Jiaxin Zhang

## Quick usage


## Table of contents
* [Description](#team-challenge---medical-image-analysis)
* [Dependencies](#dependencies)
* [Folder Structure](#folder-structure)
* [Workflow](#workflow)
    * [Config file](#config-file)
    * [Dataset and annotation](#dataset-and-annotation)
    * [Data preprocessing, augmentation and splitting](#data-preprocessing-augmentation-and-splitting)
    * [3D U-net architecture](#3d-u-net-architecture)
    * [Training](#training)
        * [Model weights](#model-weights)
    * [Model output](#model-output)
    * [Model testing](#model-testing)
        * [Qualitative results](#qualitative-results)
        * [Quantitative results](#quantitative-results)
    * [Postprocessing](#postprocessing)
        * [Chest volume](#chest-volume)
        * [Spinal length](#spinal-length)


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
```bash
pip install -r requirements.txt
```
## Folder Structure
```
SegmentationMRI/
├───config.json - holds configuration preprocessing, training, validation and testing
├───main.py - main script to run training, validation and testing
├───postprocessing.py - contains class for volume and spinal length calculation
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

```JSON
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
        "batch_size": 1,                    // batch size
        "device": "cuda",                   // selected device for training
        "epochs": 100,                      // number of epochs 
        "lr": 0.0005,                       // learning rate
        "loss_fn": "CrossEntropyLoss"       // loss function used for training
    },

    "tester": {
        "batch_size": 1,                    // batch size
        "device": "cuda"                    // selected device for testing
    }
}
```

### Dataset and annotation
Describe dataset. Modality and T2 weighted etc. Gaps. Patient info. Source. Physical spacings. Dimensions.
Annotation anatomical boundaries (our definition). Mention 3D slicer.
### Data preprocessing, augmentation and splitting
Small geometric transformation: rotation 10 deg. Resampling due to Unet. Physical spacings.

### 3D U-net architecture
Describe 3D Unet encoding decoding path
### Training
Training parameters. Epochs. Loss function. 
#### Model weights
Weights loading
### Model output
Logits to heatmap
### Model testing
#### Qualitative results
Visualization of data (show images low score and high score)
#### Quantitative results

|Filename |   Volume [mm^3] |   Volume [L] |   DSC (pre-resample) |   DSC |
:-------------|----------------:|-------------:|---------------------:|------:|
EBS_7        |     1.95105e+06 |         1.95 |                0.975 | 0.975 |
|  EBS16        |     2.07702e+06 |         2.08 |                0.976 | 0.977 |
|  EBS11        |     2.7426e+06  |         2.74 |                0.968 | 0.968 |
|  EBS18        |     2.3289e+06  |         2.33 |                0.979 | 0.979 |
|  Volunteer 23 |     5.68862e+06 |         5.69 |                0.912 | 0.917 |
|  EBS_8        |     1.80228e+06 |         1.8  |                0.973 | 0.973 |
|  Volunteer 16 |     8.51858e+06 |         8.52 |                0.985 | 0.985 |

### Postprocessing
#### Chest volume
Describe calculations
#### Spinal length
Describe calculations