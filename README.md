# Team Challenge - Medical Image Analysis

This repository contains a PyTorch implementation used for the Team Challenge project 2023-2024, hosted by University of Technology Eindhoven and University Utrecht. The objective is to quantify the chest volume and/or spinal length in MR images. To this end, we perform voxel-wise semantic segmentation of the *spine* and the *chest volume*, adhering to our definitions. Our method applies the 3D U-Net on provided MRI data from the ScoliStorm project. The predicted segmentations are then used to quantify the volume and length. The workflow and usage of our method is described below. 
## Group 2

* Romy Buijs
* Lisa Cornelissen
* Dimo Devetzis
* Daniel Le
* Jiaxin Zhang

## Quick usage
The individual ``.py`` files can be run as main and have their corresponding functionality. It is recommended however to execute from `main.py`:

**1. Load dataset** ```scoliosis_dataset.py```
```python
config = load_config("config.json") # Load config
train_set_raw, val_set, test_set = scoliosis_dataset(config) # Base datasets
train_set = TransformDataset(train_set_raw, config) # Augmentation in train dataset only!
```
**2. Load model and weights**
```python
model = UNet3D(in_channels=1, num_classes=config["dataloader"]["N_classes"]).cuda()
model.load_state_dict(torch.load(R"weights.pth")) # Optionally, load weights
```
**3. Train** ```train.py```
```python
trainer = Trainer(model, train_set, train_set, config) # Trainer object
trainer.train() # Start training
```
**4. Calculate volume & testing** ```postprocessing.py```
```python
pd_data = calc_volume_dsc_hd(test_set, model) # Create pandas data 
show_table(pd_data) # Show pandas table 
```
**5. Export prediction to file** 
```python
i=2 # index subject
data = train_set[i]
export_plot(image=data[0],
            mask=data[1],
            prediction=model(data[0].unsqueeze(0).unsqueeze(0).cuda()),
            mask_only=False) # Exported to "/example_results/"
```
## Table of contents
* [Description](#team-challenge---medical-image-analysis)
* [Dependencies](#dependencies)
* [Folder Structure](#folder-structure)
* [Workflow](#workflow)
    * [Config file](#config-file)
    * [Dataset and annotation process](#dataset-and-annotation)
    * [Data preprocessing, augmentation and splitting](#data-preprocessing-augmentation-and-splitting)
    * [3D U-net architecture](#3d-u-net-architecture)
    * [Model output](#model-output)
    * [Training](#training)
    * [Model testing](#model-testing)
        * [Qualitative results](#qualitative-results)
        * [Quantitative results](#quantitative-results)
    * [Postprocessing](#postprocessing)
        * [Chest volume](#chest-volume)
        * [Spinal length](#spinal-length)


## Dependencies

* Python==3.10.9 (may work with other versions)
* matplotlib==3.8.2
* monai==1.3.0
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
├───weights.pth - best weights
│
├───data/ - folder for .nrrd and .seg.nrrd data files 
|   ├── EBSx/ - contains scans children 8 - 10
|   └── Volunteer X/ - contains scans volunteers
│
├───dataloader/ - code concerning dataloading 
│   ├── scoliosis_dataset.py - class for loading original dataset
│   └── transform_dataset.py - class for augmentation training data
│ 
├───example_results/ - contains example plots for segmentation prediction
│
├───logger/ - logger for training
│   └── logger.py
│
├───model/ - model used for training
│   └── UNet3D.py
│
├───saved/ - weights, log files and plots during training for visualization
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
Describe dataset. Modality and T2 weighted etc. Gaps. Patient info. Source. Physical spacings. diff dimensions each subject.
Annotation anatomical boundaries (our definition). Mention 3D slicer.

### Data preprocessing, augmentation and splitting

In order to improve generalization and robustness of the model, we perform data augmentation using small geometric transformations. Our dataset, however, consists of axial slices with spacings of 24 mm in the inferior-superior axis. For this reason, we only use small random rotations in the range of -10 to 10 degrees along the inferior-superior axis. This effectively doubles the amount of data in the training set. The ratio of splitting the data into training, validation and testing set is 0.6:0.1:0.3. To avoid data contamination, no augmentation is performed on the test set.

Additionally, one of the characteristics of the U-Net is that the spatial dimensions of the input are reduced by a factor 2 in each encoder block. More specifically, each dimension must be divisible by $2^n$ where $n$ is the total number of pooling operators in the encoding path. As such, we resampled the depth of the original MRI image to 16, using cubic spline interpolation. The corresponding masks are resampled to the same dimension using nearest-neighbor interpolation. Furthermore, due to computational resources, we also resample the axial dimensions from 640 to 160/320. The new physical spacings are recalculated and stored, which are used for the volume and spinal length calculations in subsequent analysis.

### 3D U-net architecture

<a name="3dunet">![3dunet](https://github.com/TeamChallengeG2/SegmentationMRI/assets/159690372/04712ec6-721d-4a04-a748-08922e62c498)</a>

For the segmentation task, we have chosen to utilize the 3D U-Net model. The U-Net is a commonly used architecture in the domain of medical imaging. Although there are varying implementations, the 3D U-Net for example has three encoding and decoding blocks (opposed to for example four in 2D U-Net). The encoding path captures features through convolutional and max-pooling layers, while the decoding path reconstructs from the compressed representation using transpose-convolution layers combined with skip connections. Skip connections preserve spatial information by concatenating low-level feature maps with high-level feature maps. 

The *input* of the model is a grayscale image with shape `(1, 160, 160, 16)` (C, H, W, D). Since the objective is to make a prediction for a voxel belonging to a certain class, the output must contain 3 channels (`N_classes=3`: background, volume, spine). The channels correspond to the logits of a certain class. For the specific architecture refer to <a href="#3dunet">Fig. 1</a> and the model summary details below.

<details closed>

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1     [-1, 32, 160, 160, 16]             896
       BatchNorm3d-2     [-1, 32, 160, 160, 16]              64
              ReLU-3     [-1, 32, 160, 160, 16]               0
            Conv3d-4     [-1, 64, 160, 160, 16]          55,360
       BatchNorm3d-5     [-1, 64, 160, 160, 16]             128
              ReLU-6     [-1, 64, 160, 160, 16]               0
         MaxPool3d-7        [-1, 64, 80, 80, 8]               0
       Conv3DBlock-8      [[-1, 64, 80, 80, 8],
                        [-1, 64, 160, 160, 16]]               0
            Conv3d-9        [-1, 64, 80, 80, 8]         110,656
      BatchNorm3d-10        [-1, 64, 80, 80, 8]             128
             ReLU-11        [-1, 64, 80, 80, 8]               0
           Conv3d-12       [-1, 128, 80, 80, 8]         221,312
      BatchNorm3d-13       [-1, 128, 80, 80, 8]             256
             ReLU-14       [-1, 128, 80, 80, 8]               0
        MaxPool3d-15       [-1, 128, 40, 40, 4]               0
      Conv3DBlock-16     [[-1, 128, 40, 40, 4],
                          [-1, 128, 80, 80, 8]]               0
           Conv3d-17       [-1, 128, 40, 40, 4]         442,496
      BatchNorm3d-18       [-1, 128, 40, 40, 4]             256
             ReLU-19       [-1, 128, 40, 40, 4]               0
           Conv3d-20       [-1, 256, 40, 40, 4]         884,992
      BatchNorm3d-21       [-1, 256, 40, 40, 4]             512
             ReLU-22       [-1, 256, 40, 40, 4]               0
        MaxPool3d-23       [-1, 256, 20, 20, 2]               0
      Conv3DBlock-24     [[-1, 256, 20, 20, 2],
                          [-1, 256, 40, 40, 4]]               0
           Conv3d-25       [-1, 256, 20, 20, 2]       1,769,728
      BatchNorm3d-26       [-1, 256, 20, 20, 2]             512
             ReLU-27       [-1, 256, 20, 20, 2]               0
           Conv3d-28       [-1, 512, 20, 20, 2]       3,539,456
      BatchNorm3d-29       [-1, 512, 20, 20, 2]           1,024
             ReLU-30       [-1, 512, 20, 20, 2]               0
      Conv3DBlock-31     [[-1, 512, 20, 20, 2],
                          [-1, 512, 20, 20, 2]]               0
  ConvTranspose3d-32       [-1, 512, 40, 40, 4]       2,097,664
           Conv3d-33       [-1, 256, 40, 40, 4]       5,308,672
      BatchNorm3d-34       [-1, 256, 40, 40, 4]             512
             ReLU-35       [-1, 256, 40, 40, 4]               0
           Conv3d-36       [-1, 256, 40, 40, 4]       1,769,728
      BatchNorm3d-37       [-1, 256, 40, 40, 4]             512
             ReLU-38       [-1, 256, 40, 40, 4]               0
    UpConv3DBlock-39       [-1, 256, 40, 40, 4]               0
  ConvTranspose3d-40       [-1, 256, 80, 80, 8]         524,544
           Conv3d-41       [-1, 128, 80, 80, 8]       1,327,232
      BatchNorm3d-42       [-1, 128, 80, 80, 8]             256
             ReLU-43       [-1, 128, 80, 80, 8]               0
           Conv3d-44       [-1, 128, 80, 80, 8]         442,496
      BatchNorm3d-45       [-1, 128, 80, 80, 8]             256
             ReLU-46       [-1, 128, 80, 80, 8]               0
    UpConv3DBlock-47       [-1, 128, 80, 80, 8]               0
  ConvTranspose3d-48    [-1, 128, 160, 160, 16]         131,200
           Conv3d-49     [-1, 64, 160, 160, 16]         331,840
      BatchNorm3d-50     [-1, 64, 160, 160, 16]             128
             ReLU-51     [-1, 64, 160, 160, 16]               0
           Conv3d-52     [-1, 64, 160, 160, 16]         110,656
      BatchNorm3d-53     [-1, 64, 160, 160, 16]             128
             ReLU-54     [-1, 64, 160, 160, 16]               0
           Conv3d-55      [-1, 3, 160, 160, 16]             195
    UpConv3DBlock-56      [-1, 3, 160, 160, 16]               0
================================================================
Total params: 19,073,795
Trainable params: 19,073,795
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.56
Forward/backward pass size (MB): 2734.62
Params size (MB): 72.76
Estimated Total Size (MB): 2808.95
----------------------------------------------------------------
```
</details>

### Model output
As mentioned above, given an input of `(1, 160, 160, 16)` the output is of shape `(3, 160, 160, 16)` where the channel (dim=0) represent the logits. The logits are normalized using a Softmax function, ensuring that the voxel class probabilities sum to 1, defined as:

${\displaystyle \sigma (\mathbf {z} )_{i}={\frac {e^{z_{i}}}{\sum _{j=1}^{N}e^{z_{j}}}}\ \ {\text{ for }}i=1,\dotsc ,N}$

 This probability distribution is then utilized to create a heatmap, visually representing the probability of each voxel belonging to a specific class.

### Training
The segmentation model is trained for 150 epochs with an initial learning rate of 0.0005. The loss function used is the Cross-Entropy Loss, which is defined as:

${\displaystyle CE(p,q)=-\sum _{x\in {\mathcal {X}}}p(x)\,\log q(x)}$.
 
where $p$ is the ground truth probability (1 or 0) and $q$ the predicted probability.
 
The training time is `xxxx` s and the weights corresponding to the minimum validation score is saved and used for subsequent calculations. A visualization of the training process is shown below. 
![Training visualization](visualization/visual.gif)

### Model testing
The performance of the trained segmentation model is evaluated on the test set. This test set contains random 11 subjects, without any transformation applied. The qualitative and quantitative results are shown below.
#### Quantitative results
For quantitative results we compute the Dice Similarity Score (DSC), Hausdorff Distance (HD), True Positive Rate (TPR), False Positive Rate (FPR) and False Negative Rate (FNR). The tabel below is an overview of the computed metrics (↑ higher is better, ↓ lower is better). 

|Filename	|Volume [mm^3]|	Volume [L]|	DSC↑|	HD↓	|HD95↓|	FPR|
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
|  EBS_1 | 2819175 | 2.82 | 0.87 | 25.622 | 25.5 | 0.0 |
| Volunteer 6 | 6825093 | 6.83 | 0.987 | 18.75 | 1.875 | 0.002 | 
| EBS19 | 2647950 | 2.65 | 0.983 | 5.0 | 2.5 | 0.0 | 
| EBS14 | 2789100 | 2.79 | 0.973 | 25.5 | 2.5 | 0.001 | 
| Volunteer 10 | 8782300 | 8.78 | 0.986 | 27.188 | 1.875 | 0.005 | 
| Volunteer 11 | 5715370 | 5.72 | 0.991 | 5.303 | 0.0 | 0.001 | 
| EBS_6 | 1684375 | 1.68 | 0.94 | 25.5 | 25.5 | 0.0 | 
| EBS15 | 2524050 | 2.52 | 0.958 | 25.5 | 25.5 | 0.002 | 
| EBS12 | 3180375  | 3.18 | 0.903 | 25.986 | 25.5 | 0.008 | 
| Volunteer 28 | 4697850 | 4.7 | 0.99 | 7.071 | 0.0 | 0.001 | 
| EBS_5 | 1939074 | 1.94 | 0.942 | 25.5 | 25.5 | 0.0 |

#### Qualitative results
An example of the qualitative results are shown below. The segmentation in `Fig X`. corresponds to subject EBS_1, which has the worst DSC score and HD distance. `Fig X.` corresponds to subject EBS_5, which has the best DSC and HD.

### Postprocessing
The trained segmentation model is used to make predictions about the corresponding segmentation mask. The output masks have physical gaps of roughly 20 mm (depending on subject) between the slices along the inferior-superior axis and are upsampled such that there are no physical gaps in the mask, with the assumption that the slice thickness is 4 mm. A median filter is then applied to smoothen out rough edges. The resulting 3D view of volume and spine can be found [here](visualization/mesh_prediction.stl).
#### Chest volume
To calculate the chest volume, we count the number of voxels corresponding to "volume" in our upsampled 3D mesh and multiply it with the physical volume of a single voxel. The results for calculated volumes are shown in the table above.

#### Spinal length

1. Calculate center of mass each pixel
2. Connect center of masses
3. Spline interpolation through points?
4. Calculate distance 

For left/right/anterior/posterior distances --> take most X pixel instead of CoM?


