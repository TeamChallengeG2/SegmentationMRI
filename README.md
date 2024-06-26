# Team Challenge - Medical Image Analysis

This repository contains a PyTorch implementation used for the Team Challenge project 2023-2024, hosted by the University of Technology Eindhoven and University Utrecht. The objective is to quantify the chest volume and/or spinal length in MR images. To this end, we perform voxel-wise semantic segmentation of the *spine* and the *chest volume*, adhering to our definitions. Our method applies the 3D U-Net on provided MRI data from UMC Utrecht. The predicted segmentations are used to quantify the volume and spinal length. The workflow and usage of our method is described below. 
## Group 2

* Romy Buijs
* Lisa Cornelissen
* Dimo Devetzis
* Daniel Le
* Jiaxin Zhang

[Link](TC_group2_part1.pdf) to proposal part 1.
## Quick usage
While the individual ``.py`` files can be executed as main script to perform their corresponding functionality, it is recommended to run them from `main.py`:

**1. Load dataset** ```scoliosis_dataset.py```
```python
train_set_raw, val_set, test_set = scoliosis_dataset() # Base datasets
train_set = TransformDataset(base_dataset=train_set_raw) # Augmentation in train dataset only!
```
**2. Load model and weights**

Download weights from: https://filesender.surf.nl/?s=download&token=259f5214-b77a-4920-b3dc-022e25237508
```python
model = UNet3D().cuda()
model.load_state_dict(torch.load(R"weights.pth")) # Optionally, load weights
```
**3. Train** ```train.py```
```python
trainer = Trainer(model=model, 
                  train_set=train_set, 
                  val_set=val_set)
trainer.train() # Start training
```
**4. Calculate volume & testing** ```postprocessing.py```
```python
pd_data = calc_scores(test_set, model) # Create pandas data 
df = show_table(pd_data) # Show pandas table
```
**5. Export prediction to file** 
```python
i=2 # index subject
data=test_set[i]
export_plot(image=data[0], # Shows prediction slice by slice in /test_results/
            mask=data[1],
            prediction=model(data[0].unsqueeze(0).unsqueeze(0).cuda()))
```
**6. Visualization 3D mesh**
```python
vobj = Volume(image=data[0], 
              mask=data[1], 
              prediction=model(data[0].unsqueeze(0).unsqueeze(0).cuda()), 
              header=data[2])
vobj.get_objective()
plot_3D_mesh(vobj)
```
## Table of contents
* [Description](#team-challenge---medical-image-analysis)
* [Dependencies](#dependencies)
* [Folder Structure](#folder-structure)
* [Workflow](#workflow)
    * [Config file](#config-file)
    * [Dataset and annotation](#dataset-and-annotation)
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
* [Discussion](#discussion)
          

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
├───logger/ - logger for training
│   └── logger.py│ 
│
├───model/ - model used for training
│   └── UNet3D.py
│
├───test_results/ - contains test plots for segmentation prediction slice by slice
│
├───train_results/ - weights, log files and plots during training for visualization
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
        "splitratio": [0.7, 0.1, 0.2],      // split ratio train, val, test
        "normalize": true,                  // boolean value for normalization
        "extension": ".nrrd",               // extension of data files
        "LP_dimension": 240,                // dimension after resample in LP
        "S_dimension": 16,                  // dimension after resample in S
        "rotation_angle": 10,               // rotation angle, set to 0 for no aug    
        "N_classes": 3                      // number of classes
    },
        
    "trainer": {    
        "batch_size": 1,                    // batch size
        "device": "cuda",                   // selected device for training
        "epochs": 150,                      // number of epochs 
        "decay_lr_after": 100,              // lambda decay after epochs 
        "lr": 5e-4,                         // learning rate
        "loss_fn": "CrossEntropyLoss"       // loss function used for training
    },

    "tester": {
        "batch_size": 1,                    // batch size
        "device": "cuda"                    // selected device for testing
    }
}
```

### Dataset and annotation
Data was obtained at the UMC Utrecht. The data included 19 T2w MRI from adult volunteers and from 19 children between 8 and 10 years. The amount of image slices varied from 12 to 17 slices, with a slice thickness of 4 mm and total gaps ranging from 20 to 25 mm. Pixel spacing ranged from 0.46875 to 0.625 mm. All pixels were isotropic. 

The spinal length was defined between the level of the top of the sternum and T12. The definition of T12 was primarily based on the median arcuate ligament. The most caudal plane with some ventral covering of the aorta was defined as T12. This was done regardless of if the vertebral body was visible. The caudal part of T12 is not always covered by the median arcuate ligament. Therefore, for a second check, the presence of attached ribs, the shape of the vertebrae and its relation to the other vertebrae were also considered.
The most cranial visible part of the sternum was identified and used as the top border of the spinal length. This slice was also used as the top border for the definition of thoracic volume. The bottom border was defined as the most cranial plane where both kidneys were visible. 

Using these borders segmentations of the volume inside the thoracic cage or abdominal cavity were made. For the spinal length the borders of the vertebral bodies were used. All segmentations were made using 3D Slicer. 
![image](https://github.com/TeamChallengeG2/SegmentationMRI/assets/159581756/6df8aae2-f3ae-4908-89bc-d111647b95c9)![image](https://github.com/TeamChallengeG2/SegmentationMRI/assets/159581756/f8d43de9-974d-4ad8-872a-004893eaa512)

<p align="center">
  <img src="visualization/annotation.gif" border=1px/>
</p>


### Data preprocessing, augmentation and splitting

In order to improve generalization and robustness of the model, we perform data augmentation using geometric transformations. Since our dataset consists of axial slices with spacings of 24 mm in the inferior-superior axis, we will only use small random rotations in the range of -10 to 10 degrees around this axis. The ratio of splitting the data into training, validation and testing set is `0.7:0.1:0.2`. To avoid data contamination, no augmentation is performed on the test set.

Additionally, one of the characteristics of the U-Net is that the spatial dimensions of the input are reduced by a factor 2 in each encoder block. More specifically, each dimension must be divisible by $2^n$ where $n$ is the total number of pooling operators in the encoding path. As such, we resampled the depth of the original MRI image to 16, using cubic spline interpolation. The corresponding masks are resampled to the same dimension using nearest-neighbor interpolation. Furthermore, due to computational resources, we also resample the axial dimensions from 640 to 240. The new physical spacings are recalculated and stored, which are used for the volume and spinal length calculations in subsequent analysis.

### 3D U-net architecture
<p align="center">
  <img src="visualization/unet.png" />
</p>
For the segmentation task, the 3D U-Net model is applied. The U-Net is a commonly used architecture in the domain of medical imaging. Although there are varying implementations, the 3D U-Net for example has three encoding and decoding blocks (opposed to four in 2D U-Net). The encoding path captures features through convolutional and max-pooling layers, while the decoding path reconstructs from the compressed representation using transpose-convolution layers combined with skip connections. Skip connections preserve spatial information by concatenating low-level feature maps with high-level feature maps. 

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
Given an input of `(1, 160, 160, 16)` the output is of shape `(3, 160, 160, 16)` where the channel (dim=0) represent the logits. The logits are normalized using a Softmax function, ensuring that the voxel class probabilities sum to 1, defined as:

${\sigma (\mathbf {z})\_{i}= {\frac {e^{z_{i}}}{\sum_{j=1}^ {N} e^{z_j}}}\ \ {\text{ for }}i=1,\dotsc ,N}$

 This probability distribution is then utilized to create a heatmap, visually representing the probability of each voxel belonging to a specific class.

### Training
The segmentation model is trained for 150 epochs with an initial learning rate of `5e-4`. The loss function used is the Cross-Entropy Loss, which is defined as:

${CE(p,q)=-\sum _{x\in {\mathcal {X}}}p(x) \log q(x)}$.
 
where $p$ is the ground truth probability and $q$ the predicted probability.
 
The training time is `6h`  and the weights corresponding to the best validation score is saved and used for subsequent calculations. 


### Model testing
The performance of the trained segmentation model is evaluated on the test set. This test set contains 7 random subjects, without any transformation applied. The qualitative and quantitative results are shown below.
#### Quantitative results
For quantitative results we compute the Dice Similarity Score (DSC), two-sided Hausdorff Distance (HD) and its 95th percentile, precision and recall. The tabel below is an overview of the computed metrics (↑ higher is better, ↓ lower is better). The upper table is for the volume prediction and bottom for the spine.


| |Filename	 |	Volume [L]|	DSC↑|  HD↓  |HD95↓     |	Precision| Recall |
|-|-------------|-------------|--------|-------|----------|----------|--------| 
|0|Volunteer 10 |10.10	     |0.891   |54.375 |27.188	   |0.828     |0.964   |
|1|Volunteer 11 |6.03	     |0.935   |27.188 |1.768	   |0.953	    |0.998   |
|2|EBS_6	      |2.42	     |0.874   |179.408|25.500	   |0.779	    |0.997   |
|3|EBS15	      |2.68	     |0.912   |25.717 |3.333	   |0.928	    |0.999   |
|4|EBS12	      |3.02	     |0.937   |25.500 |2.357	   |0.936	    |0.999   |
|5|Volunteer 28 |4.93	     |0.941   |25.500 |1.667	   |0.964	    |0.999   |
|6|EBS_5	      |2.42	     |0.903   |25.554 |25.500	   |0.892	    |1.000   |

| |Filename	 |Length [cm]  |	DSC↑|  HD↓  |HD95↓     |	Precision| Recall |
|-|-------------|-------------|--------|-------|----------|----------|--------| 
|0|Volunteer 10 |30.6	     |0.794   |55.428 |27.559	   |0.679	    |0.955   |
|1|Volunteer 11 |28.4	     |0.852   |27.245 |4.507	   |0.743	    |1.000   |
|2|EBS_6	      |22.8	     |0.828   |26.039 |4.714	   |0.713	    |0.989   |
|3|EBS15	      |17.6	     |0.869   |8.333  |3.333	   |0.769	    |1.000   |
|4|EBS12	      |19.6	     |0.837   |13.437 |4.714	   |0.719	    |1.000   |
|5|Volunteer 28 |21.5	     |0.860   |25.771 |25.500	   |0.755	    |1.000   |
|6|EBS_5	      |20.2	     |0.878   |25.609 |3.333	   |0.783	    |1.000   |



#### Qualitative results
An example of the qualitative results are shown below. The segmentation corresponds to subject EBS_6, which has the lowest DSC score and HD distance. The corresponding low precision and high recall (for volume) also implies that a large amount of the voxels predicted as volume are incorrectly classified as volume.
![EBS_6](visualization/EBS_6_test.gif)

### Postprocessing
The trained segmentation model is used to make predictions about the corresponding segmentation mask. The output masks have physical gaps of roughly 20 mm (different per subject) between the slices along the inferior-superior axis and are upsampled such that there are no physical gaps in the mask, with the assumption that the slice thickness is 4 mm. A median filter is then applied to smoothen out rough edges. The resulting 3D view of volume and spine can be seen below.

<p align="center">
  <img src="visualization/V11_mesh.gif" border=1px />
</p>

#### Chest volume
To calculate the chest volume, we count the number of voxels corresponding to "volume" in our upsampled 3D mesh and multiply it with the physical volume of a single voxel. The results for calculated volumes are shown in the table above.

#### Spinal length

The spinal length is calculated using the center of mass of the upsampled spine in each slice. The individual distances between center of masses between two slices is calculated, and corrected for physical spacings. These distances are then summed up. 

## Discussion
The median arcuate ligament is formed by the right and left muscular extension (crus) of the diafragm. These crus attach the diaphragm to L2 on the left and L3 on the right. The ligament is at the level of T12 and we thought it would stay around there during growth due to the attachment of the diafragm to the surrounding vertebrae. Therefore it would be a good indicator of where T12 is. We did however not find any literature about this. 

During inspiration the median arcuate ligament moves 8 mm caudally [[1]](#bibliography). Therefore we do introduce an uncertainty, but given our radiological experience this is probably better than defining T12 based only on vertebral anatomy. 

In the provided dataset there were several pelvic kidneys and a horseshoe kidney. Given the incidence of pelvic kidneys (1 in 1000 [[2]](#bibliography)) this is a notable difference. Therefore our definition of the lower boundry for the thoracic volume might not be fully accurate. 


## Bibliography
[1] Stewart R. Reuter, M.D. Eugene F. Bernstein, M.D., Ph.D. "The anatomic basis for respiratory variation in median arcuate ligament compression of the celiac artery". DOI:https://doi.org/10.5555/uri:pii:003960607390305X

[2] G. Bingham, “Pelvic kidney,” StatPearls., https://www.ncbi.nlm.nih.gov/books/NBK563239/ (accessed Mar. 22, 2024). 
