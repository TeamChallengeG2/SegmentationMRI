# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

from torch.utils.data import Dataset
from torchvision.transforms import v2
from matplotlib.widgets import Button, Slider
from scipy.ndimage import zoom
import nibabel as nib
import torch
import numpy as np
import glob
import scienceplots
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nrrd
import slicerio
import random
import scipy



class Dataset(Dataset):
    """
    Dataset class to load and preprocess image data from file. 
    Inherited and adheres to torch.Dataset superclass.

    Arguments:
        Dataset -- torch.Dataset object
    """    

    def __init__(self, config,logger):
        data_dir = config["dataloader"]["data_dir"]
        self.extension = config["dataloader"]["extension"]
        self.normalize = config["dataloader"]["normalize"]
        self.LP_dimension = config["dataloader"]["LP_dimension"]
        self.S_dimension = config["dataloader"]["S_dimension"]
        self.data_paths = self.dir_to_list(data_dir)
        self.length = len(self.data_paths)
        self.transforms = [None]
        self.seed = np.random.randint(2147483647)
        self.logger = logger
        
    def dir_to_list(self, data_dir):
        """
        Retrieves path from data directory and returns as list of [img, mask].

        Parameters
        ----------
        data_dir : String
            Directory path to image data from root, i.e., "/data/"

        Returns
        -------
        data : List
            List of [image path, mask path] for each image.

        """
        file_list = glob.glob(data_dir + "*")
        data = list()
        for file_path in file_list:
            img_path = glob.glob(file_path + "/*" + self.extension)[0]
            mask_path = glob.glob(file_path + "/*.seg" + self.extension)[0] # segmentation mask
            data.append([img_path, mask_path])
        return data
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        @override
        Method which returns image and mask data by index, modified to perform 
        intended transformation.
        
        Parameters
        ----------
        index : Integer
            Natural number. Index of image in dataset.

        Returns
        -------
        img
            Torch image tensor. LPS coordinate system.
        mask
            Torch mask tensor. LPS coordinate system.
            
        """
        if index >= self.length:
            f"index should be smaller than {self.length}"
            raise IndexError(f"index should be smaller than {self.length}")
        index_transforms = index // len(self.data_paths)
        transform = self.transforms[index_transforms]
    
        if index >= len(self.data_paths):
            index = (index - len(self.data_paths)) % len(self.data_paths)
                
        img, mask = self.path_to_tensor(self.data_paths[index])
        img, mask = self.resample(img, mask)

        if transform is not None:
            img, mask = transform(img, mask, index)
            
        if self.normalize:
            img -= torch.min(img)
            img /= torch.max(img)
                                      
        # print(f"Transform: {transform}")
        # print(f"Index true: {index}")
            
        return img, mask
    
    def resample(self, img, mask):
        img = zoom(input=img, zoom=(self.LP_dimension/img.shape[0], self.LP_dimension/img.shape[0], self.S_dimension/img.shape[-1]))
        mask = zoom(input=mask, zoom=( self.LP_dimension/mask.shape[0],  self.LP_dimension/mask.shape[0], self.S_dimension/mask.shape[-1]), order=0, mode="nearest")
        self.logger.save_fig_slice(img)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


    def collate_fn(self, batch):
        """Collate (collect and combine) function for varying input size."""
        return batch
    
    def path_to_tensor(self, file_name):
        """
        Extracts torch array data from .nrrd files given a list of one image and one mask path.
        Additionally, sets volume label to 1, and background to 0.

        Parameters
        ----------
        file_name : list
            List of string path to image and mask, i.e., [img_path, mask_path]

        Returns
        -------
        tensor_data : Tensor
            Tensor object of torch module containing image data.

        """
        img, _ = nrrd.read(file_name[0])
        # mask, _ = nrrd.read(file_name[1])
        segmentation_info = slicerio.read_segmentation_info(file_name[1])
        segment_names_to_labels = [("Background", 0), ("Volume", 1)]
        mask, mask_header = nrrd.read(file_name[1])
        mask, _ = slicerio.extract_segments(mask, mask_header, segmentation_info, segment_names_to_labels)
        # img = np.transpose(img, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
    
    def augment_all(self, transform):
        """
        Artificially increases dataset length and saves transform.

        Parameters
        ----------
        transform : torchvision.transforms.v2 module
            A data transformation module from Torchvision

        Returns
        -------
        None.

        """
        self.length += len(self.data_paths)
        self.transforms.append(transform)
        print(f"Augmentation done. Total images: {self.length}")

    def get_original_spacings(self, index):
        file_name = self.data_paths[index]
        _, h = nrrd.read(file_name[0])
        LP_spacing = h['space directions'][0][0]
        S_spacing = h['space directions'][2][2]
        return [LP_spacing, LP_spacing, S_spacing]

    def get_new_spacings(self, index):
        file_name = self.data_paths[index]
        _, h = nrrd.read(file_name[0])
        LP_spacing = self.get_original_spacings(index)[0]*h['sizes'][0] / self.LP_dimension  
        S_spacing = self.get_original_spacings(index)[2]*h['sizes'][2] / self.S_dimension
        return [LP_spacing, LP_spacing, S_spacing]
            
    