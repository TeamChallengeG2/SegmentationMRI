# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

import torch
import numpy as np
import glob
import scienceplots
import nrrd
import slicerio
from torch.utils.data import Dataset, random_split
from scipy.ndimage import zoom

class ScoliosisDataset(Dataset):
    """
    Dataset class to load and preprocess ScoliStorm dataset. 
    Inherited child class from torch.Dataset.
    """    

    def __init__(self, config):
        """Init method for ScoliosisDataset class.

        Args:
            config (collections.OrderedDict): OrderedDict config from .json file
        """        
        cfg = config["dataloader"]
        data_dir = cfg["data_dir"]                      
        self.extension = cfg["extension"]               
        self.normalize = cfg["normalize"]               # boolean value normalizing
        self.LP_dimension = cfg["LP_dimension"]         # LP-dim after resampling 
        self.S_dimension = cfg["S_dimension"]           # S-dim after resampling
        self.N_classes = cfg["N_classes"]               # number of classes
        self.data_paths = self.dir_to_list(data_dir)    # list of filepaths
        self.length = len(self.data_paths)              # dataset size
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Retrieves item and resamples by index.

        Arguments:
            index (int): index number

        Raises:
            IndexError: index > length of dataset

        Returns:
            img
                Torch image tensor. 
            mask
                Torch mask tensor. 
            header
                collections.OrderedDict
            filename
                filename string
        """
        if index >= self.length:
            raise IndexError(f"index should be smaller than {self.length}")
                
        img, mask, header = self.path_to_tensor(self.data_paths[index]) # Get tensor data from datapath
        img, mask = self.resample(img, mask) # Resample to U-Net valid dimensions
            
        if self.normalize:
            img -= torch.min(img)
            img /= torch.max(img)
                                      
        filename = self.data_paths[index][0].split("\\")[1]
        return img, mask, header, filename
    
    def dir_to_list(self, data_dir):
        """
        Retrieves path from data directory and returns as list of strings [img, mask].

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
            mask_path = glob.glob(file_path + "/*.seg" + self.extension)[0] 
            data.append([img_path, mask_path])
        return data
    
    def path_to_tensor(self, file_name):
        """
        Extracts torch array data from .nrrd files given a list of one image and one mask path.

        Args:
            file_name (list): list of strings path [img, mask]

        Raises:
            ValueError: if number of classes in mask does not equal config setting

        Returns:
            img
                Torch image tensor. 
            mask
                Torch mask tensor. 
            header
                collections.OrderedDict

        """
        img, header = nrrd.read(file_name[0])
        img = img.astype(np.float32)
        mask, mask_header = nrrd.read(file_name[1])
        segmentation_info = slicerio.read_segmentation_info(file_name[1])
        segment_names_to_labels = [("Background", 0), ("Volume", 1)]
        if self.N_classes==3:
            segment_names_to_labels.append(("Spine", 2))

        mask, _ = slicerio.extract_segments(mask, mask_header, segmentation_info,
                                            segment_names_to_labels)

        if len(np.unique(mask))!=self.N_classes:
            raise ValueError(f"number of classes in mask does not equal {self.N_classes}, change in config!")
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float(), header

    def resample(self, img, mask):
        """Resamples input image and mask to given dimensions in config.

        Args:
            img (torch.Tensor): input image
            mask (torch.Tensor): input mask

        Returns:
            img
                Torch image tensor. 
            mask
                Torch mask tensor. 
        """                
        img = zoom(input=img, 
                   zoom=(self.LP_dimension/img.shape[0], 
                         self.LP_dimension/img.shape[1], 
                         self.S_dimension/img.shape[2]))
        
        mask = zoom(input=mask,
                    zoom=(self.LP_dimension/mask.shape[0],
                          self.LP_dimension/mask.shape[1],
                          self.S_dimension/mask.shape[2]), 
                    order=0, 
                    mode="nearest")
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
    
def scoliosis_dataset(config):
    """Splits the dataset and returns as train/val/test set.

    Arguments:
        config (dictionary): contains config parameters

    Returns:
        train_set, val_set, test_set: subset of dataset objects
    """    
    dataset = ScoliosisDataset(config)
    train_set_raw, val_set, test_set = random_split(dataset=dataset,
                                                    lengths=config["dataloader"]["splitratio"], 
                                                    generator=torch.Generator().manual_seed(42))
    return train_set_raw, val_set, test_set
    
if __name__=="__main__":
    from utils import load_config
    import matplotlib.pyplot as plt

    config = load_config("config.json")     # Load config
    train_set_raw, val_set, test_set = scoliosis_dataset(config) # Base datasets
    plt.imshow(train_set_raw[0][0][:,:,10], "gray")