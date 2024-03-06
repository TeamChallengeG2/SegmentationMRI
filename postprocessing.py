from scipy.ndimage import zoom
import torch
import numpy as np
from utils import calc_dsc
from tqdm import tqdm

class Volume():
    def __init__(self, image, mask, prediction, header):
        self.header = header
        self.image = image
        self.mask = mask
        self.prediction = prediction
        self.slice_thickness = 4

    def prediction_to_binary(self):
        """
        Converts raw logits from prediction to probability map.
        """        
        prob = torch.softmax(self.prediction, dim=1)
        return np.argmax(prob.detach().cpu().squeeze(), axis=0)

    def get_volume(self):
        """Returns volume in mm^3 for an input image.

        Returns:
            float: volume in mm^3
        """        
        self.S_dimension = self.get_new_S_dimension(self.header)
        self.prediction_mask = self.prediction_to_binary()
        self.dsc_preresample = calc_dsc(self.prediction_mask, self.mask)
        self.image, self.mask, self.prediction_mask = self.resample(self.image, self.mask, self.prediction_mask)
        nr_volume_voxels = np.count_nonzero(self.prediction_mask == 1)
        [L_dim, P_dim, S_dim] = self.get_new_spacings(self.header, self.prediction_mask.shape)
        volume_voxel = L_dim*P_dim*S_dim
        return nr_volume_voxels * volume_voxel

    def get_new_S_dimension(self, header):
        """Method to determine new dimension size in superior axis

        Args:
            header (OrderedDict): Dictionary header from .nrrd file

        Returns:
            int: dimension in S direction
        """        
        return self.get_original_spacings(header)[2]*header['sizes'][2] / self.slice_thickness

    def get_new_spacings(self, header, image_shape):
        """Recalculates new spacings between voxels from image dimensions.

        Args:
            header (OrderedDict): Dictionary header from .mha file
            image_shape (tuple): image dimensions

        Returns:
            list: new dimensions in LPS axes
        """        
        LP_spacing = self.get_original_spacings(header)[0]*header['sizes'][0] / image_shape[0]
        S_spacing = self.get_original_spacings(header)[2]*header['sizes'][2] / image_shape[2]
        return [LP_spacing, LP_spacing, S_spacing]     
       
    def get_original_spacings(self, header):
        """Returns voxel physical spacings from header file

        Args:
            header (OrderedDict): Dictionary header from .mha file

        Returns:
            list: original dimensions in LPS axes
        """        
        LP_spacing = header['space directions'][0][0]
        S_spacing = header['space directions'][2][2]
        return [LP_spacing, LP_spacing, S_spacing]
    
    def resample(self, img, mask, prediction_mask):
        """Resamples input to dimensions without slice gaps.

        Args:
            img (numpy.ndarray): input image
            mask (numpy.ndarray): input mask
            prediction (numpy.ndarray): prediction

        Returns:
            numpy.ndarray: resampled img, mask and prediction
        """        
        img = zoom(input=img, zoom=(1, 1, self.S_dimension/img.shape[-1]))
        mask = zoom(input=mask, zoom=(1, 1, self.S_dimension/mask.shape[-1]), order=0, mode="nearest")
        prediction_mask = zoom(input=prediction_mask, zoom=(1, 1, self.S_dimension/prediction_mask.shape[-1]), order=0, mode="nearest")
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float(), torch.from_numpy(prediction_mask).float()
    
def calc_volumes(dataset, model):
    """Returns list with volume and DSC score for a specific dataset.

    Args:
        dataset (list): list of [img, mask, header, filefolder]
        model: UNet3D

    Returns:
        list: containing volume and DSC scores
    """    
    data = list()
    for img, mask, header, filefolder in tqdm(dataset):
        pred = model(img.unsqueeze(0).unsqueeze(0).cuda())
        volume_object = Volume(img, mask, pred, header)
        volume_mm3 = volume_object.get_volume()
        volume_L = volume_mm3 / 1000000 
        dsc_preresample = volume_object.dsc_preresample
        dsc = calc_dsc(volume_object.prediction_mask, volume_object.mask)
        data.append([filefolder, round(volume_mm3, 2), round(volume_L, 2), round(dsc_preresample.item(), 3), round(dsc.item(), 3)])
    return data
