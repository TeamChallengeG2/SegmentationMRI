import torch
import numpy as np
import pandas as pd
import seg_metrics.seg_metrics as sg
from utils import calc_dsc, calc_hd95
from tqdm import tqdm
from scipy.ndimage import zoom, median_filter

class Volume():
    def __init__(self, image, mask, prediction, header):
        self.header = header
        self.image = image
        self.mask = mask
        self.prediction = prediction
        self.slice_thickness = 4

    def prediction_to_mask(self):
        """
        Converts raw logits from prediction to binary map.
        """        
        prob = torch.softmax(self.prediction, dim=1)
        return np.argmax(prob.detach().cpu().squeeze(), axis=0)

    def get_volume(self):
        """Returns volume in mm^3 for an input image.

        Returns:
            float: volume in mm^3
        """        
        self.prediction_mask = self.prediction_to_mask()
        labels = np.unique(self.mask).tolist()
        spacings = self.get_new_spacings(self.header, self.prediction_mask.shape)

        self.metrics = sg.write_metrics(labels=labels,  # exclude background if needed
                        gdth_img=self.mask.numpy(),
                        pred_img=self.prediction_mask.numpy(),
                        spacing=spacings,
                        metrics=['hd', 'hd95','dice','fpr','fnr'])
        
        self.image, self.mask, self.prediction_mask = self.resample(self.image, self.mask, self.prediction_mask)
        nr_voxels = np.count_nonzero(self.prediction_mask == 1)
        [L_dim, P_dim, S_dim] = self.get_new_spacings(self.header, self.prediction_mask.shape)
        volume_voxel = L_dim*P_dim*S_dim
        return nr_voxels * volume_voxel

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
        self.S_dimension = self.get_new_S_dimension(self.header)        
        img = zoom(input=img, zoom=(1, 1, self.S_dimension/img.shape[-1]))
        mask = zoom(input=mask, zoom=(1, 1, self.S_dimension/mask.shape[-1]), order=0, mode="nearest")
        prediction_mask = zoom(input=prediction_mask, zoom=(1, 1, self.S_dimension/prediction_mask.shape[-1]))
        prediction_mask = median_filter(prediction_mask, 5)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float(), torch.from_numpy(prediction_mask).float()
    
def calc_volume_dsc_hd(dataset, model):
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
        metric = volume_object.metrics[0]
        data.append([filefolder,
                     round(volume_mm3, 2), 
                     round(volume_L, 2), 
                     round(metric["dice"][1], 3), 
                     round(metric["hd"][1], 3), 
                     round(metric["hd95"][1], 3), 
                     round(metric["fpr"][1], 3)],
                     round(metric["fnr"][1], 3))
        print(metric["fpr"][0])

    return data

def show_table(pd_data):
    df = pd.DataFrame(pd_data, columns=["Filename", 
                                        "Volume [mm^3]", 
                                        "Volume [L]", 
                                        "DSC\u2193", 
                                        "HD\u2191",
                                        "HD95\u2191",
                                        "FPR\u2191",
                                        "FNR\u2191"]) # \u2191
    display(df)
    return df

if __name__=="__main__":
    from model.UNet3D import UNet3D
    from dataloader import scoliosis_dataset, TransformDataset
    from utils import load_config
    from postprocessing import Volume, calc_volume_dsc_hd, show_table

    config = load_config("config.json")     # Load config
    train_set_raw, val_set, test_set = scoliosis_dataset(config) # Base datasets
    train_set = TransformDataset(train_set_raw, config) # Augmentation in train dataset only!
    model = UNet3D(in_channels=1, num_classes=config["dataloader"]["N_classes"]).cuda()
    model.load_state_dict(torch.load(R"saved\20240227_172605 320x320x16 150e 0.0005 aug\weights.pth"))
    pd_data = calc_volume_dsc_hd(test_set, model)
    df = show_table(pd_data)    
    