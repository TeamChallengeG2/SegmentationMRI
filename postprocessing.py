import torch
import numpy as np
import pandas as pd
import seg_metrics.seg_metrics as sg
from tqdm import tqdm
from scipy.ndimage import zoom, median_filter

class Volume():
    def __init__(self, image, mask, prediction, header):
        self.header = header
        self.image = image
        self.mask = mask
        self.prediction = prediction
        self.slice_thickness = 4 # mm

    def prediction_to_mask(self):
        """
        Converts raw logits from prediction to mask by index with highest value.
        """        
        prob = torch.softmax(self.prediction, dim=1)
        return np.argmax(prob.detach().cpu().squeeze(), axis=0)

    def get_volume(self):
        """Returns volume in mm^3 for an input image.

        Returns:
            float: volume in mm^3
        """        
        self.prediction_mask = self.prediction_to_mask() # Prediction mask
        labels = np.unique(self.mask).tolist() # Class label values in mask [0, 1, 2]
        spacings = self.get_new_spacings(self.header, self.prediction_mask.shape) # Prediction spacings

        self.metrics = sg.write_metrics(labels=labels, # Computes metrics
                        gdth_img=self.mask.numpy(),
                        pred_img=self.prediction_mask.numpy(),
                        spacing=spacings,
                        metrics=['hd', 'hd95','dice','recall','fpr','fnr'])
        
        self.mask_volume, self.mask_spine = self.upsample(self.image, self.mask, self.prediction_mask)
        nr_voxels = np.count_nonzero(self.mask_volume == 1) # Count volume voxels
        [L_dim, P_dim, S_dim] = self.get_new_spacings(self.header, self.mask_volume.shape) 
        volume_voxel = L_dim*P_dim*S_dim # Voxel volume
        return nr_voxels * volume_voxel # Total volume

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
    
    def upsample(self, img, mask, prediction_mask):
        """Upsamples input to dimensions without slice gaps.

        Args:
            img (numpy.ndarray): input image
            mask (numpy.ndarray): input mask
            prediction (numpy.ndarray): prediction

        Returns:
            numpy.ndarray: resampled img, mask and prediction
        """        
        self.S_dimension = self.get_new_S_dimension(self.header)        
        # Split up volume and spine, and then upsample.
        mask_volume = zoom(input=np.where(prediction_mask == 1, 1, 0), 
                           zoom=(1, 1, self.S_dimension/prediction_mask.shape[-1]))
        mask_spine = zoom(input=np.where(prediction_mask == 2, 1, 0), 
                          zoom=(1, 1, self.S_dimension/prediction_mask.shape[-1]))
        
        mask_volume = median_filter(mask_volume, 5)
        mask_spine = median_filter(mask_spine, 5)

        return torch.from_numpy(mask_volume).float(), torch.from_numpy(mask_spine).float()
    
    def get_new_S_dimension(self, header):
            """Method to determine new dimension size in inferior-superior axis for upsampling.

            Args:
                header (OrderedDict): Dictionary header from .nrrd file

            Returns:
                int: dimension in S direction
            """        
            return self.get_original_spacings(header)[2]*header['sizes'][2] / self.slice_thickness    
    
def calc_scores(dataset, model):
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
        volume_L = volume_mm3 / 1e6
        metric = volume_object.metrics[0]
        data.append([filefolder,
                     round(volume_mm3, 0), 
                     round(volume_L, 2), 
                     round(metric["dice"][1], 3), 
                     round(metric["hd"][1], 3), 
                     round(metric["hd95"][1], 3), 
                     round(metric["recall"][1], 3),
                     round(metric["fpr"][1], 3),
                     round(metric["fnr"][1], 3)])

    return data

def show_table(pd_data):
    df = pd.DataFrame(pd_data, columns=["Filename", 
                                        "Volume [mm^3]", 
                                        "Volume [L]", 
                                        "DSC\u2191", 
                                        "HD\u2193",
                                        "HD95\u2193",
                                        "Recall",
                                        "FPR",
                                        "FNR"]) # \u2191
    display(df)
    return df

if __name__=="__main__":
    from model.UNet3D import UNet3D
    from dataloader import scoliosis_dataset, TransformDataset
    from utils import load_config, plot_3D_mesh
    from postprocessing import Volume, calc_scores, show_table

    config = load_config("config.json")     # Load config
    train_set_raw, val_set, test_set = scoliosis_dataset(config) # Base datasets
    train_set = TransformDataset(train_set_raw, config) # Augmentation in train dataset only!
    model = UNet3D(in_channels=1, num_classes=config["dataloader"]["N_classes"]).cuda()
    model.load_state_dict(torch.load(R"weights.pth"))
    pd_data = calc_scores(test_set, model)
    df = show_table(pd_data)    

    data = train_set[0]
    pred = model(data[0].unsqueeze(0).unsqueeze(0).cuda())
    vobj = Volume(image=data[0], 
                mask=data[1], 
                prediction=pred, 
                header=data[2])
    vobj.get_volume()
    plot_3D_mesh(vobj)
    