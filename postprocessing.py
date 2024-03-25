import torch
import numpy as np
import pandas as pd
import seg_metrics.seg_metrics as sg
from tqdm import tqdm
from scipy.ndimage import zoom, median_filter, center_of_mass


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

    def get_objective(self):
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
                        metrics=['hd', 'hd95','dice','fpr','fnr', 'precision', 'recall'])
        
        self.mask_volume, self.mask_spine = self.upsample(self.image, self.mask, self.prediction_mask)
        nr_voxels = np.count_nonzero(self.mask_volume == 1) # Count volume voxels
        [L_dim, P_dim, S_dim] = self.get_new_spacings(self.header, self.mask_volume.shape) 
        volume_voxel = L_dim*P_dim*S_dim # Voxel volume
        volume_L = (nr_voxels * volume_voxel)/1e6 # Total volume
        spinal_length_cm = self.get_length()
        return volume_L, spinal_length_cm
    
    def get_length(self):
        """Calculates the length of spine based on center of masses.

        Returns:
            float: length in cm
        """        
        self.coords_com = list()
        for i in range(self.mask_spine.shape[2]): # Add indices of center of mass to list
            com = list(center_of_mass(self.mask_spine.numpy()[:,:,i]))
            com.append(i)
            self.coords_com.append(com)
        
        self.coords_com = np.array(self.coords_com)

        [L_dim, P_dim, S_dim] = self.get_new_spacings(self.header, self.mask_volume.shape) 
        distance = 0

        for i in range(len(self.coords_com)-1):
            # Calculate distance between COM's between 2 slices
            if (not np.isnan(self.coords_com[i][0])) and (not np.isnan(self.coords_com[i+1][0])):
                # Calculate physical distance based on voxel-distance
                physical_distance = [L_dim, P_dim, S_dim] * (self.coords_com[i+1]-self.coords_com[i])
                distance += np.linalg.norm(physical_distance)
        
        return distance/10

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
    data_volume = list()
    data_spinal_l = list()
    for img, mask, header, filefolder in tqdm(dataset):
        pred = model(img.unsqueeze(0).unsqueeze(0).cuda())
        volume_object = Volume(img, mask, pred, header)
        volume_L, spinal_length_cm = volume_object.get_objective()
        metric = volume_object.metrics[0]
        data_volume.append([filefolder,
                     round(volume_L, 2), 
                     round(metric["dice"][1], 3), 
                     round(metric["hd"][1], 3), 
                     round(metric["hd95"][1], 3),
                     round(metric["precision"][1], 3),
                     round(metric["recall"][1], 3)])
        
        data_spinal_l.append([filefolder,
                     round(spinal_length_cm, 1), 
                     round(metric["dice"][2], 3), 
                     round(metric["hd"][2], 3), 
                     round(metric["hd95"][2], 3),   
                     round(metric["precision"][2], 3),
                     round(metric["recall"][2], 3)])
        
    return data_volume, data_spinal_l

def show_table(pd_data):
    df = pd.DataFrame(pd_data[0], columns=["Filename",
                                           "Volume [L]",
                                           "DSC\u2191", 
                                           "HD\u2193",
                                           "HD95\u2193",
                                           "Precision",
                                           "Recall"])
    
    df2 = pd.DataFrame(pd_data[1], columns=["Filename",
                                            "Length [cm]",
                                            "DSC\u2191",
                                            "HD\u2193",
                                            "HD95\u2193",
                                            "Precision",
                                            "Recall"])
    
    display(df)
    display(df2)

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
    