# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

# %% Import libraries
import torch
from model.UNet3D import UNet3D
from dataloader import scoliosis_dataset, TransformDataset
from train import Trainer
from utils import export_plot, plot_3D_mesh
from postprocessing import calc_scores, show_table, Volume

train_set_raw, val_set, test_set = scoliosis_dataset() # Base datasets
train_set = TransformDataset(base_dataset=train_set_raw) # Augmentation in train dataset only!
model = UNet3D().cuda() # Build model
model.load_state_dict(torch.load(R"weights.pth")) # Load weights
# %% ============== Train ==============
# trainer = Trainer(model=model, 
#                   train_set=train_set, 
#                   val_set=val_set)
# trainer.train()

# %% ============== Quantitative results ==============
pd_data = calc_scores(test_set, model) # Create pandas data for volume and spine
show_table(pd_data) # Show pandas table

# %% ============== Qualitative results ==============
data = test_set[0]
export_plot(image=data[0], # Shows prediction slice by slice in /test_results/
            mask=data[1],
            prediction=model(data[0].unsqueeze(0).unsqueeze(0).cuda()))

# %% ============== Plot 3D Mesh ==============
data = test_set[0]
vobj = Volume(image=data[0], 
              mask=data[1], 
              prediction=model(data[0].unsqueeze(0).unsqueeze(0).cuda()), 
              header=data[2])
vobj.get_objective()
plot_3D_mesh(vobj) 
plot_3D_mesh(vobj, length=True) # No spine, only Center of Mass voxels

