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
from utils import load_config, export_plot
from postprocessing import calc_volume_dsc_hd, show_table

config = load_config("config.json")     # Load config
train_set_raw, val_set, test_set = scoliosis_dataset(config) # Base datasets
train_set = TransformDataset(train_set_raw, config) # Augmentation in train dataset only!

model = UNet3D(in_channels=1, num_classes=config["dataloader"]["N_classes"]).cuda() # Build model

#%% ============== Train ==============
# trainer = Trainer(model, train_set, val_set, config)
# trainer.train()

#%% ============== Volume ==============
model.load_state_dict(torch.load(R"saved\20240227_172605 320x320x16 150e 0.0005 aug\weights.pth"))
pd_data = calc_volume_dsc_hd(test_set, model) # Create pandas data 
show_table(pd_data) # Show pandas table

#%% ============== Visualization ==============
data=test_set[0]
pred = model(data[0].unsqueeze(0).unsqueeze(0).cuda())
export_plot(image=data[0],
            mask=data[1],
            prediction=pred,
            mask_only=False)
