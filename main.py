# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

# %% Import libraries

from model.UNet import UNet
from utils import load_config, plot_overlay
from dataloader import Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from train import Trainer, Tester
from logger import Logger
import torch 

print(torch.__version__) 
print(torch.cuda.is_available())

# % Main
if __name__ == "__main__":  # must be enabled for num_workers > 0
    config = load_config("config.json")
    dataset = Dataset(config)
    dataset.augment_all(v2.RandomRotation(30)) 
    # dataset.augment_all(v2.RandomAffine(degrees=0, translate=(0.1, 0.1))) 
    train_set, val_set, test_set = random_split(dataset=dataset,
                                                lengths=[0.7,0.1,0.2], 
                                                generator=torch.Generator().manual_seed(42))
    
    plot_overlay(dataset[0][0], dataset[0][1])
    # train_loader = DataLoader(dataset=train_set, 
    #                         batch_size=config["trainer"]["batch_size"],
    #                         collate_fn=lambda batch : batch,
    #                         )
    
    # val_loader = DataLoader(dataset=val_set, 
    #                         batch_size=1,
    #                         collate_fn=dataset.collate_fn,
    #                         )
    
    # test_loader = DataLoader(dataset=test_set, 
    #                         batch_size=1,
    #                         collate_fn=dataset.collate_fn,
    #                         )
    
    # model = UNet()
    # myLogger = Logger(config=config, save=True)
    # trainer = Trainer(model=model, 
    #                 train_loader=train_loader, 
    #                 val_loader=val_loader, 
    #                 config=config, 
    #                 logger=myLogger)

    # trainer.train()
    
#%% 
# model = UNet()
# tester = Tester(model, test_loader)
# %%
