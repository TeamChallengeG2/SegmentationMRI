# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

# %% Import libraries

# from model.UNet import UNet
from model.UNet3D import UNet3D
from utils import load_config, plot_overlay, plot_slices, plot_test
from dataloader import Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from train import Trainer, Tester
from logger import Logger
import torch
from utils.transforms import RandomRotate3D

print(torch.__version__) 
print(torch.cuda.is_available())

if __name__ == "__main__":  # must be enabled for num_workers > 0
    config = load_config("config.json")
    dataset = Dataset(config)
    if config["dataloader"]["transformation"]=="withrotation":
        dataset.augment_all(RandomRotate3D((-10,10),axes=(0,1)))
    train_set, val_set, test_set = random_split(dataset=dataset,
                                                lengths=[0.7,0.2,0.1], 
                                                generator=torch.Generator().manual_seed(42))

    # plot_slices(dataset[0][0], dataset[0][1], 6)

    train_loader = DataLoader(dataset=train_set, 
                            batch_size=config["trainer"]["batch_size"],
                            shuffle=True
                            )
    
    val_loader = DataLoader(dataset=val_set, 
                            batch_size=1
                            )
    
    test_loader = DataLoader(dataset=test_set,
                            batch_size=1
                            )
    
    print(f"Train: {len(train_loader)}\nVal: {len(val_loader)}\nTest: {len(test_loader)}")

    model = UNet3D(in_channels=1, num_classes=2)
    myLogger = Logger(config=config, save=True)

    #%% ============== Train ==============
    trainer = Trainer(model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    logger=myLogger,
                    visualize_img=dataset[0][0])

    trainer.train()

    #%% Spacings
    # import nrrd
    # _, h = nrrd.read(dataset.data_paths[0][0])

    # #%% Test 1
    # model.cuda()
    # model.load_state_dict(torch.load(r'saved\20240227_172605 320x320x16 150e 0.0005 aug\weights.pth'))
    # img = val_set[1][0]
    # mask = val_set[1][1]
    # plot_overlay(img, mask)
    # prediction = model(img.unsqueeze(0).unsqueeze(0).cuda())
    # plot_test(img, mask, prediction)

    #%% Test 
    # if torch.cuda.is_available():
    #     model.cuda()
    # myLogger_test = Logger(config=config, save=True)
    # tester=Tester(model, val_loader,config=config, logger=myLogger_test)
    # tester.test()

# %%

# %%
