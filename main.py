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
from utils import load_config, plot_overlay, plot_slices, plot_test
from dataloader import Dataset
from torch.utils.data import DataLoader, random_split
from train import Trainer, Tester
from logger import Logger
from postprocessing import calc_volumes, Volume

print(torch.__version__) 
print(torch.cuda.is_available())

if __name__ == "__main__":  # must be enabled for num_workers > 0
    config = load_config("config.json")
    dataset = Dataset(config)

    train_set, val_set, test_set = random_split(dataset=dataset,
                                                lengths=[0.7,0.2,0.1], 
                                                generator=torch.Generator().manual_seed(42))

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
    model.cuda()
    model.load_state_dict(torch.load(R"saved\lr0.0005withoutrotation320_16\epoch24_weights.pth"))
    #%% ============== Train ==============
    trainer = Trainer(model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    logger=myLogger,
                    visualize_img=dataset[0][0])

    trainer.train()

    #%% ============== Test ==============
    if torch.cuda.is_available():
        model.cuda()
    myLogger_test = Logger(config=config, save=False)
    tester = Tester(model, val_loader, config=config, logger=myLogger_test)
    tester.test()
    #%% ============== Volume ==============
    pd_data = calc_volumes(test_set, model)

    #%% ============== Visualization ==============
    pred = model(test_set[4][0].unsqueeze(0).unsqueeze(0).cuda())
    plot_test(image=test_set[4][0],
              mask=test_set[4][1],
              prediction=pred,
              mask_only=False)
# %%
