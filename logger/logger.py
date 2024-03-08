# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

"""

import time
import os
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import torch
import numpy as np
from utils import write_config
import pandas as pd


class Logger():
    def __init__(self, config, save):
        self.config = config
        labels = ["train_loss", "val_loss",
                                   "train_individual", "val_individual", "time_elapsed"]
        self.dict = {key:[] for key in labels}
        self.loss_fn = config["trainer"]["loss_fn"]
        self.save = save
        self.lr=config["trainer"]["lr"]
        self.transform=config["dataloader"]["rotation_angle"]
        self.LP=config["dataloader"]["LP_dimension"]
        self.S=config["dataloader"]["S_dimension"]

    def make_dir(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.path = f"saved/{timestr}/"
        # self.path=f"saved/LR{self.lr}_rot{self.transform}_{self.LP}^2x{self.S}/"
        isExist = os.path.exists(self.path)
        if not isExist:
            os.makedirs(self.path)

    def append_train_loss(self, input):
        self.dict["train_loss"].append(input)
        
    def append_val_loss(self, input):
        self.dict["val_loss"].append(input)
        
    def append_data_loss(self, input):
        self.dict["train_individual"].append(input)

    def append_valdata_loss(self, input):
        self.dict["val_individual"].append(input)   

    def append_time_elapsed(self, input):
        self.dict["time_elapsed"].append(input)            
        
    def plot(self, epoch):
        """Plot graph for losses.

        Arguments:
            epoch (int): number epoch
        """        
        matplotlib.use('Agg')
        plt.style.use(['science', 'ieee', 'no-latex'])
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.00
        fig, axes = plt.subplots(1, 1)

        axes.plot(range(0, len(self.dict["train_loss"])), self.dict["train_loss"],
                  "blue", label="Train loss")
        for xe, ye in zip(range(0, len(self.dict["train_individual"])), self.dict["train_individual"]):
                axes.scatter([xe] * len(ye), ye, c="blue", s=0.1, alpha=0.5)
        axes.plot(range(0, len(self.dict["val_loss"])), self.dict["val_loss"], 'r-', label="Validation loss")
        for xe, ye in zip(range(0, len(self.dict["val_individual"])), self.dict["val_individual"]):
                axes.scatter([xe] * len(ye), ye, c="red", s=0.1, alpha=0.5)        
        axes.set_ylim([0, 1])
        plt.xticks(range(0, len(self.dict["train_loss"]), 20))

        plt.ylabel(f"{self.loss_fn}", fontweight='bold')
        plt.xlabel("Epochs", fontweight='bold')
        plt.legend()
        if self.save:
            plot_path = self.path + "loss_plot"
            isExist = os.path.exists(plot_path)
            if not isExist:
                os.makedirs(plot_path)
            plt.savefig(f"{plot_path}/epoch_{epoch}.png")
        plt.close()    
        
    def save_weights(self, model, epoch):
        """Saves model and weights to file.

        Arguments:
            model: PyTorch model
            epoch (int): epoch
        """        
        torch.save(model.state_dict(), f"{self.path}/epoch{epoch}_weights.pth")
        write_config(self.config, f"{self.path}/config.json")

    def save_loss(self):
        """Saves loss dictionary to .json file
        """        
        write_config(self.dict, f"{self.path}/logs.json")
