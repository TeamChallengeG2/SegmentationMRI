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
import pandas as pd
from utils import write_config, get_overlay
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker


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
                axes.scatter([xe] * len(ye), ye, c="blue", s=0.1, alpha=0.3)
        axes.plot(range(0, len(self.dict["val_loss"])), self.dict["val_loss"], 'r-', label="Validation loss")
        for xe, ye in zip(range(0, len(self.dict["val_individual"])), self.dict["val_individual"]):
                axes.scatter([xe] * len(ye), ye, c="red", s=0.1, alpha=0.3)        
        axes.set_ylim([0, 0.6])
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
        
    def export_train(self, epoch, img, mask, output):
        prob = torch.softmax(output, dim=1)
        heatmap_vol = prob[:, 1, :, :, :].squeeze().detach().cpu() 
        heatmap_spine= prob[:, 2, :, :, :].squeeze().detach().cpu() 
        segm_mask = np.argmax(prob.detach().cpu().squeeze(), axis=0)

        fig, axs = plt.subplots(2, 2, figsize=(9, 9))
        axs[0][0].imshow(np.rot90(img[:,:,10], 3), cmap="gray") # img + mask
        axs[0][0].imshow(np.rot90(get_overlay(mask[:,:,10]), 3), cmap="gnuplot2", alpha=0.4, vmin=0, vmax=2.5)   
        
        cbar1 = axs[0][1].imshow(np.rot90(heatmap_vol[:,:,10], 3), cmap="jet") # heatmap volume
        axins = inset_axes(axs[0][1],
                    width="5%",  
                    height="100%",
                    loc="center right",
                    borderpad=-2.0
                    )
        cb1 = plt.colorbar(cbar1, axins, orientation="vertical", format=ticker.FormatStrFormatter("%.2f"))
        cb1.outline.set_linewidth(0.5)
        cb1.ax.locator_params(nbins=3)
        
        axs[1][0].imshow(np.rot90(segm_mask[:,:,10], 3), cmap="gnuplot2", vmin=0, vmax=2.5) # prediction

        cbar2 = axs[1][1].imshow(np.rot90(heatmap_spine[:,:,10], 3), cmap="jet") # heatmap spine
        axins = inset_axes(axs[1][1],
                    width="5%",  
                    height="100%",
                    loc="center right",
                    borderpad=-2.0
                    )
        cb2 = plt.colorbar(cbar2, axins, orientation="vertical", format=ticker.FormatStrFormatter("%.2f"))
        cb2.outline.set_linewidth(0.5)

        cb2.ax.locator_params(nbins=3)

        for ax in axs.ravel():
            ax.set_axis_off()
        fig.subplots_adjust(wspace=0, hspace=0)    
        plt.savefig(self.path + f"/{epoch}.png")
    

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
