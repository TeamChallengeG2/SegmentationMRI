# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

import time
import os
import matplotlib.pyplot as plt
import scienceplots
import torch
from utils import write_config

class Logger():
    def __init__(self, config, save):
        self.config = config
        self.loss_train_list = list()
        self.loss_data_list = list()
        self.loss_val_list = list()
        self.loss_fn = config["trainer"]["loss_fn"]
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.path = f"saved/{timestr}/"
        self.save = save

    def append_train_loss(self, input):
        self.loss_train_list.append(input)
        
    def append_val_loss(self, input):
        self.loss_val_list.append(input)
        
    def append_data_loss(self, input):
        self.loss_data_list.append(input)
        
    def plot(self, epoch):
        plt.style.use(['science', 'ieee', 'no-latex'])
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.00
        fig, axes = plt.subplots(1, 1)
        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
        axes.plot(range(1, len(self.loss_train_list) + 1), self.loss_train_list,
                  "orange", label="Train loss")
        for xe, ye in zip(range(1, len(self.loss_data_list) + 1), self.loss_data_list):
                axes.scatter([xe] * len(ye), ye, c="orange", s=1.0)
        # axes.plot(range(1, len(self.loss_data_list) + 1), ,
        #           '.', markerfacecolor=(0, 0, 1, 0.3), label="Batch loss")
        # axes.plot(range(1, len(self.loss_val_list) + 1), self.loss_val_list, 'r', label="Validation loss")
        plt.xticks(range(1, len(self.loss_train_list) + 1))
        # axes.title.set_text(f"{self.loss_fn}")
        plt.ylabel(f"{self.loss_fn}", fontweight='bold')
        plt.xlabel("Epochs", fontweight='bold')
        # plt.grid()
        plt.legend()
        if self.save:
            plot_path = self.path + "loss_plot"
            isExist = os.path.exists(plot_path)
            if not isExist:
                os.makedirs(plot_path)
            plt.savefig(f"{plot_path}/epoch_{epoch}.png")
        plt.show()
        
    def save_weights(self, model):
        isExist = os.path.exists(self.path)
        if not isExist:
            os.makedirs(self.path)
        torch.save(model.state_dict(), f"{self.path}/weights.pth")
        write_config(self.config, f"{self.path}/config.json")

    def save_loss(self):
        d = {"loss_train_list": self.loss_train_list,
             "loss_data_list": self.loss_data_list,
             "loss_val_list": self.loss_val_list
             }
        write_config(d, f"{self.path}/logs.json")