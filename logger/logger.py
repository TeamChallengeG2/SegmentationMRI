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
        self.loss_train_list = list()
        self.loss_data_list = list()
        self.loss_val_list = list()
        self.loss_valdata_list = list()
        self.loss_fn = config["trainer"]["loss_fn"]
        self.save = save
        self.lr=config["trainer"]["lr"]
        self.transform=config["dataloader"]["transformation"]
        self.LP=config["dataloader"]["LP_dimension"]
        self.S=config["dataloader"]["S_dimension"]

    def make_dir(self):
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        # self.path = f"saved/{timestr}/"
        self.path=f"saved/lr{self.lr}{self.transform}{self.LP}_{self.S}/"
        isExist = os.path.exists(self.path)
        if not isExist:
            os.makedirs(self.path)

    def append_train_loss(self, input):
        self.loss_train_list.append(input)
        
    def append_val_loss(self, input):
        self.loss_val_list.append(input)
        
    def append_data_loss(self, input):
        self.loss_data_list.append(input)

    def append_valdata_loss(self, input):
        self.loss_valdata_list.append(input)        
        
    def plot(self, epoch):
        matplotlib.use('Agg')
        plt.style.use(['science', 'ieee', 'no-latex'])
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.00
        fig, axes = plt.subplots(1, 1)
        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
        axes.plot(range(0, len(self.loss_train_list)), self.loss_train_list,
                  "blue", label="Train loss")
        for xe, ye in zip(range(0, len(self.loss_data_list)), self.loss_data_list):
                axes.scatter([xe] * len(ye), ye, c="blue", s=0.1, alpha=0.5)
        # axes.plot(range(1, len(self.loss_data_list) + 1),
                #   '.', markerfacecolor=(0, 0, 1, 0.3), label="Batch loss")
        axes.plot(range(0, len(self.loss_val_list)), self.loss_val_list, 'r-', label="Validation loss")
        for xe, ye in zip(range(0, len(self.loss_valdata_list)), self.loss_valdata_list):
                axes.scatter([xe] * len(ye), ye, c="red", s=0.1, alpha=0.5)        
        axes.set_ylim([0, 1])
        plt.xticks(range(0, len(self.loss_train_list), 20))
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
        plt.close()    
        
    def save_weights(self, model,epoch):
        torch.save(model.state_dict(), f"{self.path}/epoch{epoch}_weights.pth")
        write_config(self.config, f"{self.path}/config.json")

    def save_loss(self):
        d = {"loss_train_list": self.loss_train_list,
             "loss_data_list": self.loss_data_list,
             "loss_val_list": self.loss_val_list
             }
        write_config(d, f"{self.path}/logs.json")

    def save_fig(self, prediction, epoch):
        # prob = torch.sigmoid(prediction)
        prob = torch.softmax(prediction, dim=1)
        heatmap = prob[:,1,:,:,10].squeeze().detach().cpu()
        matplotlib.use('Agg')
        plt.figure()        
        plt.imshow(np.rot90(heatmap, 3), cmap='hot', interpolation='nearest')
        plt.title(f"Epoch: {epoch}")
        plt.colorbar()
        plt.savefig(self.path + f"{epoch}.png")
        plt.close()
    def save_fig_slice(self, image):
        for i in range(len(image[0][2])-1):
            plt.figure()
            plt.imshow(np.rot90(image[:,:,i], 3), cmap='grey')
            plt.title(f"slice: {i+1}")
            plt.savefig('saved/resampled' + f"slice{i+1}.png")
            plt.close()

    def save_mask_fig(self, mask, prediction, index):
        # prob = torch.sigmoid(prediction)
        mask = mask.squeeze().cpu()
        mask=mask[:,:,8]
        print(mask.shape)
        prob = torch.softmax(prediction, dim=1)
        heatmap = prob[:, 1, :, :, 8].squeeze().detach().cpu()
        matplotlib.use('Agg')
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.rot90(mask, 3), cmap='grey')
        plt.title(f"{index}_mask")
        plt.subplot(1,2,2)
        plt.imshow(np.rot90(heatmap, 3), cmap='hot', interpolation='nearest')
        plt.title(f"{index}_prediction")
        plt.colorbar()
        plt.savefig(self.path + f"patient{index}.png")
        plt.close()

    def dice_acc_csv(self,dice,acc,index):
        cvs_file_path=self.path+"metrics.csv"
        if not os.path.isfile(cvs_file_path):
            df=pd.DataFrame(columns=['index', 'dice', 'acc'])
        else:
            df=pd.read_csv(cvs_file_path)
        new_data=pd.DataFrame([[index, dice, acc]],columns=['index', 'dice', 'acc'])
        df=pd.concat([df,new_data],ignore_index=True)
        df.to_csv(cvs_file_path,index=False)

