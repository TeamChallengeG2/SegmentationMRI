# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

import os
import torch.nn as nn
import torch
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from tqdm import tqdm


class Trainer():
    def __init__(self, model, train_loader, val_loader, config, logger):
        self.config = config["trainer"]
        self.device = torch.device(self.config["device"])
        self.batch_size = self.config["batch_size"]
        self.epochs = self.config["epochs"]
        self.loss_fn = getattr(nn, self.config["loss_fn"])() 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(self.device)
        self.logger = logger
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.config["lr"]) 

    def train(self):
        """Training of model"""
        for epoch in range(1, self.config["epochs"] + 1):
            self.model.train() 
            loss_train_epoch = self._run_epoch(epoch, self.train_loader)
            self.model.eval()
            loss_val_epoch = self._run_epoch(epoch, self.val_loader)              
            print(f"Train loss: {loss_train_epoch:.4f} | Validation loss: {loss_val_epoch:.4f}")
            self.logger.append_train_loss(loss_train_epoch.detach().cpu())
            self.logger.append_val_loss(loss_val_epoch.detach().cpu())
            self.logger.plot(epoch)
            img, _ = next(iter(self.train_loader))
            self.visualize(self.model, img, epoch)
            if self.logger.save:
                self.logger.save_weights(self.model)
           

    def _run_epoch(self, epoch, loader):
        """Helper function for one single batch training"""
        loss_epoch_list = list()
        loss_data_list = list()
        with tqdm(loader, unit="batch") as tepoch:
            for img, mask in tepoch:
                tepoch.set_description(f"Epoch: {epoch}/{self.epochs}")
                self.optimizer.zero_grad() # set gradients to 0           
                loss_batch_list = list()             
                img = img.to(self.device)
                mask = mask.to(self.device)
                prediction = self.model(img.unsqueeze(0)) # forward pass
                loss = self.loss_fn(prediction, mask.long())
                """Loss input: (batch,C,h,w) and (batch,h,w):target with class VALUES"""
                if self.model.training:
                    loss_data_list.append(loss.cpu().detach())
                    loss.backward() # backward pass based on training of 1 batch        
                    self.optimizer.step() # update weight
                loss_epoch_list.append(loss.cpu().detach())
        # if self.model.training:
            # self.logger.append_data_loss(torch.stack(loss_data_list).cpu().detach().numpy())
            # self.visualize(self.model, img, epoch)
        del img, mask
        return sum(loss_epoch_list) / len(loss_epoch_list) # average loss 1 epoch
    
    def visualize(self, model, input, epoch):
        output = model(input.unsqueeze(0).to(self.device))
        self.logger.save_fig(output, epoch)

class Tester():
    def __init__(self, model, test_loader):
        filename_pth = self.get_path()
        directory = os.path.dirname(filename_pth)
        print(directory)
    
    def test(self):
        pass
    
    def test_batch(self):
        pass
    
    def plot_score(self):
        pass
    
    def plot_data(self):
        pass
    
    def get_path(self):
        window = tk.Tk()
        window.wm_attributes('-topmost', 1)
        window.withdraw()   # this supress the tk window
        filename_pth = filedialog.askopenfilename(parent=window,
                                          initialdir="saved",
                                          title="Select a file (.pth)",
                                          filetypes = (("Model weights", "*.pth"), ("All files", "*")))
        return filename_pth
    
                 
            
        
        