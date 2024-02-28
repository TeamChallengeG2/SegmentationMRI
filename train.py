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
import time
from tkinter import filedialog
from tqdm import tqdm
import csv


class Trainer():
    def __init__(self, model, train_loader, val_loader, config, logger, visualize_img):
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
        self.schedular=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[5,10,15],gamma=0.5)
        self.visualize_img = visualize_img.unsqueeze(0)

    def train(self):
        """Training of model"""
        training_start_time = time.time()
        self.logger.make_dir()
        bestval_loss=1
        for epoch in range(0, self.config["epochs"]):
            self.model.train() 
            loss_train_epoch = self._run_epoch(epoch, self.train_loader)
            # self.schedular.step()
            print(self.schedular.get_lr())
            self.model.eval()
            loss_val_epoch = self._run_epoch(epoch, self.val_loader)

            print(f"Train loss: {loss_train_epoch:.4f} | Validation loss: {loss_val_epoch:.4f}")
            self.logger.append_train_loss(loss_train_epoch.detach().cpu().item())
            self.logger.append_val_loss(loss_val_epoch.detach().cpu().item())
            self.logger.plot(epoch)
            self.visualize(self.model, self.visualize_img, epoch)
            if loss_val_epoch<bestval_loss:
                bestval_loss=loss_val_epoch
                if self.logger.save:
                    self.logger.save_weights(self.model,epoch)
        print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))
        self.logger.save_loss()
           

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
                    loss_data_list.append(loss.cpu().detach().item())
                    loss.backward() # backward pass based on training of 1 batch        
                    self.optimizer.step() # update weight

                loss_epoch_list.append(loss.cpu().detach())
        if self.model.training:
            self.logger.append_data_loss(loss_data_list)
        del img, mask
        return sum(loss_epoch_list) / len(loss_epoch_list) # average loss 1 epoch

    # def _run_epoch_val(self, epoch, loader):
    #     """Helper function for one single batch training"""
    #     loss_epoch_list = list()
    #     loss_data_list = list()
    #     with torch.no_grad():
    #         with tqdm(loader, unit="batch") as tepoch:
    #
    #             for img, mask in tepoch:
    #                 tepoch.set_description(f"Epoch: {epoch}/{self.epochs}")
    #                 img = img.to(self.device)
    #                 mask = mask.to(self.device)
    #                 prediction = self.model(img.unsqueeze(0))  # forward pass
    #                 loss = self.loss_fn(prediction, mask.long())
    #                 """Loss input: (batch,C,h,w) and (batch,h,w):target with class VALUES"""
    #             loss_epoch_list.append(loss.cpu().detach())
    #     del img, mask
    #     return sum(loss_epoch_list) / len(loss_epoch_list)  # average loss 1 epoch
    
    def visualize(self, model, input, epoch):
        output = model(input.unsqueeze(0).to(self.device))
        self.logger.save_fig(output, epoch)

class Tester():
    def __init__(self, model, test_loader, config, logger):
        # filename_pth = self.get_path()
        # directory = os.path.dirname(filename_pth)
        # print(directory)
        self.model=model
        self.loader=test_loader
        self.config = config["tester"]
        self.device = torch.device(self.config["device"])
        self.logger = logger
    def test(self):
        self.model.eval()
        self.logger.make_dir()
        self.test_batch(self.loader)


    def test_batch(self,loader):
        with tqdm(loader, unit="batch") as tepoch:
            print(tepoch)
            index = 1
            for img, mask in tepoch:
                img = img.to(self.device)
                mask = mask.to(self.device)
                prediction = self.model(img.unsqueeze(0))
                prob = torch.softmax(prediction, dim=1)
                pred_mask = (prob[:, 1, :, :, :] >= 0.5).squeeze().detach().cpu().numpy()
                self.plot_data(img, prediction, mask, index)# f
                self.plot_score(index,pred_mask, mask)
                index += 1
    
    def plot_score(self,index,prediction,mask):
        dice=str(self.cal_dice(prediction,mask))
        acc=str(self.cal_acc(prediction, mask))
        self.logger.dice_acc_csv(dice,acc,index)


    def cal_dice(self, pred_mask,mask):
        mask=mask.detach().cpu().numpy()
        smooth=1e-5
        intersection=(pred_mask*mask).sum()
        return (2.*intersection+smooth)/(pred_mask.sum()+mask.sum()+smooth)
    def cal_acc(self,pred_mask,mask):
        mask = mask.detach().cpu().numpy()
        TP=(pred_mask*mask).sum()
        return TP/mask.sum()


    def plot_data(self,img, prediction, mask, index):
        self.logger.save_mask_fig(mask,prediction,index)
    
    def get_path(self):
        window = tk.Tk()
        window.wm_attributes('-topmost', 1)
        window.withdraw()   # this supress the tk window
        filename_pth = filedialog.askopenfilename(parent=window,
                                          initialdir="saved",
                                          title="Select a file (.pth)",
                                          filetypes = (("Model weights", "*.pth"), ("All files", "*")))
        return filename_pth
    
                 
            
        
        