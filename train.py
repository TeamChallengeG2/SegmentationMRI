# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from logger import Logger
from utils import export_plot

class Trainer():
    def __init__(self, model, train_set, val_set, config, visualize_img=None):
        cfg = config["trainer"]
        self.train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=cfg["batch_size"])
        self.val_loader = DataLoader(dataset=val_set, batch_size=1)
        self.device = torch.device(cfg["device"])
        self.batch_size = cfg["batch_size"]
        self.epochs = cfg["epochs"]
        self.loss_fn = getattr(nn, cfg["loss_fn"])() 
        self.model = model.to(self.device)
        self.logger = Logger(config=config, save=True)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=cfg["lr"])
        self.schedular=torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                            milestones=[5,10,15],
                                                            gamma=0.5)
        if visualize_img:
            self.visualize_img = visualize_img
        else:
            self.visualize_img = train_set[0]

    def train(self):
        """Training of model"""
        training_start_time = time.time()
        self.logger.make_dir()
        self.visualize(epoch="init")
        best_val_loss=1
        for epoch in range(0, self.epochs):
            self.model.train() 
            train_loss = self.run_epoch(epoch, self.train_loader)
            self.model.eval()
            val_loss = self.run_epoch(epoch, self.val_loader)

            print(f"Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")
            self.logger.append_train_loss(train_loss.detach().cpu().item())
            self.logger.append_val_loss(val_loss.detach().cpu().item())
            self.logger.plot(epoch)
            self.visualize(epoch)
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                if self.logger.save:
                    self.logger.save_weights(self.model, epoch)
        print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))
        self.logger.append_time_elapsed(time.time() - training_start_time)
        self.logger.save_loss()
           
    def run_epoch(self, epoch, loader):
        """Helper function for one single epoch training"""
        loss_epoch_list = list()
        loss_data_list = list()
        with tqdm(loader, unit="epoch") as tepoch:
            for img, mask, _, _ in tepoch:
                tepoch.set_description(f"Epoch: {epoch}/{self.epochs}")
                self.optimizer.zero_grad() # set gradients to 0           
                img = img.to(self.device)
                mask = mask.to(self.device)
                prediction = self.model(img.unsqueeze(0)) # forward pass
                loss = self.loss_fn(prediction, mask.long())
                """Loss input: (batch,C,h,w) and (batch,h,w):target with class VALUES"""
                loss_data_list.append(loss.cpu().detach().item())                
                if self.model.training:
                    loss.backward() # backward pass based on training of 1 batch        
                    self.optimizer.step() # update weight

                loss_epoch_list.append(loss.cpu().detach())
        if self.model.training:
            self.logger.append_data_loss(loss_data_list)
        else:
            self.logger.append_valdata_loss(loss_data_list)

        del img, mask
        return sum(loss_epoch_list) / len(loss_epoch_list) # average loss 1 epoch
    
    def visualize(self, epoch):
        """Exports image during training for visualization.
        """        
        img = self.visualize_img[0]
        mask = self.visualize_img[1]
        output = self.model(img.unsqueeze(0).unsqueeze(0).to(self.device))
        self.logger.export_train(epoch, img, mask, output)
            
if __name__=="__main__":
    from model.UNet3D import UNet3D
    from utils import load_config
    from dataloader import scoliosis_dataset, TransformDataset

    config = load_config("config.json")     # Load config
    train_set_raw, val_set, test_set = scoliosis_dataset(config) # Base datasets
    train_set = TransformDataset(train_set_raw, config) # Augmentation in train dataset only!
    model = UNet3D(in_channels=1, num_classes=config["dataloader"]["N_classes"]).cuda()
    trainer = Trainer(model, train_set, val_set, config)
    trainer.train()    
        
        