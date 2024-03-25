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
from torch.optim.lr_scheduler import LambdaLR
from logger import Logger
from utils import load_config

class Trainer():
    def __init__(self, model, train_set, val_set, cfg=None, visualize_img=None):
        if not cfg:
            cfg = load_config("config.json")     
        config = cfg["trainer"]
        self.train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=config["batch_size"])
        self.val_loader = DataLoader(dataset=val_set, batch_size=1)
        self.device = torch.device(config["device"])
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"] # Total epochs
        self.Loss_fn = getattr(nn, config["loss_fn"]) # CrossEntropyLoss
        self.model = model.to(self.device)
        self.logger = Logger(config=cfg, save=True)
        self.decay_lr_after = config["decay_lr_after"]
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=config["lr"])
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=self.lr_lambda)
        if visualize_img: # This image is consistently used to visualize during training
            self.visualize_img = visualize_img
        else:
            self.visualize_img = train_set[0]

    def train(self):
        """Training of model"""
        training_start_time = time.time() # Start time
        self.logger.make_dir() # Create directories
        self.visualize(epoch="init") # Export initialize image/prediction
        best_val_loss=10 # Save best validation weights
        for epoch in range(0, self.epochs):
            self.model.train() # Train mode
            train_loss = self.run_epoch(epoch, self.train_loader) # Train
            self.model.eval() # Validation mode
            val_loss = self.run_epoch(epoch, self.val_loader) # Validate

            print(f"Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")

            self.logger.append_train_loss(train_loss.detach().cpu().item()) # Save losses
            self.logger.append_val_loss(val_loss.detach().cpu().item())
            self.logger.plot(epoch)
            self.visualize(epoch)
            # if val_loss<best_val_loss:
            #     best_val_loss=val_loss
            #     if self.logger.save:
            #         self.logger.save_weights(self.model, epoch)
            self.logger.save_weights(self.model, epoch)


        print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))
        self.logger.append_time_elapsed(time.time() - training_start_time) # Save training duration
        self.logger.save_loss() # Save loss to .json
           
    def run_epoch(self, epoch, loader):
        """Helper function for one single epoch training"""
        loss_epoch_list = list() # Average loss over one epoch
        loss_data_list = list() # Individual losses per image
        with tqdm(loader, unit="epoch") as tepoch:
            for img, mask, _, _ in tepoch:
                tepoch.set_description(f"Epoch: {epoch}/{self.epochs}")
                self.optimizer.zero_grad() # Zero gradients after every image    
                img = img.to(self.device)
                mask = mask.to(self.device)
                prediction = self.model(img.unsqueeze(0)) # Forward pass
                weights = torch.bincount(mask.type(torch.int).view(-1)).type(torch.float)
                weights = 1/weights
                self.loss_fn = self.Loss_fn(weight=weights)
                # self.loss_fn = self.Loss_fn()
                loss = self.loss_fn(prediction, mask.long())
                """Loss input: (batch,C,h,w,d) C->LOGITS and (batch,C,h,w,d) C->CLASS VALUES"""
                loss_data_list.append(loss.cpu().detach().item())                
                if self.model.training:
                    loss.backward() # Backward pass, calculate gradients       
                    self.optimizer.step() # Update weight

                loss_epoch_list.append(loss.cpu().detach())
        if self.model.training:
            self.logger.append_data_loss(loss_data_list)
        else:
            self.logger.append_valdata_loss(loss_data_list)

        self.scheduler.step()
        del img, mask
        return sum(loss_epoch_list) / len(loss_epoch_list) # Average loss one epoch
    
    def visualize(self, epoch):
        """Exports image during training for visualization.
        """        
        img = self.visualize_img[0]
        mask = self.visualize_img[1]
        logits = self.model(img.unsqueeze(0).unsqueeze(0).to(self.device))
        self.logger.export_train(epoch, img, mask, logits)

    def lr_lambda(self, the_epoch):
        """Function for scheduling learning rate"""
        return (
            1.0
            if the_epoch < self.decay_lr_after
            else 1 - float(the_epoch - self.decay_lr_after) / (self.epochs - self.decay_lr_after)
        )            
            
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
        
        