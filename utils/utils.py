# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

# %%
from collections import OrderedDict
from pathlib import Path
from skimage import color, segmentation
import torch
import json
import colorsys
import numpy as np
import matplotlib.pyplot as plt

# %%

def load_config(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_config(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def plot_overlay(image, mask, slice=10):
        """
        Plots segmentation mask over an image.

        Parameters
        ----------
        image : numpy.ndarray or torch.Tensor
            image
        mask : numpy.ndarray or torch.Tensor
            mask
        slice : int

        Returns
        -------
        None.

        """
        overlay = np.ma.masked_where(mask == 0, mask)
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.rot90(image[:,:,slice], 3), cmap="gray")
        axs[1].imshow(np.rot90(mask[:,:,slice], 3), cmap="gray")
        axs[2].imshow(np.rot90(image[:,:,slice], 3), cmap="gray")
        axs[2].imshow(np.rot90(overlay[:,:,slice], 3), cmap="prism", alpha=0.4)
        titles = ["Image", "Mask", "Overlay"]

        for ax, title in zip(axs, titles):
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


