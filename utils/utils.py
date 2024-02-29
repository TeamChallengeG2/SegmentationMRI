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
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def plot_slices(image, mask, nr_slices=6):
    slices = list()
    for i in range(0, mask.shape[2]):
        if np.isin(1, mask[:,:,i]):
            slices.append(i)

    steps = np.linspace(slices[0], slices[-1], 6, dtype="uint8")
    fig, axs = plt.subplots(nr_slices, 3, figsize=(10,10))
    overlay = np.ma.masked_where(mask == 0, mask)

    for i, ax in enumerate(axs):
        ax[0].imshow(np.rot90(image[:,:,steps[i]], 3), cmap="gray")
        ax[1].imshow(np.rot90(mask[:,:,steps[i]], 3), cmap="gray")
        ax[2].imshow(np.rot90(image[:,:,steps[i]], 3), cmap="gray")
        ax[2].imshow(np.rot90(overlay[:,:,steps[i]], 3), cmap="prism", alpha=0.4)
    
    axs[0][0].set_title("Image")
    axs[0][1].set_title("Mask")
    axs[0][2].set_title("Overlay")
    
    for ax in axs.ravel():
        ax.set_axis_off()

    fig.subplots_adjust(wspace=-0.75, hspace=0)
    plt.show()

def plot_test(image, mask, prediction, mask_only=True, nr_slices=0):
    prob = torch.softmax(prediction, dim=1)
    heatmap = prob[:, 1, :, :, :].squeeze().detach().cpu()    
    slices_with_mask = list()

    for i in range(0, mask.shape[2]):
        if np.isin(1, mask[:,:,i]):
            slices_with_mask.append(i)

    if nr_slices>0 and mask_only:
        steps = np.linspace(slices_with_mask[0], slices_with_mask[-1], nr_slices, dtype="uint8")
    elif mask_only:
        steps = slices_with_mask

    if nr_slices>0 and not mask_only:
        steps = np.linspace(0, mask.shape[2]-1, nr_slices, dtype="uint8")
    elif not mask_only:
        steps = np.arange(0, mask.shape[2])

    fig, axs = plt.subplots(len(steps), 3, figsize=(9, 3.125*len(steps)))
    # overlay = np.ma.masked_where(mask == 0, mask)
    
    plt.style.use(['science','no-latex'])
    get_ipython().run_line_magic('matplotlib', 'inline')

    for i, ax in enumerate(axs):
        ax[0].imshow(np.rot90(image[:,:,steps[i]], 3), cmap="gray")
        ax[1].imshow(np.rot90(mask[:,:,steps[i]], 3), cmap="gray")
        cbar = ax[2].imshow(np.rot90(heatmap[:,:,steps[i]], 3), cmap="hot", interpolation="nearest")
        axins = inset_axes(ax[2],
                    width="5%",  
                    height="100%",
                    loc="center right",
                    borderpad=-1
                   )
        plt.colorbar(cbar, axins, orientation="vertical")
        # ax[2].imshow(np.rot90(overlay[:,:,steps[i]], 3), cmap="prism", alpha=0.4)
    
    axs[0][0].set_title("Image")
    axs[0][1].set_title("Mask")
    axs[0][2].set_title("Prediction")
    
    for ax in axs.ravel():
        ax.set_axis_off()

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


