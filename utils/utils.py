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

def plot_overlay(img, mask, color_overlay=[(255, 0, 0)], slice_z=12, save=False):
    """
    Plots segmentation mask and border over an image.

    Parameters
    ----------
    img : numpy.ndarray or torch.Tensor
        image
    mask : numpy.ndarray or torch.Tensor
        mask
    color_overlay : List of tuples
        The colors that are used as overlay. 

    Returns
    -------
    None.

    """
    img -= torch.min(img)
    img /= torch.max(img)
    img = img.numpy()
    mask = mask.numpy()

    overlay = color.label2rgb(mask, img, colors=color_overlay, alpha=0.005, bg_label=0, bg_color=None)
    output = segmentation.mark_boundaries(overlay, mask, mode='inner', color=color_overlay)
    fig, axes = plt.subplots(1, 1)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.imshow(np.clip(output[:,:,slice_z,:], 0, 1))
    if save is not False:
        plt.savefig(f"{save}/plot.png")
    plt.show()

def _rainbow_colors(num_colors):
    hues = np.linspace(0, 1, num_colors)
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in map(lambda h: colorsys.hsv_to_rgb(h, 1.0, 1.0), hues)]
    return colors    
# %%
