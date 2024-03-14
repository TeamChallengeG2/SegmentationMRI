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
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
from monai.metrics import compute_hausdorff_distance, compute_generalized_dice

# %%

def load_config(filename):
    """Loads config from .json file

    Arguments:
        filename (string): path to .json file

    Returns:
        config dictionary
    """    
    filename = Path(filename)
    with filename.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_config(content, filename):
    """Writes dictionary to .json file

    Arguments:
        content (dictionary)
        filename (string to path)
    """    
    filename = Path(filename)
    with filename.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def export_plot(image, mask, prediction=None, mask_only=False,
                nr_slices=0, export_path=None, slice=None, epoch=None):
    """Exports plot image and mask. If given prediction, will also plot heatmap and overlay.
       Exports all slices if no slice given.

    Arguments:
        image (torch.Tensor): input image
        mask (torch.Tensor): input mask

    Keyword Arguments:
        prediction: prediction from model (default: {None})
        mask_only (boolean): plot slices with mask only (default: {False})
        nr_slices (int): number of slices to export, 0 for all  (default: {0})
        export_path: path (default: {None})
        slice: index slice (default: {None})
        epoch: parameter for filename (default: {None})
    """    
    if prediction is not None:
        prob = torch.softmax(prediction, dim=1)
        heatmap_vol = prob[:, 1, :, :, :].squeeze().detach().cpu() 
        heatmap_spine= prob[:, 2, :, :, :].squeeze().detach().cpu() 
        segm_mask = np.argmax(prob.detach().cpu().squeeze(), axis=0)
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

    if not export_path:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        export_path = "example_results/" + timestr

    isExist = os.path.exists(export_path)
    if not isExist:
        os.makedirs(export_path)

    mpl.use('Agg')
    plt.style.use(['science','no-latex'])
    if slice is not None:
        steps = [10]

    for i in steps:
        if prediction is not None:
            TP_vol = np.logical_and(mask[:,:,i]==1, segm_mask[:,:,i]==1)*1
            TP_spine = np.logical_and(mask[:,:,i]==2, segm_mask[:,:,i]==2)*2
            
            FP_vol = np.logical_and(mask[:,:,i]==0, segm_mask[:,:,i]==1)*3
            FP_spine = np.logical_and(mask[:,:,i]==0, segm_mask[:,:,i]==2)*4

            FN_vol = np.logical_and(mask[:,:,i]==1, segm_mask[:,:,i]==0)*5
            FN_spine = np.logical_and(mask[:,:,i]==2, segm_mask[:,:,i]==0)*6
            
            sum = np.sum([TP_spine, TP_vol, FP_spine, FP_vol, FN_spine, FN_vol], axis=0)
            fig, axs = plt.subplots(2, 3, figsize=(12, 9))
            axs[0][0].imshow(np.rot90(image[:,:,i], 3), cmap="gray") # img + mask
            axs[0][0].imshow(np.rot90(get_overlay(mask[:,:,i]), 3), cmap="gnuplot2", alpha=0.4, vmin=0, vmax=2.5)   

            axs[0][1].imshow(np.rot90(segm_mask[:,:,i], 3), cmap="gnuplot2", vmin=0, vmax=2.5) # prediction
            cbar1 = axs[0][2].imshow(np.rot90(heatmap_vol[:,:,i], 3), cmap="jet") # heatmap volume

            axins = inset_axes(axs[0][2],
                        width="5%",  
                        height="100%",
                        loc="center right",
                        borderpad=-1.5
                        )
            cb1 = plt.colorbar(cbar1, axins, orientation="vertical", format=ticker.FormatStrFormatter("%.2f"))
            cb1.ax.locator_params(nbins=3)

            cmap = {0:[0.0,0.0,0.0,1],
                    1:[0.0,1.0,0.0,1],
                    2:[0.0,1.0,0.0,1],
                    3:[0.5,0.0,0.0,1],
                    4:[0.5,0.0,0.0,1],
                    5:[0.0,0.0,0.5,1],
                    6:[0.0,0.0,0.5,1]}
            
            labels = {0:'', 1:'TP Volume', 2:'TP Spine', 3:'FP Volume', 4:'FP Spine', 5:'FN Volume', 6:'FN Spine' }
            patches =[mpatches.Patch(color=cmap[j], label=labels[j]) for j in cmap]
            overlay = np.array([[cmap[k] for k in j] for j in sum])      
            axs[1][0].imshow(np.rot90(overlay, 3), interpolation="none")
            axs[1][0].legend(handles=patches, loc="upper right", labelspacing=0.1, labelcolor="w")
            axs[1][1].imshow(np.zeros(mask[:,:,i].shape), cmap="gray", vmin=0, vmax=1) # matrix
            cbar2 = axs[1][2].imshow(np.rot90(heatmap_spine[:,:,i], 3), cmap="jet") # heatmap spine

            axins = inset_axes(axs[1][2],
                        width="5%",  
                        height="100%",
                        loc="center right",
                        borderpad=-1.5
                        )
            cb2 = plt.colorbar(cbar2, axins, orientation="vertical", format=ticker.FormatStrFormatter("%.2f"))
            cb2.ax.locator_params(nbins=3)

            labels = ["Image", "Prediction mask", "Heatmap volume", "Overlap", "", "Heatmap spine"]
            for ax, label in zip(axs.ravel(), labels):
                ax.set_title(label)

        else:
            fig, axs = plt.subplots(1, 2, figsize=(4.5, 9))
            axs[0].imshow(np.rot90(image[:,:,steps[i]], 3), cmap="gray")
            axs[1].imshow(np.rot90(mask[:,:,steps[i]], 3), cmap="gnuplot2")   

        for ax in axs.ravel():
            ax.set_axis_off()
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        if epoch is not None:
            i = epoch

        plt.savefig(export_path + f"/{i}.png")
        plt.close()        

def get_overlay(img):
    return np.ma.masked_where(img==0, img)

def calc_dsc(pred_mask, mask):
    # return compute_generalized_dice(pred_mask.detach().cpu(), mask.detach().cpu())
    mask=mask.detach().cpu().numpy()
    smooth=1e-5
    intersection=(pred_mask*mask).sum()
    return (2.*intersection+smooth)/(pred_mask.sum()+mask.sum()+smooth)

def calc_hd95(pred_mask, mask):
    return compute_hausdorff_distance(pred_mask.detach().cpu(), mask.detach().cpu(), percentile=95.0)

def prediction_to_mask(prediction):
    """
    Converts raw logits from prediction to probability map.
    """        
    prob = torch.softmax(prediction, dim=1)
    return np.argmax(prob.detach().cpu().squeeze(), axis=0)

def plot_3D_mesh(volume_obj):
    import meshlib.mrmeshpy as mr
    import meshlib.mrmeshnumpy as mrn
    import open3d as o3d
    
    simpleVolume = mrn.simpleVolumeFrom3Darray(volume_obj.mask_volume.numpy())
    floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
    mesh = mr.gridToMesh(floatGrid , mr.Vector3f(0.1, 0.1, 0.1), 0.5)
    mr.saveMesh(mesh, "visualization/volume.stl")
    mesh = o3d.io.read_triangle_mesh("visualization/volume.stl")
    mesh = mesh.compute_vertex_normals()
    mesh.paint_uniform_color([119, 0, 255])


    simpleVolume2 = mrn.simpleVolumeFrom3Darray(volume_obj.mask_spine.numpy())
    floatGrid2 = mr.simpleVolumeToDenseGrid(simpleVolume2)
    mesh2 = mr.gridToMesh(floatGrid2 , mr.Vector3f(0.1, 0.1, 0.1), 0.5)
    mr.saveMesh(mesh2, "visualization/spine.stl")
    mesh2 = o3d.io.read_triangle_mesh("visualization/spine.stl")
    mesh2 = mesh2.compute_vertex_normals()
    mesh2.paint_uniform_color([1, 0.756, 0.239])
    o3d.visualization.draw_geometries([mesh, mesh2])