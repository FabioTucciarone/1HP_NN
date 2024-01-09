import logging
import time
import warnings
from dataclasses import dataclass, field
from math import inf
from typing import Dict

import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from data_stuff.transforms import NormalizeTransform
from networks.unet import UNet
from utils.measurements import measure_len_width_1K_isoline

mpl.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = [8, 2.5]

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "forschungsprojekt-pumpen-demonstrator", "demonstrator_backend"))
import model_communication as mc
# TODO: look at vispy library for plotting 3D data

@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs = {"cmap": "RdBu_r", 
                           "extent": extent}

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"


def get_plots(model: UNet, x: torch.Tensor, y: torch.Tensor, info, norm, color_palette, device: str = "cpu"):

    x = torch.unsqueeze(x, 0)
    y_out = model(x).to(device)

    # reverse transform for plotting real values
    x = norm.reverse(x.detach().squeeze(), "Inputs").cpu()
    y = norm.reverse(y.detach(),"Labels")[0].cpu()
    y_out = norm.reverse(y_out.detach()[0],"Labels")[0].cpu()

    dict_to_plot = prepare_data_to_plot(x, y, y_out, info)
    display_data = mc.DisplayData(color_palette)
    display_data.set_figure("model_result", dict_to_plot["t_out"].data.T, **dict_to_plot["t_out"].imshowargs)
    display_data.set_figure("groundtruth", dict_to_plot["t_true"].data.T, **dict_to_plot["t_true"].imshowargs)
    display_data.set_figure("error_measure", dict_to_plot["error"].data.T, **dict_to_plot["error"].imshowargs)
    display_data.average_error = torch.mean(torch.abs(y_out - y)).item()

    return display_data


def get_2hp_plots(model: UNet, model_2hp_info, hp_inputs, corners_ll, corner_dist, color_palette, device: str = "cpu"):
    # TODO: Hier noch langsam

    size_hp_box = model_2hp_info["CellsNumberPrior"]
    image_shape = model_2hp_info["OutFieldShape"]
    out_image = torch.full(image_shape, 10.6)

    with torch.no_grad():
        model.eval()
        y_out = model(hp_inputs.detach()) # TODO: Zwischen 0.02s und 0.25s ...

    for i in range(2):
        import preprocessing.prepare_2ndstage as prep
        y = y_out[i].detach()[0]
        y = prep.reverse_temperature_norm(y, model_2hp_info)

        ll_x = corners_ll[i][0] - corner_dist[1]
        ll_y = corners_ll[i][1] - corner_dist[0]
        ur_x = ll_x + size_hp_box[0]
        ur_y = ll_y + size_hp_box[1]
        clip_ll_x = max(ll_x, 0)
        clip_ll_y = max(ll_y, 0)
        clip_ur_x = min(ur_x, image_shape[0])
        clip_ur_y = min(ur_y, image_shape[1])

        out_image[clip_ll_x : clip_ur_x, clip_ll_y : clip_ur_y] = y[clip_ll_x - ll_x : y.shape[0] - ur_x + clip_ur_x, 
                                                                    clip_ll_y - ll_y : y.shape[1] - ur_y + clip_ur_y]

    extent_highs = out_image.shape * np.array(model_2hp_info["CellsSize"][:2])
    display_data = mc.DisplayData(color_palette)
    display_data.set_figure("model_result", out_image.T, cmap="RdBu_r", extent=(0, extent_highs[0], extent_highs[1], 0))

    return display_data


def visualizations(model: UNet, dataloader: DataLoader, device: str, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    norm = dataloader.dataset.dataset.norm
    info = dataloader.dataset.dataset.info
    model.eval()
    settings_pic = {"format": pic_format}

    current_id = 0
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            settings_pic["name"] = f"{plot_path}_{current_id}"

            x = torch.unsqueeze(inputs[datapoint_id].to(device), 0)
            y = labels[datapoint_id]
            y_out = model(x).to(device)

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)

            plot_datafields(dict_to_plot, settings_pic)
            plot_isolines(dict_to_plot, settings_pic)
            # measure_len_width_1K_isoline(dict_to_plot)

            if current_id >= amount_datapoints_to_visu-1:
                return None
            current_id += 1

def reverse_norm_one_dp(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu().squeeze(), "Inputs")
    y = norm.reverse(y.detach().cpu(),"Labels")[0]
    y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]
    return x, y, y_out

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())
    extent_highs = (np.array(info["CellsSize"][:2]) * y.shape)

    dict_to_plot = {
        "t_true": DataToVisualize(y, "Label: Temperature in [°C]",extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]",extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y-y_out), "Absolute error in [°C]",extent_highs),
    }
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], input,extent_highs)

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], settings_pic: dict):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    num_subplots = len(data)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)
    
    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(datapoint.name)
        if name in ["t_true", "t_out"]:  
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
            plt.clabel(CS, inline=1, fontsize=10)

        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel("x [m]")
        _aligned_colorbar()

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{settings_pic['name']}.{settings_pic['format']}", format=settings_pic['format'])

def plot_isolines(data: Dict[str, DataToVisualize], settings_pic: dict):
    # plot isolines of temperature fields
    num_subplots = 3 if "Original Temperature [C]" in data.keys() else 2
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            data[name].data = torch.flip(data[name].data, dims=[1])
            plt.title("Isolines of "+data[name].name)
            plt.contourf(data[name].data.T, **data[name].contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(ticks=[11.6, 15.6])
        except:
            pass

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{settings_pic['name']}_isolines.{settings_pic['format']}", format=settings_pic['format'])


def infer_all_and_summed_pic(model: UNet, dataloader: DataLoader, device: str):
    '''
    sum inference time (including reverse-norming) and pixelwise error over all datapoints
    '''
    
    norm = dataloader.dataset.dataset.norm
    model.eval()

    current_id = 0
    avg_inference_time = 0
    summed_error_pic = torch.zeros_like(torch.Tensor(dataloader.dataset[0][0][0])).cpu()


    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            start_time = time.perf_counter()
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)
            y_out = model(x).to(device)
            y = labels[datapoint_id]

            # reverse transform for plotting real values
            x = norm.reverse(x.cpu().detach().squeeze(), "Inputs")
            y = norm.reverse(y.cpu().detach(),"Labels")[0]
            y_out = norm.reverse(y_out.cpu().detach()[0],"Labels")[0]
            avg_inference_time += (time.perf_counter() - start_time)
            summed_error_pic += abs(y-y_out)

            current_id += 1

    avg_inference_time /= current_id
    summed_error_pic /= current_id
    return avg_inference_time, summed_error_pic

def plot_avg_error_cellwise(dataloader, summed_error_pic, settings_pic: dict):
    # plot avg error cellwise AND return time measurements for inference

    info = dataloader.dataset.dataset.info
    extent_highs = (np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape)
    extent = (0,int(extent_highs[0]),int(extent_highs[1]),0)

    plt.figure()
    plt.imshow(summed_error_pic.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Cellwise averaged error [°C]")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(f"{settings_pic['folder']}/avg_error.{settings_pic['format']}", format=settings_pic['format'])

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)

def aligne_colorbar_to_figure(figure: Figure, mappable, *args, **kwargs):
    cax = make_axes_locatable(figure.gca()).append_axes("right", size=0.3, pad=0.05)
    figure.colorbar(mappable, *args, cax=cax, **kwargs)