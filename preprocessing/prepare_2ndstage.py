import argparse
import logging
import os
import pathlib
from typing import List
import time
import yaml

import torch # TODO: [BAFO]
from data_stuff.utils import load_yaml # TODO: [BAFO]
import domain_classes.domain as domain

from torch import stack, load, unsqueeze, save, Tensor
from tqdm.auto import tqdm

from networks.unet import UNet
import preprocessing.prepare_1ststage as prep_1hp
from preprocessing.prepare_1ststage import prepare_dataset
from domain_classes.domain import Domain
from domain_classes.heat_pump import HeatPump
from domain_classes.utils_2hp import save_config_of_separate_inputs, save_config_of_merged_inputs, save_yaml
from domain_classes.stitching import Stitching
from utils.prepare_paths import Paths2HP


def prepare_demonstrator_input_2nd_stage(paths: Paths2HP, inputs_1hp: str, groundtruth_info, device: str = "cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - 1hpnn is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    - device: attention, all stored need to be produced on cpu for later pin_memory=True and all other can be gpu
    """

    model_1HP = UNet(in_channels=len(inputs_1hp)).float()
    model_1HP.load(paths.model_1hp_path, device)
    
    ## prepare 2hp dataset for 1st stage  
    # with open(paths.dataset_model_trained_with_prep_path / "info.yaml", "r") as file: # norm with data from dataset that NN was trained with!!
    #     info = yaml.safe_load(file)
    #prepare_dataset(paths, inputs_1hp, info=info, power2trafo=False) #TODO: Muss vorbereitet vorliegen!!!!!
    # print(f"Domain prepared ({paths.dataset_1st_prep_path})")


    # Zwei Datenpunkte:
    # prep_1hp.prepare_demonstrator_input(paths, groundtruth_info, -2.354183660481613122e-03, 9.921513056473029286e-11, info)
    # prep_1hp.prepare_demonstrator_input(paths, groundtruth_info, -2.088833953514686054e-03, 3.882683654233747158e-11, info)


    # for each run, load domain and 1hp-boxes
    run_file = "RUN_0.pt"
    domain = Domain(paths.dataset_1st_prep_path, stitching_method="max", file_name=run_file) # Domäne von Punkt 0
    # if domain.skip_datapoint: logging.warning(f"Skipping {0}") # Pressure Gradient [-] oder Permeability X [m^2] nicht in [0,1]

    single_hps, inputs, label = prep2_test(paths, run_file)
    # single_hps = domain.extract_hp_boxes(device=device)

    run_id = f'{run_file.split(".")[0]}_'
    single_hps, _ = prepare_hp_boxes(paths, model_1HP, single_hps, domain, run_id, save_bool=True) # apply learned NN to predict the heat plumes


    # save infos of info file about separated (only 2!) inputs
    # save_config_of_separate_inputs(domain.info, path=paths.datasets_boxes_prep_path)

    return domain, single_hps

# test
def get_name_from_index(index: int, info):
        for property, values in info["Inputs"].items():
            if values["index"] == index:
                return property


# Vorbedingung: # Pressure Gradient [-] oder Permeability X [m^2] in [0,1]
def prep2_test(paths, file_name):

    # __init__

    info_path = paths.dataset_1st_prep_path
    info = load_yaml(info_path, "info")
    size: tuple[int, int] = [info["CellsNumber"][0], info["CellsNumber"][1],]  # (x, y), cell-ids
    background_temperature: float = 10.6

    file_path = os.path.join(info_path, "Inputs", file_name)
    inputs = load(file_path)
    file_path = os.path.join(info_path, "Labels", file_name)
    label = load(file_path)

    prediction: torch.tensor = (torch.ones(size) * background_temperature).to("cpu")
    stitching: Stitching = Stitching("max", background_temperature)
    normed_label: bool = True
    file_name: str = file_name

    field_idx = info["Inputs"]["Pressure Gradient [-]"]["index"]
    p_related_field = inputs[field_idx, :, :]

    # extract_hp_boxes

    field_idx = info["Inputs"]["Material ID"]["index"]
    material_ids = inputs[field_idx, :, :]

    size_hp_box = torch.tensor([info["CellsNumberPrior"][0],info["CellsNumberPrior"][1],])
    distance_hp_corner = torch.tensor([info["PositionHPPrior"][1], info["PositionHPPrior"][0]-2])
    hp_boxes = []
    pos_hps = torch.stack(list(torch.where(material_ids == torch.max(material_ids))), dim=0).T
    print(f"pos hp: f{pos_hps}")
    names_inputs = [get_name_from_index(i, info) for i in range(inputs.shape[0])] #TODO: was ist das?

    for idx in range(len(pos_hps)):
        try:
            pos_hp = pos_hps[idx]

            corner_ll, corner_ur = domain.get_box_corners(pos_hp, size_hp_box, distance_hp_corner, inputs.shape[1:], run_name=file_name,)
            
            tmp_input = inputs[:, corner_ll[0] : corner_ur[0], corner_ll[1] : corner_ur[1]].detach().clone()
            tmp_label = label[:, corner_ll[0] : corner_ur[0], corner_ll[1] : corner_ur[1]].detach().clone()


            tmp_mat_ids = torch.stack(list(torch.where(tmp_input == torch.max(material_ids))), dim=0).T
            if len(tmp_mat_ids) > 1:
                for i in range(len(tmp_mat_ids)):
                    tmp_pos = tmp_mat_ids[i]
                    if (tmp_pos[1:2] != distance_hp_corner).all():
                        tmp_input[tmp_pos[0], tmp_pos[1], tmp_pos[2]] = 0

            tmp_hp = HeatPump(id=idx, pos=pos_hp, orientation=0, inputs=tmp_input, names=names_inputs, dist_corner_hp=distance_hp_corner, label=tmp_label, device="cpu",)
  
            if "SDF" in info["Inputs"]:
                tmp_hp.recalc_sdf(info)

            hp_boxes.append(tmp_hp)
            logging.info(f"HP BOX at {pos_hp} is with ({corner_ll}, {corner_ur}) in domain")
        except:
            logging.warning(f"BOX of HP {idx} at {pos_hp} is not in domain")
                
    return hp_boxes, inputs, label


def prepare_dataset_for_2nd_stage(paths: Paths2HP, inputs_1hp: str, device: str = "cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - 1hpnn is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    - device: attention, all stored need to be produced on cpu for later pin_memory=True and all other can be gpu
    """
    
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

# prepare domain dataset if not yet done
    ## load model from 1st stage
    time_start_prep_domain = time.perf_counter()
    model_1HP = UNet(in_channels=len(inputs_1hp)).float()
    model_1HP.load(paths.model_1hp_path, device)
    
    ## prepare 2hp dataset for 1st stage
    if not os.path.exists(paths.dataset_1st_prep_path):        
        # norm with data from dataset that NN was trained with!!
        with open(paths.dataset_model_trained_with_prep_path / "info.yaml", "r") as file:
            info = yaml.safe_load(file)
        prepare_dataset(paths, inputs_1hp, info=info, power2trafo=False)
    print(f"Domain prepared ({paths.dataset_1st_prep_path})")

# prepare dataset for 2nd stage
    time_start_prep_2hp = time.perf_counter()
    avg_time_inference_1hp = 0
    list_runs = os.listdir(paths.dataset_1st_prep_path / "Inputs")
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        # for each run, load domain and 1hp-boxes
        run_id = f'{run_file.split(".")[0]}_'
        domain = Domain(paths.dataset_1st_prep_path, stitching_method="max", file_name=run_file)
        ## generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        single_hps = domain.extract_hp_boxes(device)
        # apply learned NN to predict the heat plumes
        single_hps, avg_time_inference_1hp = prepare_hp_boxes(paths, model_1HP, single_hps, domain, run_id, avg_time_inference_1hp, save_bool=True)
        
    time_end = time.perf_counter()
    avg_inference_times = avg_time_inference_1hp / len(list_runs)

    # save infos of info file about separated (only 2!) inputs
    save_config_of_separate_inputs(domain.info, path=paths.datasets_boxes_prep_path)

    # save measurements
    with open(paths.datasets_boxes_prep_path / "measurements.yaml", "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"model 1HP: {paths.model_1hp_path}\n")
        f.write(f"input params: {inputs_1hp}\n")
        f.write(f"separate inputs: {True}\n")
        f.write(f"location of prepared domain dataset: {paths.dataset_1st_prep_path}\n")
        f.write(f"name of dataset prepared with: {paths.dataset_model_trained_with_prep_path}\n")
        f.write(f"name of dataset domain: {paths.raw_path.name}\n")
        f.write(f"name_destination_folder: {paths.datasets_boxes_prep_path}\n")
        f.write(f"avg inference times for 1HP-NN in seconds: {avg_inference_times}\n")
        f.write(f"device: {device}\n")
        f.write(f"duration of preparing domain in seconds: {(time_start_prep_2hp-time_start_prep_domain)}\n")
        f.write(f"duration of preparing 2HP in seconds: {(time_end-time_start_prep_2hp)}\n")
        f.write(f"duration of preparing 2HP /run in seconds: {(time_end-time_start_prep_2hp)/len(list_runs)}\n")
        f.write(f"duration of whole process in seconds: {(time_end-time_begin)}\n")

    return domain, single_hps

def load_and_prepare_for_2nd_stage(paths: Paths2HP, inputs_1hp: str, run_id: int, device: str = "cpu"):
    model_1HP = UNet(in_channels=len(inputs_1hp)).float()
    model_1HP.load(paths.model_1hp_path, device)
    model_1HP.eval()

    domain = Domain(paths.dataset_1st_prep_path, stitching_method="max", file_name=f"RUN_{run_id}.pt")
    single_hps = domain.extract_hp_boxes(device)
    single_hps, _ = prepare_hp_boxes(paths, model_1HP, single_hps, domain, run_id, save_bool=False) # apply 1HP-NN to predict the heat plumes

    # TODO replace with loading from file  - requires saving the position of a hp within its domain and the connection domain - single hps  
    return domain, single_hps

def prepare_hp_boxes(paths:Paths2HP, model_1HP:UNet, single_hps:List[HeatPump], domain:Domain, run_id:int, avg_time_inference_1hp:float=0, save_bool:bool=True):
    hp: HeatPump
    for hp in single_hps:
        time_start_run_1hp = time.perf_counter()
        hp.primary_temp_field = hp.apply_nn(model_1HP)
        avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp
        hp.primary_temp_field = domain.reverse_norm(hp.primary_temp_field, property="Temperature [C]")
    avg_time_inference_1hp /= len(single_hps)

    for hp in single_hps:
        hp.get_other_temp_field(single_hps)

    for hp in single_hps:
        hp.primary_temp_field = domain.norm(hp.primary_temp_field, property="Temperature [C]")
        hp.other_temp_field = domain.norm(hp.other_temp_field, property="Temperature [C]")
        inputs = stack([hp.primary_temp_field, hp.other_temp_field])
        if save_bool:
            hp.save(run_id=run_id, dir=paths.datasets_boxes_prep_path, inputs_all=inputs,)
    return single_hps, avg_time_inference_1hp

def merge_inputs_for_2HPNN(path_separate_inputs:pathlib.Path, path_merged_inputs:pathlib.Path, stitching_method:str="max"):
    begin = time.perf_counter()
    assert stitching_method == "max", "Other than max stitching required reasonable background temp and therefor potentially norming."
    stitching = Stitching(stitching_method, background_temperature=0)
    
    (path_merged_inputs/"Inputs").mkdir(exist_ok=True)

    begin_prep = time.perf_counter()
    # get separate inputs if exist
    for file in (path_separate_inputs/"Inputs").iterdir():
        input = load(file)
        # merge inputs via stitching
        input = stitching(input[0], input[1])
        # save merged inputs
        input = unsqueeze(Tensor(input), 0)
        save(input, path_merged_inputs/"Inputs"/file.name)
    end_prep = time.perf_counter()

    # save config of merged inputs
    info_separate = yaml.load(open(path_separate_inputs/"info.yaml", "r"), Loader=yaml.FullLoader)
    save_config_of_merged_inputs(info_separate, path_merged_inputs)

    # save command line arguments
    cla = {
        "dataset_separate": path_separate_inputs.name,
        "command": "prepare_2HP_merged_inputs.py"
    }
    save_yaml(cla, path=path_merged_inputs, name_file="command_line_args")
    end = time.perf_counter()

    # save times in measurements.yaml (also copy relevant ones from separate)
    measurements_prep_separate = yaml.load(open(path_separate_inputs/"measurements.yaml", "r"), Loader=yaml.FullLoader)
    num_dp = len(list((path_separate_inputs/"Inputs").iterdir()))
    duration_prep = end_prep - begin_prep
    duration_prep_avg = duration_prep / num_dp
    measurements = {
        "duration of preparation in seconds": duration_prep,
        "duration of preparing 2HP /run in seconds": duration_prep_avg,
        "duration total in seconds": end - begin,
        "number of datapoints": num_dp,
        "separate-preparation": {"duration of preparing domain in seconds": measurements_prep_separate["duration of preparing domain in seconds"],
                                    "duration of preparing 2HP /run in seconds": measurements_prep_separate["duration of preparing 2HP /run in seconds"],
                                    "duration of preparing 2HP in seconds": measurements_prep_separate["duration of preparing 2HP in seconds"],
                                    "duration of whole process in seconds": measurements_prep_separate["duration of whole process in seconds"]},
    }
    save_yaml(measurements, path=path_merged_inputs, name_file="measurements")


def main_merge_inputs(dataset: str, merge: bool):
    #get dir of prepare_2HP_separate_inputs
    paths = yaml.load(open("paths.yaml", "r"), Loader=yaml.FullLoader)
    dir_separate_inputs = pathlib.Path(paths["datasets_prepared_dir_2hp"])
    path_separate_inputs = dir_separate_inputs / dataset

    if merge:
        path_merged_inputs = dir_separate_inputs / f"{dataset}_merged"
        path_merged_inputs.mkdir(exist_ok=True)
        # labels are the same as for separate inputs
        os.system(f"cp -r '{path_separate_inputs/'Labels'}' '{path_merged_inputs}'")

        if os.path.exists(path_separate_inputs):
            merge_inputs_for_2HPNN(path_separate_inputs, path_merged_inputs, stitching_method="max")
        else:
            print(f"Could not find prepared dataset with separate inputs at {path_separate_inputs}.")
    else:
        if os.path.exists(path_separate_inputs):
            print("You need to set --merge_inputs=True to merge inputs otherwise you're done. Your separate inputs are already prepared.")
        else:
            print(f"Could not find prepared dataset with separate inputs at {path_separate_inputs}. Please go to file main.py for that.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset_2hps_1fixed_10dp inputs_gki100 boxes")
    parser.add_argument("--merge", type=bool, default=False)
    args = parser.parse_args()

    main_merge_inputs(args.dataset, args.merge)
