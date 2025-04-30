"""
Filename: file_utils.py
Description: Module containing functions that prepare the graph datasets fed into the PNA GNN (training_loader,validation_loader,testing_loader). The datasets are prepared according 
to the study case chosen.
Author: Olivier Simard
Date: 2024-08-11
License: MIT License
"""
import numpy as np
import torch
#dataloader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.linalg import norm
import pickle
from typing import List, Dict, Tuple
import json, os

__all__=["split_string_around_substring","tup_edges","load_datasets_mag_NN_NNN_δ"]

def load_training_parameters(path: str) -> Dict:
    with open(path, 'r') as jsonf:
        parameters = json.load(jsonf)

    gnn_parms = parameters["GNN_hyperparameters"]
    phys_parms = parameters["Physical_hyperparameters"]
    gen_parms = parameters["General_parameters"]

    jsonf.close()
    return gnn_parms, phys_parms, gen_parms
    
def load_test_parameters(path: str) -> Dict:
    with open(path, 'r') as jsonf:
        parameters = json.load(jsonf)

    test_parms = parameters["Test_hyperparameters"]

    jsonf.close()
    return test_parms

def realization_slicing(dictionary: dict[str,float], target_key: str, total_samples: int) -> float:
    total = 0.0
    total_before_target = 0.0
    for key in dictionary:
        if key == target_key:
            total += dictionary[key]
            break
        else:
            total_before_target += dictionary[key]
            total += dictionary[key]

    samples_before_target = int(total_before_target*total_samples)
    samples_target = int(total*total_samples)
    return samples_before_target, samples_target

def check_pickle_files_exist(phys_parms: dict, gen_parms: dict) -> bool:
    for L in phys_parms["Ls"]:
        for dataset in phys_parms["datasets"].keys():
            pickle_path = os.path.join(gen_parms["folder_datasets"], f"MPS_dict_{L}_{dataset}.pkl")
            if not os.path.exists(pickle_path):
                return False
    return True

def split_string_around_substring(s: str, substring: str) -> Tuple[str,str]:
    # Find the index of the first occurrence of the substring
    index = s.find(substring)
    
    if index != -1:
        # Split the string at the found index
        part1 = s[:index]
        part2 = s[index + len(substring):]
        return part1, part2
    else:
        # If no match is found, return the original string and an empty string
        return s, ""
    
def tup_edges(Lx: int,Ly: int,*,is_NNN: bool=False,is_XZ: bool=False) -> Tuple[int,int]:
    a1, a2 = np.array([1.,0.]), np.array([0.,1.]) # rectangular lattice
    coordinates = []
    for y in range(Ly):
        for x in range(Lx):
            a_tmp = x*a1+y*a2
            coordinates.append(a_tmp)
    # create graph
    edges = []
    for idx_i in range(len(coordinates)):
        ii = coordinates[idx_i]
        for idx_j in range(len(coordinates)):
            jj = coordinates[idx_j]
            dist = norm(jj-ii)
            edges.append((idx_i,idx_j,dist))
    paths_by_length = {}
    for ee in edges:
        path_length = np.round(ee[2],8)
        if path_length not in paths_by_length:
            paths_by_length[path_length] = []
        paths_by_length[path_length].append((ee[0], ee[1]))
    keys = sorted(list(paths_by_length.keys()))

    edges = None
    if not is_NNN:
        if not is_XZ:
            edges = paths_by_length[keys[1]]
        else:
            edges = paths_by_length[keys[1]] + paths_by_length[keys[1]] # Z, X NNs
    else:
        if not is_XZ:
            edges = paths_by_length[keys[1]] + paths_by_length[keys[2]]
        else:
            edges = paths_by_length[keys[1]] + paths_by_length[keys[2]] + paths_by_length[keys[1]] + paths_by_length[keys[2]] # Z, X NNs+NNNs
    
    return edges

def load_datasets_mag_NN_NNN_δ(num_realizations: List[int], Ls: List[str], num_deltas: int, *, incl_scnd: bool = False, trgt_diff: bool = True, meas_basis: str = "Z",
                               data_folder: str = "./dataset_mps_NNN", datasets: List[str] = ["training", "validation", "test"], batch_sizes: List[int] = [32, 32, 32], dtype = torch.float64):
    """
    Prepares the batches of data for training. It loads the training datasets and properly set up the `Dataloader` objects, 
    which consist in lists of `Data` objects.

        Args:
            num_realizations: total number of `Data` objects that will be split into training, validation and test sets.
            Ls: array of square lattice lengths describing the Rydberg arrays.
            num_deltas: length of the array of Rabi frequencies pacted into one graph dataset sample.
            *
            incl_scnd: whether to consider NNN correlators in the loss function during the training.
            trgt_diff: whether the targets are expressed in relative NN distances or in absolute distances.
            meas_basis: basis in which the observables (training input) are measured.
            data_folder: path string to the dataset folder.
            datasets: datasets to prepare.
            batch_sizes: array of batch sizes for the training, validation and test sets.
            dtype: torch data type.

        Returns:
            A three-tuple of training, validation and test sets.
    """
    train_loader, validation_loader, test_loader = None, None, None
    for it in range(len(datasets)):
        data_list = []
        
        for L in Ls:
            Lx, Ly = L.split('x')
            Lx, Ly = int(Lx), int(Ly)
            edges = tup_edges(Lx,Ly,is_NNN=True,is_XZ=(True if meas_basis=="ZX" else False))
            num_edges = len(edges)
            num_nodes = Lx*Ly
            # Load datasets
            with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=datasets[it]), "rb") as tf:
                results_dict = pickle.load(tf)
            
            edge_index = torch.as_tensor(edges, dtype=torch.long).T

            Rs_nom = np.load(data_folder + "/{Lx}x{Ly}/Rs_nom.npy".format(Lx = Lx, Ly = Ly))
            Rps_nom = np.load(data_folder + "/{Lx}x{Ly}/Rps_nom.npy".format(Lx = Lx, Ly = Ly))
            for realization in range(num_realizations[it]):
                
                edge_attr = torch.empty((num_edges,num_deltas), dtype=dtype)
                x_labels = torch.empty((num_nodes,num_deltas), dtype=dtype)
                x = torch.empty((num_nodes,num_deltas), dtype=dtype)
                if trgt_diff:
                    ΔRs = np.array(results_dict[realization][0]['Rs']) - Rs_nom
                    ΔRps = np.array(results_dict[realization][0]['Rps']) - Rps_nom
                else:
                    ΔRs = np.array(results_dict[realization][0]['Rs'])
                    ΔRps = np.array(results_dict[realization][0]['Rps'])
                
                if not incl_scnd: # decide to include the second-nearest neighbor targets in the training or not
                    if not meas_basis=="ZX":
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                    else:
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,np.zeros_like(ΔRps),np.zeros_like(ΔRs),np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                else:
                    if not meas_basis=="ZX":
                        edge_labels = torch.as_tensor(
                                np.concatenate(
                                    (ΔRs,ΔRps), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                    else:
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,ΔRps,np.zeros_like(ΔRs),np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                
                for δ_snapshot in range(num_deltas):

                    x_labels_δ = torch.as_tensor(
                            results_dict[realization][δ_snapshot]['hs'], dtype=dtype
                            )
                    x_labels[:,δ_snapshot] = x_labels_δ
                    
                    if meas_basis=="Z":
                        x_δ = torch.as_tensor(
                            np.ones_like(results_dict[realization][δ_snapshot]['Mg']), 
                            dtype=dtype
                        ) # time-independent

                        edge_attr_δ =  torch.as_tensor(np.concatenate(
                            (np.array(results_dict[realization][δ_snapshot]['NN_corrs']),np.ones_like(results_dict[realization][δ_snapshot]['NNN_corrs'])), axis=0
                        ), dtype=dtype)
                    elif meas_basis=="ZX":
                        
                        x_δ = torch.as_tensor(
                            np.array(results_dict[realization][δ_snapshot]['Mg']), 
                            dtype=dtype
                        ) # time-independent
                        
                        edge_attr_δ =  torch.as_tensor(
                            np.concatenate(
                                (
                                    np.array(results_dict[realization][δ_snapshot]['NN_corrs']),
                                    np.array(results_dict[realization][δ_snapshot]['NNN_corrs']),
                                    np.array(results_dict[realization][δ_snapshot]['NN_corrs_X']),
                                    np.array(results_dict[realization][δ_snapshot]['NNN_corrs_X'])
                                ), axis=0
                            ), 
                            dtype=dtype
                        )

                    x[:,δ_snapshot] = x_δ
                    edge_attr[:,δ_snapshot] = edge_attr_δ

                graph = Data(
                    x = x, # node features
                    edge_index = edge_index,
                    edge_attr = edge_attr, # edge features
                    x_labels = x_labels,
                    edge_labels = edge_labels
                )

                data_list.append(graph)

                if realization==0:
                    print(f"For size {Lx}x{Ly} and dataset {datasets[it]}, x.shape = {x.shape}, edge_index.shape = {edge_index.shape}, edge_attr = {edge_attr.shape}")

        if datasets[it] == "training":
            train_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "validation":
            validation_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "test":
            test_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        else:
            raise ValueError("Nonexisting dataset keyword inputted!")
    
    return train_loader, validation_loader, test_loader






