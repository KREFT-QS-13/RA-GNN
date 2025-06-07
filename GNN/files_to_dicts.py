"""
Filename: files_to_dict.py
Description: Puts the samples into a Pickle format. This script reads the datasets in folder Datasets to gather them up.
Based on: Olivier Simard's code
"""
import numpy as np
import pickle

import utils_dataset as uds

def preprocess_data(data_folder:str, lengths:list[str], datasets:dict[str,float], realizations:int, num_δs:int, time_δ_file:int, include_Xs:bool=True, start_index:int=0, path_to_save:str=None):
    """
    Preprocesses the data from the given folder and returns a dictionary of the data saved in the pickle file format.

    start_index: int = 0 - if the raw *.npy files are not consecutive, this is the index of the first realization to be used.
    """

    total_samples = realizations*num_δs
    print(f"Total_samples = {total_samples} per size of systems.")
    print(f"Start index = {start_index}")
    print(f"Portions of data = {datasets}")
    print(f"Sizes of systems = {lengths}")
    for L in lengths:
        Lx, Ly = L.split('x')
        Lx, Ly = int(Lx), int(Ly)
        for dataset in datasets.keys():
            
            samples_before_target, samples_target = uds.realization_slicing(datasets, dataset, total_samples)
            realization_indices = np.arange(start_index+samples_before_target, start_index + samples_target, step=num_δs, dtype=int)
            
            print(f"realization_indices = {realization_indices} for {dataset} in {L}")        
            results_dict = {}
            iterator = 0
            for realization_index in realization_indices:
                snapshot_array = []
                for δ_snapshots in range(num_δs):

                    Rs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Rs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    Rps = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Rps_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    Jzzs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Jzzs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    Jpzzs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Jpzzs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    hs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/hxs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    Mg = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Mg_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    NN_corrs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NN_corrs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    NNN_corrs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NNN_corrs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                    if include_Xs:
                        Mg_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Mg_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                        NN_corrs_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NN_corrs_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))
                        NNN_corrs_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NNN_corrs_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization_index+δ_snapshots+1))

                    
                    if include_Xs:
                        tmp_vals_dict = {"Rs": Rs, "Rps": Rps, "Jzzs": Jzzs, "Jpzzs": Jpzzs,
                                            "hs": hs, "Mg": Mg, "NN_corrs": NN_corrs, "NNN_corrs": NNN_corrs,
                                            "Mg_X": Mg_X, "NN_corrs_X": NN_corrs_X, "NNN_corrs_X": NNN_corrs_X}
                    else:
                        tmp_vals_dict = {"Rs": Rs, "Rps": Rps, "Jzzs": Jzzs, "Jpzzs": Jpzzs,
                                            "hs": hs, "Mg": Mg, "NN_corrs": NN_corrs, "NNN_corrs": NNN_corrs}

                    snapshot_array.append(tmp_vals_dict)

                results_dict[iterator] = snapshot_array
                iterator += 1
            # TODO: save to test-fig4 only second round of testing
            if path_to_save: 
                with open(path_to_save + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=dataset), "wb") as tf:
                    pickle.dump(results_dict, tf)
            else:
                with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=dataset), "wb") as tf:
                    pickle.dump(results_dict, tf)
        
    return "Preprocessed data saved to {data_folder}.\n\n" 


