"""
Filename: files_to_dict.py
Description: Puts the samples into a Pickle format. This script reads the datasets in folder Datasets to gather them up.
Author: Olivier Simard
Date: 2024-08-11
License: MIT License
"""
import numpy as np
import pickle

# path pointing to dataset folder 
data_folder = "../data-gen/Datasets/dataset_NO_Dr_X_Mg_NN_NNN_delta_one"

# cluster sizes
lengths = ['4x4']
# order with which datasets are produced.
datasets = ["training", "validation", "test"]
total_samples = 2000
num_δs = 10 # 10
time_δ_file = num_δs # can be different from num_δs if time dependent calculation
total_samples*=num_δs
portion_training = 0.8
training_samples = int(total_samples*portion_training)
portion_valid = 0.1
validation_samples = int(total_samples*portion_valid)
portion_test = 0.1
test_samples = int(total_samples*portion_test)
include_Xs = True # pickle the X-basis data
assert (portion_training+portion_valid+portion_test)==1.0, "The proportions have to sum up to 1."
assert np.mod(validation_samples,num_δs)==0 and np.mod(test_samples,num_δs)==0, "The number of validation samples and\
      test samples must be an integer divisible by the number of node and edge features (num_δs)."

# during training, datasets sharing the same Jzz configuration need to be bundled up
# for the testing, this means that by exclusion, bundles are part of the same group

for L in lengths:
    Lx, Ly = L.split('x')
    Lx, Ly = int(Lx), int(Ly)
    for dataset in datasets:

        if dataset == 'training':
            realizations = np.arange(0, training_samples, step=num_δs, dtype=int)
        elif dataset == 'validation':
            realizations = np.arange(training_samples, training_samples+validation_samples, step=num_δs, dtype=int)
        else:
            realizations = np.arange(training_samples+validation_samples, training_samples+validation_samples+test_samples, step=num_δs, dtype=int)
        print(realizations)        
        results_dict = {}
        iterator = 0
        for realization in realizations:
            snapshot_array = []
            for δ_snapshots in range(num_δs):

                Rs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Rs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Rps = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Rps_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Jzzs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Jzzs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Jpzzs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Jpzzs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                hs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/hxs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Mg = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Mg_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                NN_corrs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NN_corrs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                NNN_corrs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NNN_corrs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                if include_Xs:
                    Mg_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Mg_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                    NN_corrs_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NN_corrs_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                    NNN_corrs_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NNN_corrs_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))

                
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
        with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=dataset), "wb") as tf:
            pickle.dump(results_dict, tf)