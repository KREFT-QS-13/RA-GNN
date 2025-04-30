"""
Filename: GNN_testing.py
Description: Reads off the JSON file `params.json`. Script taking care of the GNN testing based off a trained model to be passed via the command line. It will output a series of files in hdf5
format in the folder whose path is specified by the parameter `folder_models`.
Author: Olivier Simard
Date: 2024-08-11
License: MIT License
"""
import torch
#miscellaneous
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import sys
from time import time
from re import search
import h5py
import json

import argparse

from utils_GNN import *
from utils_dataset import *

# This dictionary determines the possible cases of study
# 4: "Mg + 1 + NN + NNN + 1 + delta history"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Results/models/dataset_NO_Dr_X_Mg_NN_NNN_delta_one/trans_Ising_inclscdn_False_trgtdiff_True_totsmpl_2000_hidC_32_hidE_4_hidN_4_NLay_4_Ndeltas_10_outC_32_sizes_4x4_5x5_6x6_lr_0.00025.pt')
    parser.add_argument("--params_path", type=str, default='params_test.json')
    args = parser.parse_args()

    # need to pass the path to the model to be loaded
    model_path = args.model_path
    print(f"model_path = {model_path}")

    # need to pass the path to the parameters to be loaded
    params_path = args.params_path
    print(f"params_path = {params_path}")

    # loading up the parameters inside params.json
    with open(params_path,'r') as jsonf:
        parameters = json.load(jsonf)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # some regex fetching
    num_layers = int(search(r'(?<=_NLay_)\d+',model_path).group())
    hidden_channels = int(search(r'(?<=_hidC_)\d+',model_path).group())  # Hidden node feature dimension in which message passing happens
    hidden_edges = int(search(r'(?<=_hidE_)\d+',model_path).group())
    hidden_nodes = int(search(r'(?<=_hidN_)\d+',model_path).group())
    num_deltas = int(search(r'(?<=_Ndeltas_)\d+',model_path).group())
    out_channels = int(search(r'(?<=_outC_)\d+',model_path).group()) # Dimension of output per each node

    print(f"num_layers = {num_layers}, hidden_channels = {hidden_channels}, hidden_edges = {hidden_edges}, num_deltas = {num_deltas}, out_channels = {out_channels}")

    # Doing the testing for all the sizes considered 
    LLs = ['4x4','5x5','6x6','7x6','7x7','8x7','8x8','9x8','9x9']
    
    test_samples = parameters["Physical_hyperparameters"]["total_samples"] # 200
    incl_scnd = parameters["Physical_hyperparameters"]["incl_scnd"] # whether the relative/full distances to second-nearest neighbors is included in the target set
    trgt_diff = parameters["Physical_hyperparameters"]["trgt_diff"] # whether the GNN is trained over the relative distances or the full distances between neighbors 
    save_to_one_shots = parameters["save_to_one_shots"]
    save_variances = parameters["save_variances"]
    datasets = ['test'] # to get the largest size
    realizations = [test_samples] #np.random.randint(0,high=total_samples,size=test_samples) # number of disorder realizations per training, validation, and test set PER system size
    print(f"realizations = {realizations}")
    print_freq = 5
    var_or_delta = 'var_NN' # 'var', 'var_NN' or 'delta'
    meas_basis = "Z"

    data_folder =  parameters["folder_datasets"] # path to datasets

    torch.manual_seed(int(time())) # int(time())
    np.random.seed(int(time())) # int(time())

    batch_size = 1 # batch size 1 for testing

    # generating the datasets
    for Ls in LLs:
        _, _, test_loader = load_datasets_mag_NN_NNN_δ(realizations, 
                                                       [Ls], 
                                                        num_deltas, 
                                                        incl_scnd = incl_scnd, 
                                                        trgt_diff = trgt_diff, 
                                                        meas_basis = meas_basis,
                                                        data_folder = data_folder, 
                                                        datasets = datasets,
                                                        batch_sizes = [batch_size]
                                                        )


        merged_histogram = merge_histograms(test_loader.dataset)
        # sys.exit()

        in_channels_node = num_deltas # Number of input features that nodes have (would be the time length if it were time-dependent)
        in_channels_edge = in_channels_node # Number of input features that edges have
        
        model = NodeEdgePNA(in_channels_node, 
                            in_channels_edge, 
                            out_channels, 
                            hidden_channels,
                            merged_histogram, 
                            num_layers = num_layers,
                            hidden_edges = hidden_edges,
                            hidden_nodes = hidden_nodes)
                
        # some metrics recorded in dicts
        validation_loss = {'edge_level': []}
        validation_loss_mean = {'edge_level': []}

        validation_r2 = {'edge_level': []}
        validation_r2_mean = {'edge_level': []}

        validation_r2 = {'edge_level': []}
        validation_mae = {'edge_level': []}
        validation_mape = {'edge_level': []}

        NN_tot_std = []
        try:
            print(f"Loading the model parameters saved in {model_path}!")
            model.load_state_dict(torch.load(model_path))
        except Exception as err:
            raise FileNotFoundError("The model parameters passed in was not found and couldn't be loaded: {}!".format(err))
                                                                            
        model.to(device)

    
        run_name = "testing_trans_Ising_hidC_{:d}_hidE_{:d}_hidN_{:d}_NLay_{:d}_Ndeltas_{:d}_outC_{:d}".format(hidden_channels,hidden_edges,hidden_nodes,num_layers,num_deltas,out_channels)
        
        preds_vs_targets = []
        model.eval()  # Set model to evaluating mode
        print(f"len = {len(test_loader.dataset)}")
        Lx, Ly = int(Ls.split('x')[0]), int(Ls.split('x')[1])
        index_prds_trgts = 2*(2*Lx*Ly-Lx-Ly) if not incl_scnd else 2*(2*Lx*Ly-Lx-Ly) + 2*4*((Lx-1)*(Ly-1))
        with torch.no_grad():
            for graphs in test_loader: 
                graphs = graphs.to(device)
                edge_predictions = model(graphs)
                preds_vs_targets = preds_vs_targets+list(zip(edge_predictions[:index_prds_trgts],graphs.edge_labels[:index_prds_trgts]))

                edge_r2_metric = metrics.r2_score(graphs.edge_labels.cpu().numpy()[:index_prds_trgts], edge_predictions.cpu().numpy()[:index_prds_trgts])
                edge_mae_metric = metrics.mean_absolute_error(graphs.edge_labels.cpu().numpy()[:index_prds_trgts], edge_predictions.cpu().numpy()[:index_prds_trgts])
                edge_mape_metric = metrics.median_absolute_error(graphs.edge_labels.cpu().numpy()[:index_prds_trgts], edge_predictions.cpu().numpy()[:index_prds_trgts])
                if num_deltas==1:
                    if var_or_delta=='delta':
                        validation_r2['edge_level'].append((edge_r2_metric,graphs.x_labels[0])) # h field is constant for all nodes
                        validation_mae['edge_level'].append((edge_mae_metric,graphs.x_labels[0]))
                        validation_mape['edge_level'].append((edge_mape_metric,graphs.x_labels[0]))
                    elif var_or_delta=='var':
                        data_to_get_var = np.delete(graphs.edge_labels.cpu().numpy(),np.where(graphs.edge_labels.cpu().numpy()==0.0)[0])
                        validation_r2['edge_level'].append((edge_r2_metric,np.std(data_to_get_var))) # h field is constant for all nodes
                        validation_mae['edge_level'].append((edge_mae_metric,np.std(data_to_get_var)))
                        validation_mape['edge_level'].append((edge_mape_metric,np.std(data_to_get_var)))
                    elif var_or_delta=='var_NN':
                        data_to_get_var = np.delete(graphs.edge_attr.cpu().numpy(),np.where(graphs.edge_attr.cpu().numpy()==0.0)[0])
                        NN_tot_std.append(np.std(np.abs(data_to_get_var[:index_prds_trgts])))
                        validation_r2['edge_level'].append((edge_r2_metric,np.std(data_to_get_var))) # h field is constant for all nodes
                        validation_mae['edge_level'].append((edge_mae_metric,np.std(data_to_get_var)))
                        validation_mape['edge_level'].append((edge_mape_metric,np.std(data_to_get_var)))
                elif num_deltas>1:
                    if var_or_delta=='var_NN' and save_variances:
                        data_to_get_var = np.delete(graphs.edge_attr.cpu().numpy(),np.where(graphs.edge_attr.cpu().numpy()==0.0)[0])
                        NN_tot_std.append(np.std(np.abs(data_to_get_var[:index_prds_trgts])))
                        validation_r2['edge_level'].append((edge_r2_metric,np.std(data_to_get_var))) # h field is constant for all nodes
                        validation_mae['edge_level'].append((edge_mae_metric,np.std(data_to_get_var)))
                        validation_mape['edge_level'].append((edge_mape_metric,np.std(data_to_get_var)))
                    else:
                        validation_r2['edge_level'].append(edge_r2_metric) # h field is constant for all nodes
                        validation_mae['edge_level'].append(edge_mae_metric)
                        validation_mape['edge_level'].append(edge_mape_metric)

            print(f"a = {len(validation_r2['edge_level'])}, c = {len(preds_vs_targets)}")

        print(f"len trgt vs preds {len(preds_vs_targets)}")
        for key in validation_r2.keys():
            if num_deltas==1:
                validation_r2_mean[key].append(np.mean(list(map(lambda x: x[0],validation_r2[key]))))
            elif num_deltas>1:
                validation_r2_mean[key].append(np.mean(validation_r2[key]))

        # Plot the losses and other metrics

        path_to_fig,_ = split_string_around_substring(model_path,'_lr_')

        fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(10,8))
        ax.grid()
        fig.subplots_adjust(hspace=0.1)
        
        # targets
        trgts = list(map(lambda x: x[1].item(), preds_vs_targets))
        # predictions
        prds = list(map(lambda x: x[0].item(), preds_vs_targets))

        ax.scatter(trgts[:], prds[:])
        ax.set_xlabel('Targets',fontsize=20)
        ax.set_ylabel("Predictions",fontsize=20)
        ax.tick_params(axis='both',which='major',labelsize=18)
        ax.set_title("Targets vs predictions",fontsize=20)

        # Plot the R^2 across the training
        fig2, ax2 = plt.subplots(nrows=1)
        fig2.subplots_adjust(hspace=0.1)

        if num_deltas==1:
            color_list=list(map(lambda x: x[1],validation_r2['edge_level']))
            cc = ax2.scatter(
                np.arange(len(test_loader)),
                list(map(lambda x: x[0],validation_r2['edge_level'])), 
                c=color_list,
                cmap='coolwarm')
            cbar = fig2.colorbar(cc,ax=ax2,label=r'$\delta$')
        elif num_deltas>1:
            ax2.scatter(np.arange(len(test_loader)),validation_r2['edge_level'])
            
        ax2.set_xlabel('graphs')
        ax2.set_title("$R^2$")
        
        # Plot the mae across the training
        fig3, ax3 = plt.subplots(nrows=1)
        fig3.subplots_adjust(hspace=0.1)

        if num_deltas==1:
            cc = ax3.scatter(
                np.arange(len(test_loader)),
                list(map(lambda x: x[0],validation_mae['edge_level'])), 
                c=list(map(lambda x: x[1],validation_mae['edge_level'])),
                cmap='coolwarm')
            cbar = fig3.colorbar(cc,ax=ax3,label=r'$\delta$')
        elif num_deltas>1:
            ax3.scatter(np.arange(len(test_loader)),validation_mae['edge_level'])
            
        ax3.set_xlabel('graphs')
        ax3.set_title("Mean absolute error")

        # Plot the mae across the training
        fig4, ax4 = plt.subplots(nrows=1)
        fig4.subplots_adjust(hspace=0.1)

        if num_deltas==1:
            cc = ax4.scatter(
                np.arange(len(test_loader)),
                list(map(lambda x: x[0],validation_mape['edge_level'])), 
                c=list(map(lambda x: x[1],validation_mape['edge_level'])),
                cmap='coolwarm')
            cbar = fig4.colorbar(cc,ax=ax4)
        elif num_deltas>1:
            ax4.scatter(np.arange(len(test_loader)),validation_mape['edge_level'])

        ax4.set_xlabel('graphs')
        ax4.set_title("Mean absolute percentage error")

        # storing values of metrics per size
        if save_to_one_shots:
            # training_sizes = str(search(r'(?<=_sizes_)\d+[x]\d+[_]\d+[x]\d+',model_path).group())
            training_sizes = str(search(r'(?<=_sizes_)\d+[x]\d+[_]\d+[x]\d+[_]\d+[x]\d+',model_path).group())
            # training_sizes = str(search(r'(?<=_sizes_)\d+[x]\d+?[_]\d+[x]\d+?[_]\d+[x]\d+?[_]\d+[x]\d+',model_path).group())
            print(f"training sizes = {training_sizes}")
            full_R2_metric = metrics.r2_score(trgts,prds)
            full_mae_metric = metrics.mean_absolute_error(trgts,prds)
            full_medae_metric = metrics.median_absolute_error(trgts,prds)
            full_std_metric = np.std(np.array(prds)-np.array(trgts))
            print(f"full_std_metric = {full_std_metric}")
            filename = path_to_fig + "/cluster_one_shots_" + run_name + '.h5'
            with h5py.File(filename,'a') as ff:
                try:
                    gg = ff.require_group(training_sizes)
                    gg = gg.require_group(Ls)
                    gg.require_dataset('R2',shape=(1,),data=full_R2_metric,dtype=float)
                    gg.require_dataset('MAE',shape=(1,),data=full_mae_metric,dtype=float)
                    gg.require_dataset('MEDAE',shape=(1,),data=full_medae_metric,dtype=float)
                    gg.require_dataset('STD',shape=(1,),data=full_std_metric,dtype=float)
                except Exception as err:
                    raise Exception("Error arisen: {}".format(err))
                
        if save_variances:
            training_sizes = str(search(r'(?<=_sizes_)\d+[x]\d+[_]\d+[x]\d+[_]\d+[x]\d+',model_path).group())
            print(f"training sizes = {training_sizes}")
            filename = path_to_fig + "/variances_" + run_name + '.h5'
            with h5py.File(filename,'a') as ff:
                try:
                    gg = ff.require_group(training_sizes)
                    gg = gg.require_group(Ls[0])
                    print("Variance = ", np.mean(NN_tot_std))
                    gg.require_dataset('NN_corr',shape=(1,),data=np.mean(NN_tot_std),dtype=float)
                except Exception as err:
                    raise Exception("Error arisen: {}".format(err))

    plt.show()