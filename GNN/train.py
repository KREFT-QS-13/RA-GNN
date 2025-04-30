"""
Filename: GNN_training.py
Description: 
    1. Reads off the JSON file `params.json`. 
    2. Script taking care of the GNN training to output the trained model in folder path specified by the parameter `folder_models`.
    3. Test and save the model. (Evaluation)
"""
import torch
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import datetime
import os
from time import time
from glob import glob
from re import search
from functools import partial
import h5py
import json
import pickle

import argparse

import utils_GNN as uggn
import utils_dataset as uds

# Only consider case #5 from the paper: \chi_{NN}, NNN edges and delta history (Omega history)
# Which is most likekly this one from original code: 4: "Mg + 1 + NN + NNN + 1 + delta history", # case #5


def train_GNN(gnn_parms: dict, phys_parms: dict, gen_parms: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} used for training the GNN.\n")

    # GNN hyperparameters
    num_layers = gnn_parms["num_layers"]
    hidden_channels = gnn_parms["hidden_channels"]  # Hidden node feature dimension in which message passing happens
    hidden_edges = gnn_parms["hidden_edges"]
    hidden_nodes = gnn_parms["hidden_nodes"]
    
    # Learning-rate hyperparameters
    learning_rate = gnn_parms["learning_rate"] # initial learning rate
    warmup_height = gnn_parms["warmup_height"] # maximum height of the learning rate during warmup
    warmup_ratio = gnn_parms["warmup_ratio"] # ratio of epochs dedicated to the first linear up-ramp
    minimum_lr_ratio = gnn_parms["minimum_lr_ratio"] # minimum value of the learning rate w.r.t initial learning rate reached at the end

    weight_decay = gnn_parms["weight_decay"] # Adam optimizer
    num_epochs = gnn_parms["num_epochs"]
    batch_sizes = gnn_parms["batch_sizes"] # batch sizes for training, validation and testing

    is_early_stopping = gnn_parms["is_early_stopping"]
    patience = 4

    # Training set hyperparameters
    Ls = phys_parms["Ls"]
    train_realizations = phys_parms["train_realizations"] # 2000
    train_datasets = phys_parms["datasets"] # proportions of the total number of samples dedicated to the training dataset (1st slot), validation dataset (2nd slot) and test dataset (3rd slot).
    
    samples = {key: int(train_realizations*value) for key, value in train_datasets.items()}
    training_samples = samples["training"] # should 80 %
    validation_samples = samples["validation"] # should be 10 %
    test_samples = samples["test"] # should be 10 %

    num_samples = [training_samples, validation_samples, test_samples] # number of disorder realizations per training, validation, and test set PER system size
    num_deltas = phys_parms["train_num_deltas"] # 10

    print_freq = 5
    load_former_models = phys_parms["load_former_models"] # whether to load previously saved model if compatible
    incl_scnd = phys_parms["incl_scnd"] # whether the relative/full distances to second-nearest neighbors is included in the target set
    trgt_diff = phys_parms["trgt_diff"] # whether the GNN is trained over the relative distances or the full distances between neighbors 

    meas_basis = "Z"

    # Data folder to get DMRG data from
    data_folder = gen_parms["folder_datasets"] # path to datasets
    # Data folder in which saving model parameters
    data_folder_mp = gen_parms["folder_models"]

    print(f"num_samples = {num_samples}")

    torch.manual_seed(int(time())) # int(time())
    np.random.seed(int(time())) # int(time())

    # generating the datasets depending on the case of study
    train_loader, validation_loader, test_loader = None, None, None

    # Load the datasets
    before_loading = datetime.datetime.now()
    train_loader, validation_loader, test_loader = uds.load_datasets_mag_NN_NNN_δ(
        num_samples, 
        Ls, 
        num_deltas, 
        incl_scnd = incl_scnd, 
        trgt_diff = trgt_diff,
        meas_basis = meas_basis,
        data_folder = data_folder, 
        batch_sizes = batch_sizes)

    after_loading = datetime.datetime.now()
    print(f"Time taken to load the datasets: {after_loading - before_loading}")
    print(f"train_loader = {len(train_loader.dataset)}")
    
    # bining the 'in-node' degrees
    # calculating the degree.
    # This calculates the "in-degree" of nodes, which is the number of edges coming into a node. 
    # This is used when considering the perspective of a destination node receiving information from its neighbors (normalization).
    merged_histogram = uggn.merge_histograms(train_loader.dataset)
    # sys.exit()

    in_channels_node = num_deltas # Number of input features that nodes have (would be the time length if it were time-dependent)
    in_channels_edge = in_channels_node # Number of input features that edges have
    out_channels = gnn_parms["out_channels"] # Dimension of output per each node
    
    model = uggn.NodeEdgePNA(in_channels_node, 
                        in_channels_edge, 
                        out_channels, 
                        hidden_channels,
                        merged_histogram, 
                        num_layers = num_layers,
                        hidden_edges = hidden_edges,
                        hidden_nodes = hidden_nodes)

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of variational parameters: {num_params}")
    
    criterion = torch.nn.MSELoss() # mean-square loss

    # Train the model
    epoch_time = []

    # some metrics recorded in dicts
    training_loss = {'edge_level': []}
    training_loss_mean = {'edge_level': []}
    training_loss_std = {'edge_level': []}

    validation_loss = {'edge_level': []}
    validation_loss_mean = {'edge_level': []}
    validation_loss_std = {'edge_level': []}

    training_r2 = {'edge_level': []}
    training_r2_mean = {'edge_level': []}

    training_mae = {'edge_level': []}
    training_mape = {'edge_level': []}

    validation_r2 = {'edge_level': []}
    validation_r2_mean = {'edge_level': []}

    validation_mae = {'edge_level': []}
    validation_mape = {'edge_level': []}

    sizes = '_'.join(list(map(lambda x: str(x),Ls)))
    meas_basis_str = '_'+meas_basis if meas_basis != "Z" else ""
    run_name = "trans_Ising_inclscdn_{}_trgtdiff_{}_totsmpl_{:d}_hidC_{:d}_hidE_{:d}_hidN_{:d}_NLay_{:d}_Ndeltas_{:d}_outC_{:d}_sizes_{}{}".format(incl_scnd,trgt_diff,train_realizations,hidden_channels,hidden_edges,hidden_nodes,num_layers,num_deltas,out_channels,sizes,meas_basis_str)
    model_folder = data_folder_mp+os.path.basename(data_folder)+'/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    elif load_former_models: # check if model parameters have been saved
        list_files_pt = [ff for ff in glob(model_folder+'/'+run_name+'_lr_'+'*') if ff.endswith('.pt')]
        # pick out the latest saved model
        if len(list_files_pt)>0:
            model_to_load = sorted(list_files_pt, key=lambda x: os.path.getmtime(x))[-1]
            print(f"Loading the model parameters saved in {model_to_load}!")
            model.load_state_dict(torch.load(model_to_load))
            learning_rate = float(search(r'(?<=_lr_)\d*\.\d+',model_to_load).group())
        else:
            print(f"Did not find any models for that particular system configuration...")

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    custom_scheduler = partial(uggn.custom_scheduler,
                                lr=learning_rate,
                                wrm_rt=warmup_ratio,
                                wrm_hght=warmup_height,
                                min_lr_rt=minimum_lr_ratio,
                                n_epchs=num_epochs)
    
    lambda_scheduler = uggn.ArrayLRScheduler(optimizer, lr_array=[custom_scheduler(x) for x in range(num_epochs)])
                                                                        
    # initialize the early_stopping object
    path_to_save = model_folder + run_name
    early_stopping = uggn.EarlyStopping(patience = patience)
    model.to(device)

    st = time()
    for epoch in range(num_epochs):
        
        model.train()  # Set model to training mode
        for i, graphs in enumerate(train_loader):

            graphs = graphs.to(device)

            # Forward pass
            edge_predictions = model(graphs)
            edge_loss = criterion(edge_predictions, graphs.edge_labels)

            # Backward and optimize
            optimizer.zero_grad()
            edge_loss.backward()
            optimizer.step()

            training_loss['edge_level'].append(edge_loss.item())
            edge_r2_metric = metrics.r2_score(graphs.edge_labels.cpu().detach().numpy(), edge_predictions.cpu().detach().numpy())
            training_r2['edge_level'].append(edge_r2_metric)
            edge_mae_metric = metrics.mean_absolute_error(graphs.edge_labels.cpu().detach().numpy(), edge_predictions.cpu().detach().numpy())
            training_mae['edge_level'].append(edge_mae_metric)
            edge_mape_metric = metrics.mean_absolute_percentage_error(graphs.edge_labels.cpu().detach().numpy(), edge_predictions.cpu().detach().numpy())
            training_mape['edge_level'].append(edge_mape_metric)

        for key in training_loss.keys():
            training_loss_mean[key].append(np.mean(training_loss[key]))
            training_loss_std[key].append(np.std(training_loss[key]))
        for key in training_r2.keys():
            training_r2_mean[key].append(np.mean(training_r2[key]))
        
        if epoch % print_freq == 0:
            print("Epoch: ", epoch, "Loss: ", edge_loss.item(), flush=True)

        model.eval()  # Set model to evaluating mode
        with torch.no_grad():
            for graphs in validation_loader: 
                graphs = graphs.to(device)

                edge_predictions = model(graphs)
                val_loss_edge = criterion(edge_predictions, graphs.edge_labels)

                validation_loss['edge_level'].append(val_loss_edge.item())
                edge_r2_metric = metrics.r2_score(graphs.edge_labels.cpu().numpy(), edge_predictions.cpu().numpy())
                validation_r2['edge_level'].append(edge_r2_metric)
                edge_mae_metric = metrics.mean_absolute_error(graphs.edge_labels.cpu().numpy(), edge_predictions.cpu().numpy())
                validation_mae['edge_level'].append(edge_mae_metric)
                edge_mape_metric = metrics.mean_absolute_percentage_error(graphs.edge_labels.cpu().numpy(), edge_predictions.cpu().numpy())
                validation_mape['edge_level'].append(edge_mape_metric)

        for key in validation_loss.keys():
            validation_loss_mean[key].append(np.mean(validation_loss[key]))
            validation_loss_std[key].append(np.std(validation_loss[key]))
        for key in validation_r2.keys():
            validation_r2_mean[key].append(np.mean(validation_r2[key]))

        if epoch % print_freq == 0:
            print("Epoch: ", epoch, "Val loss: ", val_loss_edge.item(), "t_elapsed/epoch: ", (time() - st) / print_freq, flush=True)
            epoch_time.append((time() - st) / print_freq)
            st = time()
        
        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        filename_to_save = path_to_save+'_lr_{:.5f}'.format(lambda_scheduler.get_last_lr()[0]) + '.pt'
        if is_early_stopping is True:
            early_stopping(validation_loss_mean['edge_level'][0], model, path=filename_to_save)
            
            if early_stopping.early_stop:
                print("Early stopping", flush = True)
                break
        
        if epoch == num_epochs - 1:
            # Save the model checkpoint
            torch.save(model.state_dict(), filename_to_save)

        lambda_scheduler.step() # here for ExpLR scheduler
        if epoch%print_freq == 0:
            print(f"Learning rate = {lambda_scheduler.get_last_lr()[0]} at epoch {epoch}")
        #schedulerMSLR.step # here for milestones scheduler

    # Load either the final model or early-stopped model
    model.load_state_dict(torch.load(filename_to_save))

    preds_vs_targets = []
    model.eval()  # Set model to evaluating mode
    with torch.no_grad():
        for graphs in test_loader:
            graphs = graphs.to(device)

            edge_predictions = model(graphs)
            val_loss_edge = criterion(edge_predictions, graphs.edge_labels)
            # print(f"edge_predictions = {edge_predictions} and {edge_predictions.shape} and labels = {graphs.edge_labels} and {graphs.edge_labels.shape}")
            preds_vs_targets=preds_vs_targets+list(zip(edge_predictions,graphs.edge_labels))
    
    preds = np.array(list(map(lambda x: x[0].item(),preds_vs_targets)))
    targets = np.array(list(map(lambda x: x[1].item(),preds_vs_targets)))

    plotting = datetime.datetime.now()

    # Create directory for saving plots and data
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    plots_dir = os.path.join("Figs", "train", f"model_{today_date}")
    os.makedirs(plots_dir, exist_ok=True)

    # Dictionary to store all plot data
    plot_data = {
        'training_loss': training_loss_mean['edge_level'],
        'validation_loss': validation_loss_mean['edge_level'],
        'training_r2': training_r2['edge_level'],
        'validation_r2': validation_r2['edge_level'],
        'training_mae': training_mae['edge_level'],
        'validation_mae': validation_mae['edge_level'],
        'training_mape': training_mape['edge_level'],
        'validation_mape': validation_mape['edge_level'],
        'predictions': preds,
        'targets': targets,
        'epochs': list(range(len(training_loss_mean['edge_level']))),
        'model_params': {
            'num_layers': num_layers,
            'hidden_channels': hidden_channels,
            'hidden_edges': hidden_edges,
            'hidden_nodes': hidden_nodes,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_sizes': batch_sizes
        }
    }

    # Plot the losses and other metrics
    fig, ax = plt.subplots(nrows=1, sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    ax.plot(training_loss_mean['edge_level'], label="edges' training loss")
    ax.plot(validation_loss_mean['edge_level'], label="edges' validation loss")
    ax.set_xlabel('epochs')
    ax.set_yscale("log")
    ax.set_title("Losses plots")
    ax.legend()
    plt.savefig(os.path.join(plots_dir, 'losses.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the R^2 across the training
    fig2, ax2 = plt.subplots(nrows=1)
    fig2.subplots_adjust(hspace=0.1)

    ax2.plot(training_r2['edge_level'], label="edges' training loss")
    ax2.plot(validation_r2['edge_level'], label="edges' validation loss")
    ax2.set_xlabel('epochs')
    ax2.set_title("$R^2$ plots")
    ax2.legend()
    plt.savefig(os.path.join(plots_dir, 'r2_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # plot predictions vs targets
    fig3, ax3 = plt.subplots()

    ax3.scatter(targets, preds, c='g')
    ax3.set_title(r"Edge prediction values")
    ax3.set_xlabel(r'Target values')
    ax3.set_ylabel(r'Predicted values')
    ax3.plot(np.arange(0.4, 0.8, 200), np.arange(0.4, 0.8, 200), ls='-', c='k', lw=3)
    plt.savefig(os.path.join(plots_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the mae across the training
    fig_mae, ax_mae = plt.subplots(nrows=1)
    fig_mae.subplots_adjust(hspace=0.1)

    ax_mae.plot(training_mae['edge_level'], label="edges' mae")
    ax_mae.plot(validation_mae['edge_level'], label="edges' mae")
    ax_mae.set_xlabel('epochs')
    ax_mae.set_title("Mean absolute error")
    ax_mae.legend()
    plt.savefig(os.path.join(plots_dir, 'mae_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the mape across the training
    fig_mape, ax_mape = plt.subplots(nrows=1)
    fig_mape.subplots_adjust(hspace=0.1)

    ax_mape.plot(training_mape['edge_level'], label="edges' mape")
    ax_mape.plot(validation_mape['edge_level'], label="edges' mape")
    ax_mape.set_xlabel('epochs')
    ax_mape.set_title("Mean absolute percentage error")
    ax_mape.legend()
    plt.savefig(os.path.join(plots_dir, 'mape_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save all plot data to a single file
    with open(os.path.join(plots_dir, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(plot_data, f)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # saving the data into files
    basename_data = path_to_save + '/' + "totsmpls_{:d}_trsmpls_{:d}_valsmpls_{:d}_btchs_{}_sizes_{}_numepochs_{:d}".format(
        train_realizations,
        training_samples,
        validation_samples,
        'x'.join(list(map(lambda x: str(x),batch_sizes))),
        sizes,
        num_epochs
        )
    
    iteration = 0
    list_files_h5py = [ff for ff in glob(basename_data+'*') if ff.endswith('.h5py')]
    latest_data_runs = sorted(list_files_h5py, key=lambda x: os.path.getmtime(x))
    print(f"latest_data_runs = {latest_data_runs} and list_files_h5py = {list_files_h5py}")
    if len(latest_data_runs)>0:
        iteration = int(search(r'(?<=_iter_)\d+',latest_data_runs[-1]).group())+1
    
    basename_data += "_iter_{:d}.h5py".format(iteration)

    with h5py.File(basename_data,'w') as ff:
        dataset_edge_level_tr_loss = ff.create_dataset("edge_level_tr_loss",data=training_loss_mean["edge_level"])
        dataset_edge_level_val_loss = ff.create_dataset("edge_level_val_loss",data=validation_loss_mean["edge_level"])
        dataset_tr_r2 = ff.create_dataset("tr_r2", data=training_r2['edge_level'])
        dataset_val_r2 = ff.create_dataset("val_r2", data=validation_r2['edge_level'])

        ff.attrs['Model'] = '2D Transverse Ising' 
        ff.attrs['Path'] = data_folder

    ff.close()

    print(f"All plots and data have been saved to: {plots_dir}")
