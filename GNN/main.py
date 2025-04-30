import argparse
import os

import files_to_dicts as ftd
import train

import utils_dataset as uds
import utils_GNN as ugnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, default='params.json', help='The path to the json parameters file')
    args = parser.parse_args()

    path_to_param_file = args.param_file
    if not os.path.exists(path_to_param_file):
        raise FileNotFoundError(f"The file {path_to_param_file} does not exist.")
    else:
        gnn_parms, phys_parms, gen_parms = uds.load_training_parameters(path_to_param_file)  


    # Preprocess the training data to pickle format (if needed)
    if uds.check_pickle_files_exist(phys_parms, gen_parms):
        print("Pickle files already exist, skipping preprocessing...\n")
    else:
        print(f"Begining to preprocess the training data to pickle format...")
        ftd.preprocess_data(gen_parms["folder_datasets"], 
                            phys_parms["Ls"], 
                            phys_parms["datasets"], 
                            phys_parms["train_realizations"],
                            phys_parms["train_num_deltas"],
                            phys_parms["train_num_deltas"]) # can be different from num_δs if time dependent calculation, not in our case
    
    # Train the GNN
    print(f"Begining to train the GNN...")
    train.train_GNN(gnn_parms, phys_parms, gen_parms)


if __name__ == "__main__":
    main()