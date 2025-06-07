import argparse
import os

import files_to_dicts as ftd
import train
import test
import utils_dataset as uds
import utils_GNN as ugnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, default='params.json', help='The path to the json parameters file')
    parser.add_argument("--train_model", type=bool, default=False, help='Whether to train the GNN')
    parser.add_argument("--test_model", type=bool, default=False, help='Whether to test the GNN')
    args = parser.parse_args()

    path_to_param_file = args.param_file
    gnn_parms, phys_parms, gen_parms = uds.load_training_parameters(path_to_param_file)  

    # Preprocess the training data to pickle format (if needed)
    if uds.check_pickle_files_exist(phys_parms["Ls"], phys_parms["datasets"].keys(), gen_parms["folder_datasets"]):
        print("Pickle files for training already exist, skipping preprocessing...\n")
    else:
        print(f"Begining to preprocess the training data to pickle format...")
        ftd.preprocess_data(gen_parms["folder_datasets"], 
                            phys_parms["Ls"], 
                            phys_parms["datasets"], 
                            phys_parms["train_realizations"],
                            phys_parms["train_num_deltas"],
                            phys_parms["train_num_deltas"]) # can be different from num_δs if time dependent calculation, not in our case
    
    # Train the GNN
    if args.train_model:
        print(f"Begining to train the GNN...")
        train.train_GNN(gnn_parms, phys_parms, gen_parms)
    else:
        print(f"Skipping training the GNN...")


    # Test the GNN
    model_path = './Results/models/dataset_NO_Dr_X_Mg_NN_NNN_delta_one/trans_Ising_inclscdn_False_trgtdiff_True_totsmpl_2000_hidC_32_hidE_4_hidN_4_NLay_4_Ndeltas_10_outC_32_sizes_4x4_5x5_6x6_lr_0.00025.pt'
    
    test_parms = uds.load_test_parameters(path_to_param_file)
    print(f"Begining to test the GNN...")
    if uds.check_pickle_files_exist(test_parms["test_Ls"], test_parms["datasets"], test_parms["test_path"]):
        print("Pickle files for testing already exist, skipping preprocessing...\n")
    else:
        print(f"Begining to preprocess the testing data to pickle format...")
        ftd.preprocess_data(test_parms["test_path"], 
                            test_parms["test_Ls"], 
                            test_parms["datasets"], 
                            test_parms["test_realizations"],
                            test_parms["test_num_deltas"],
                            test_parms["test_num_deltas"],
                            test_parms["test_path"]) # can be different from num_δs if time dependent calculation, not in our case
     
    if args.test_model:
        print(f"Begining to test the GNN...")
        test.test_GNN(model_path, test_parms, phys_parms, gen_parms)
    else:
        print(f"Skipping testing the GNN...")


if __name__ == "__main__":
    main()