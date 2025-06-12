import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from typing import Dict, Any, Tuple

import plot_benchmarks_utiliites as pbu

def get_output_path(base_path: str, alpha: int) -> str:
    """
    Create and return the output path with alpha folder.
    
    Args:
        base_path: Base output path from JSON
        alpha: Alpha value for the folder name
        
    Returns:
        Full path including alpha folder
    """
    alpha_path = os.path.join(base_path, f"alpha_{alpha}")
    os.makedirs(alpha_path, exist_ok=True)
    return alpha_path

def load_plot_parameters(json_file: str) -> Tuple[Dict[str, Any], str]:
    """
    Load parameters from the benchmark parameters JSON file.
    
    Args:
        json_file (str): Path to the JSON file
        
    Returns:
        Tuple containing:
        - Dict with parameters
        - Path to the output folder
    """
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    # Extract relevant parameters
    plot_params = {
        'nx': params['lattice']['nx'],
        'ny': params['lattice']['ny'],
        'amp_R': params['lattice']['amp_R'],
        'alpha': params['lattice']['alpha'],
        'R': params['lattice']['R'],
        'C6': params['physics']['C6'],
        'deltas': params['deltas'],
        'output_folder': get_output_path(params['output']['folder'], params['lattice']['alpha'])
    }

    bd_min = params['bond_dims']['start']
    bd_max = params['bond_dims']['stop']
    bd_step = params['bond_dims']['step']
    
    return plot_params, plot_params['output_folder'], bd_min, bd_max, bd_step

def main():
    parser = argparse.ArgumentParser(description='Plot error vs bond dimension for TFIM')
    parser.add_argument('--params', type=str, default='benchmark_parameters.json',
                      help='Path to the JSON file containing benchmark parameters')
    parser.add_argument('--vs', type=str, default="max_trunc_err",
                      help='Variable to plot: error or max_trunc_err')
    args = parser.parse_args()

    # Load parameters from JSON
    plot_params, output_folder, bd_min, bd_max, bd_step = load_plot_parameters(args.params)
    nx, ny = plot_params['nx'], plot_params['ny']
    alpha = plot_params['alpha']
    R = plot_params['R']
    amp_R = plot_params['amp_R']
    C6 = plot_params['C6']
    deltas = plot_params['deltas']
    vs = args.vs 
    init_state = plot_params.get('init_state', 'FM')  # Get init_state from parameters, default to 'FM'

    output_folder = os.path.join(output_folder, f"{nx}x{ny}")

    # Step 1: Plot error vs bond dimension
    print(f"Plotting for: size = {nx}x{ny}, deltas = {deltas}, init_state = {init_state}")
    pbu.draw_plots_error_vs_maxdim(nx, ny, deltas, amp_R, vs=vs, folder=output_folder, 
                                  physics_params={'C6': C6, 'alpha': alpha, 'R': R, 'amp_R': amp_R},
                                  init_state=init_state)
    
    # Step 2: Ask for optimal bond dimension
    print(f"\nWhat is the optimal bond dimension, from 1 to {bd_max} with step {bd_step} (enter below and press enter): ")
    optimal_bond_dim = int(input())
    while optimal_bond_dim < bd_min or optimal_bond_dim > bd_max or optimal_bond_dim % bd_step != 0:
        print(f"Invalid input. Please enter a number between {bd_min} and {bd_max} in steps of {bd_step} or 1.")
        optimal_bond_dim = int(input())

    # Step 3: Plot magnetization phase diagram
    pbu.plot_magnetization_phase_diagram(nx, ny, output_folder, optimal_bond_dim, save_fig=True,
                                       physics_params={'C6': C6, 'alpha': alpha, 'R': R, 'amp_R': amp_R},
                                       init_state=init_state)
    
if __name__ == "__main__":
    main()



