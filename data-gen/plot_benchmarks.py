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
        'init_state': params['physics']['init_state'],
        'init_linkdims': params['physics']['init_linkdims'],
        'deltas': params['deltas'],
        'folder': params['output']['folder'],
        'output_folder': get_output_path(params['output']['folder'], params['lattice']['alpha'])
    }

    bd_min = params['bond_dims']['start']
    bd_max = params['bond_dims']['stop']
    bd_step = params['bond_dims']['step']
    
    return plot_params, plot_params['output_folder'], bd_min, bd_max, bd_step

def main():
    parser = argparse.ArgumentParser(description='Plot error vs bond dimension for TFIM')
    parser.add_argument('--type', type=int, required=True,
                      help='Type of plot: 0: error_vs_bond_dim, 1: staggered_magnetization, 2: FM_and_AFM_error_vs_bond_dim, 3: time_vs_bond_dim_single_alpha')
    parser.add_argument('--params', type=str, default='benchmark_parameters.json',
                      help='Path to the JSON file containing benchmark parameters')
    parser.add_argument('--vs', type=str, default="max_trunc_err",
                      help='Variable to plot: error or max_trunc_err')
    args = parser.parse_args()

    if args.type == 0:
        # Load parameters from JSON
        plot_params, output_folder, bd_min, bd_max, bd_step = load_plot_parameters(args.params)
        nx, ny = plot_params['nx'], plot_params['ny']
        alpha = plot_params['alpha']
        R = plot_params['R']
        amp_R = plot_params['amp_R']
        C6 = plot_params['C6']
        deltas = plot_params['deltas']
        vs = args.vs 
        init_state = plot_params.get('init_state', 'FM')
        init_linkdims = plot_params.get('init_linkdims', 100)
        
        print(f"Plotting for: size = {nx}x{ny} with init_state = {init_state}, init_linkdims = {init_linkdims}")
        print(f"Other parameters: C6 = {C6}, alpha = {alpha}, R = {R}, amp_R = {amp_R}")
        print(f"Deltas: {deltas}")
        output_folder = os.path.join(output_folder, f"{nx}x{ny}")

        # Step 1: Plot error vs bond dimension
        pbu.draw_plots_error_vs_maxdim(nx, ny, deltas, amp_R, vs=vs, folder=output_folder, 
                                    physics_params={'C6': C6, 'alpha': alpha, 'R': R, 'amp_R': amp_R},
                                    init_state=init_state, init_linkdims=init_linkdims)
        
        # Step 2: Ask for optimal bond dimension
        print(f"\nWhat is the optimal bond dimension, from 1 to {bd_max} with step {bd_step} (enter below and press enter): ")
        optimal_bond_dim = int(input())
        while optimal_bond_dim < bd_min or optimal_bond_dim > bd_max or optimal_bond_dim % bd_step != 0:
            print(f"Invalid input. Please enter a number between {bd_min} and {bd_max} in steps of {bd_step} or 1.")
            optimal_bond_dim = int(input())

        # Step 3: Plot magnetization phase diagram
        pbu.plot_magnetization_phase_diagram(nx, ny, R, amp_R, alpha, output_folder, optimal_bond_dim, save_fig=True,
                                        physics_params={'C6': C6, 'alpha': alpha, 'R': R, 'amp_R': amp_R},
                                        init_state=init_state, init_linkdims=init_linkdims)
    elif args.type == 1:
        nx, ny = 9,9
        R = 1.0
        amp_R = 0.02    
        C6 = 1.0
        init_state = "FM"
        init_linkdims = 100
        optimal_bond_dim = 160
        folder = "./Benchmark-9x9/" # ./Benchmark-9x9/ ./Benchmark-normalized/

        pbu.plot_phase_diagram_all_alpha(nx, ny, R, amp_R, folder, optimal_bond_dim, save_fig=True,
                                       physics_params={'C6': C6, 'R': R, 'amp_R': amp_R},
                                       init_state=init_state, init_linkdims=init_linkdims)
        
    elif args.type == 2:
        nx, ny = 9,9
        R = 1.0
        amp_R = 0.02
        C6 = 1.0
        init_linkdims = 100
        folder = "./Benchmark-9x9/"
        
        print(f"Provide alpha for the plot (1, 2, 3, 6): ")
        alpha = int(input())
        
        if alpha == 1:
            deltas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        elif alpha == 2:
            deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
        elif alpha == 3:
            deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
        elif alpha == 6:
            deltas = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
        else:
            raise ValueError(f"Invalid alpha: {alpha}. Don't have data for this alpha in given folder: {folder}")

        pbu.plot_FM_and_AFM_error_vs_bond_dim(args.vs, nx, ny, folder, deltas=deltas, save_fig=True,
                                       physics_params={'C6': C6, 'alpha': alpha, 'R': R, 'amp_R': amp_R},
                                       init_linkdims=init_linkdims)
    elif args.type == 3:
        plot_params, output_folder, _, _, _ = load_plot_parameters(args.params)
        nx, ny = plot_params['nx'], plot_params['ny']
        R = plot_params['R']
        amp_R = plot_params['amp_R']
        C6 = plot_params['C6']
        init_state = plot_params.get('init_state', 'FM')
        init_linkdims = plot_params.get('init_linkdims', 100)
        alpha = plot_params['alpha']
        
        # Use output_folder which already includes alpha_* and add the size folder
        path_to_folder = os.path.join(output_folder, f"{nx}x{ny}")
        print(f"Plotting DMRG time vs bond dimension for alpha={alpha} in: {path_to_folder}")
        pbu.plot_time_vs_bond_dim(
            nx, ny, R, amp_R, alpha, path_to_folder,
            init_state=init_state, init_linkdims=init_linkdims, save_fig=True,
            physics_params={'C6': C6, 'alpha': alpha, 'R': R, 'amp_R': amp_R}
        )
    else:
        args.print_help()
        raise ValueError(f"Invalid plot type: {args.type}. Please choose 0, 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()


