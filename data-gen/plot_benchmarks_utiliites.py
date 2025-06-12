import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.colors import Normalize


from typing import Dict, List, TypeVar, Tuple

def load_nested_dict_int_to_pairs(filename: str) -> Tuple[Dict[int, Dict[float, float]], Dict[int, Dict[float, List[float]]]]:
    """
    Load and reconstruct nested dictionaries from a NumPy file where:
    - Outer keys are integers (bond dimensions)
    - Inner keys are floats (deltas)
    - Values are either floats (for staggered magnetization) or lists of floats (for magnetization per site)
    
    Args:
        filename (str): Path to the .npz file
        
    Returns:
        Tuple containing:
        - Dict[int, Dict[float, float]]: For staggered magnetization
        - Dict[int, Dict[float, List[float]]]: For magnetization per site
    """
    try:
        # Load the data
        data = np.load(filename)
        
        # Get the keys and values
        outer_keys = data["outer_keys"]
        inner_keys = data["inner_keys"]
        values = data["values"]
        
        # Reconstruct the dictionary
        reconstructed_dict = {}
        for i, outer_key in enumerate(outer_keys):
            inner_dict = {}
            if len(values.shape) == 2:  # For staggered magnetization (2D array)
                for j, inner_key in enumerate(inner_keys):
                    inner_dict[float(inner_key)] = float(values[i, j])
            else:  # For magnetization per site (3D array)
                for j, inner_key in enumerate(inner_keys):
                    inner_dict[float(inner_key)] = values[i][:, j].tolist()
            reconstructed_dict[int(outer_key)] = inner_dict

        return reconstructed_dict
        
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except KeyError as e:
        raise KeyError(f"Expected keys 'outer_keys', 'inner_keys', and 'values' in the .npz file. Missing: {e}")
    except Exception as e:
        raise Exception(f"Error loading the file: {str(e)}")

def load_data_to_plot(path_to_folder: str, nx: int, ny: int, optimal_bond_dim: int) -> Tuple[List[float], List[float]]:
    """
    Load and reconstruct data for plotting from a NumPy file.
    
    Args:
        path_to_folder (str): Path to the folder containing the data
        nx, ny (int): Dimensions of the lattice
        optimal_bond_dim (int): The bond dimension to use for the plot
        
    Returns:
        Tuple[List[float], List[float]]: A tuple containing:
            - List of deltas
            - List of staggered magnetizations
    """
    # Note: path_to_folder now includes the alpha folder
    filename = os.path.join(path_to_folder, f"staggered_magnetization.npz")

    # Load the nested dictionary
    data_dict = load_nested_dict_int_to_pairs(filename)
    
    try:
        # Get data for the optimal bond dimension
        inner_dict = data_dict[optimal_bond_dim]
        
        # Extract deltas and magnetizations
        deltas = sorted(inner_dict.keys())
        magnetizations = [inner_dict[d] for d in deltas]
    except KeyError:
        raise ValueError(f"Bond dimension {optimal_bond_dim} not found in data")

    return deltas, magnetizations


def calulate_neel_magnetization(mag_per_site: np.ndarray, L:int=5) -> float:
    """
    Calculate the Neel magnetization from the magnetization per site.
    """
    stagger = np.fromfunction(lambda x, y: (-1)**(x + y), (L, L))
    magnetization = np.abs((mag_per_site.reshape(L,L) * stagger).mean())

    return magnetization

def add_physics_textbox(ax, physics_params):
    """
    Add a text box with physics parameters to the plot.
    
    Args:
        ax: matplotlib axis
        physics_params: dict containing physics parameters (C6, alpha, R, amp_R)
    """
    print(physics_params.keys())
    textstr = ' | '.join((
        r'$C_6 = %.2f$' % (physics_params['C6'],),
        r'$\alpha = %d$' % (physics_params['alpha'],),
        r'$R_0\pm\delta R= %.2f\pm%.2f$' % (physics_params['R'], physics_params['amp_R']),
    ))
    
    # Place text box in upper center
    props = dict(boxstyle='round', facecolor='white', alpha=0.2, edgecolor='black', linewidth=1)
    ax.text(0.5, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center',
            bbox=props)

def plot_magnetization_phase_diagram(nx, ny, path_to_folder, optimal_bond_dim, save_fig=False, physics_params=None, init_state:str="FM"):
    """
    Plot the heat map phase diagram.
    """
    # Note: path_to_folder now includes the alpha folder
    filename = os.path.join(path_to_folder, f"staggered_magnetization_init={init_state}.npz")
    
    # Load the nested dictionary
    data_dict = load_nested_dict_int_to_pairs(filename)
    
    deltas, staggered_magnetization = load_data_to_plot(path_to_folder, nx, ny, optimal_bond_dim)
    
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(deltas, staggered_magnetization, label=f"Bond dim. = {optimal_bond_dim} with quick start")
    
    if physics_params:
        add_physics_textbox(ax, physics_params)
    
    ax.legend(fontsize=12)
    ax.set_title(f'Staggered magnetization of {nx}x{ny} square lattice (init={init_state})', fontsize=16)
    ax.set_ylabel('Staggered Magnetization', fontsize=13)
    ax.set_xlabel(r'Magnetic field coefficient $\Omega$', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    print(deltas)
    ax.set_xticks(deltas)
    plt.tight_layout()

    if save_fig:
        # Create imgs directory if it doesn't exist
        imgs_dir = os.path.join(path_to_folder, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)
        plt.savefig(os.path.join(imgs_dir, f"magnetization_phase_diagram_{nx}x{ny}_init={init_state}.png"))
    plt.show()

def plot_magnetization_heatmaps(nx, ny, path_to_folder, optimal_bond_dim, lattice_unit=10, color_map="inferno", physics_params=None):
    """
    Create heatmaps of magnetization for different bond dimensions and deltas.
    """
    # Extract data from files
    deltas, staggered_magnetization = load_data_to_plot(path_to_folder, nx, ny, optimal_bond_dim)
    
    print(staggered_magnetization)

    # Create a single heatmap for specified bond dimension
    fig, ax = plt.subplots(figsize=(7, 1.5))  # Single plot with smaller height
    
    # Get data for the bond dimension (using first bond dimension for example)
    data = staggered_magnetization  # Using first bond dimension
    data = data.reshape(1, -1)  # Reshape to 2D for imshow
    
    # Create heatmap
    im = ax.imshow(data, 
                    aspect='auto', 
                    origin='lower',
                    cmap=color_map, 
                    vmin=0.0,
                    vmax=1.0,
                    extent=[deltas[0], deltas[-1], 0, 1])
    
    if physics_params:
        add_physics_textbox(ax, physics_params)
    
    # Customize axes
    ax.set_title(f'Bond dim = 150 with quick start', fontsize=16)
    ax.set_ylabel('Nominal dist.', fontsize=13)
    ax.yaxis.set_visible(False)
    ax.set_xlabel(r'Magnetic field coefficient $\Omega$', fontsize=13)
    ax.set_xticks(np.linspace(deltas[0], deltas[-1], 5))
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add colorbar with custom width
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, label='Magnetization')
    # cbar = fig.colorbar(im,ax=ax, label='Magnetization')
   
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for wider colorbar
    return fig

def draw_plots_error_vs_maxdim(nx:int, ny:int, deltas:list[float], amp_R:float=0.0, filename:str="plot_err_vs_maxdim", 
                              vs:str="max_trunc_err", folder="Experiment_1", physics_params=None, init_state:str="FM"):
    # Create figure with two subplots
    plt.style.use('tableau-colorblind10') # 'seaborn-v0_8-darkgrid' , 'tableau-colorblind10'

    fig = plt.figure(figsize=(12, 12))  # Increased height to accommodate both plots
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.025)  # 4:1 height ratio, small gap
    ax1 = fig.add_subplot(gs[0])  # Main plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Small plot below, sharing x-axis
    
    # ax1.style.use('ggplot') # 
    # ax2.style.use('ggplot') # 'seaborn-v0_8-darkgrid' , 'tableau-colorblind10'
    ax2.set_xlabel("Bond dimension of DMRG", fontsize=18)  # Add x-label to bottom plot
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlabel("")  # Remove x-label from top plot
    # ax1.set_xticks([])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, which='both', linestyle='-', alpha=0.5)  # Show both major and minor gridlines
    ax2.grid(True, which='both', linestyle='-', alpha=0.5)  # Show both major and minor gridlines

    if physics_params:
        add_physics_textbox(ax1, physics_params)

    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    markers = [
        'o',    # circle
        's',    # square
        '^',    # triangle up
        'v',    # triangle down
        '<',    # triangle left
        '>',    # triangle right
        'D',    # diamond
        'd',    # thin diamond
        'p',    # pentagon
        'h',    # hexagon
        '8',    # octagon
        '*',    # star
        '+',    # plus
        'x',    # x
        '|',    # vertical line
        '_',    # horizontal line
        '.',    # point
        ',',    # pixel
        '1',    # tri_down
        '2',    # tri_up
        '3',    # tri_left
        '4',    # tri_right
        'P',    # plus (filled)
        'X',    # x (filled)
        'H'     # hexagon2
    ]
    
 
    # First, plot all deltas on the main plot (ax1)
    for i, d in enumerate(deltas[1:] if deltas[0] == 0.0 else deltas):
        if amp_R != 0.0:
            f = f'data_err_vs_maxdim_delta={d}_amp_R={amp_R}_init={init_state}.npz'
        else: 
            f = f'data_err_vs_maxdim_delta={d}_init={init_state}.npz'
        path = os.path.join(folder, f)
        
        print(f"Loading data from: {path}")
        data = np.load(path)
        maxdims = data["maxdims"]

        if vs == "error":
            vs_value = data["errors"]
            plot_title = f"Energy error vs Bond Dimension for 2D TFIM\n {nx}x{ny} sqr. lat. in {init_state} as init. state"
            fig.supylabel("Energy err. $|E - E_{ref}|$", fontsize=20)
        elif vs == "max_trunc_err":
            vs_value = data["max_truncation_errors"]
            plot_title = f"Truncation Error vs Bond Dimension for 2D TFIM\n {nx}x{ny} sqr. lat. in {init_state} as init. state"
            fig.supylabel("Max truncation error", fontsize=20)
        else:
            raise ValueError(f"Invalid vs value: {vs}")

        if amp_R != 0.0:
           plot_title += " with perturbation"

        ax1.plot(maxdims, vs_value, 
                label=f"$\delta$ = {d}", 
                linestyle=line_styles[i],
                marker=markers[i],
                markersize=5)

    # Now plot only the first delta on the small plot (ax2)
    d = deltas[0]  # Get first delta
    if amp_R != 0.0:
        f = f'data_err_vs_maxdim_delta={d}_amp_R={amp_R}_init={init_state}.npz'
    else: 
        f = f'data_err_vs_maxdim_delta={d}_init={init_state}.npz'
    path = os.path.join(folder, f)
    
    data = np.load(path)
    maxdims = data["maxdims"]
    vs_value = data["errors"] if vs == "error" else data["max_truncation_errors"]
    print(vs_value)
    ax2.plot(maxdims, vs_value,
            label=f"$\delta$ = {d}",
            linestyle=line_styles[0],
            marker=markers[-1],
            markersize=5,
            color='black')  # Use red color to distinguish in small plot

    ax1.set_title(plot_title, fontsize=20)
    ax1.legend(loc="best", fontsize=10)
    ax2.legend(loc="best", fontsize=10)

    # Create imgs directory if it doesn't exist
    save_dir = os.path.join(folder, "imgs")
    os.makedirs(save_dir, exist_ok=True)
    
    filename_base = "plot_err_vs_maxdim" if vs == "error" else "plot_trunc_err_vs_maxdim"
    filename = f'{filename_base}_{nx}x{ny}_init={init_state}' if amp_R == 0.0 else f'{filename_base}_{nx}x{ny}_per_init={init_state}'
    filename = os.path.join(save_dir, filename)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()


