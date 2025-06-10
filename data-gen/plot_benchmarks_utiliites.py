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
    size_dir = os.path.join(path_to_folder, f"{nx}x{ny}")
    filename = os.path.join(size_dir, f"staggered_magnetization.npz")

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
    props = dict(boxstyle='round', facecolor='silver', alpha=0.6, edgecolor='black', linewidth=1)
    ax.text(0.5, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=props)

def plot_magnetization_phase_diagram(nx, ny, path_to_folder, optimal_bond_dim, save_fig=False, physics_params=None):
    """
    Plot the heat map phase diagram.
    """
    deltas, staggered_magnetization = load_data_to_plot(path_to_folder, nx, ny, optimal_bond_dim)
    
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(deltas, staggered_magnetization, label=f"Bond dim. = {optimal_bond_dim} with quick start")
    
    if physics_params:
        add_physics_textbox(ax, physics_params)
    
    ax.legend(fontsize=13)
    ax.set_title(f'Staggered magnetization of {nx}x{ny} square lattice', fontsize=16)
    ax.set_ylabel('Staggered Magnetization', fontsize=13)
    ax.set_xlabel(r'Magnetic field coefficient $\Omega$', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    print(deltas)
    ax.set_xticks(deltas)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"{path_to_folder}/imgs/magnetization_phase_diagram_{nx}x{ny}.png")
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
                              vs:str="max_trunc_err", folder="Experiment_1", physics_params=None):
    plt.figure(figsize=(11, 7))
    plt.style.use('ggplot') # 'seaborn-v0_8-darkgrid' , 'tableau-colorblind10'
    plt.xlabel("Bond dimension of DMRG", fontsize=18)
    plt.yscale("log")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    if physics_params:
        add_physics_textbox(plt.gca(), physics_params)

    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]

    # All possible markers
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
    if 0 in deltas:
        deltas = deltas[1:]

    for i, d in enumerate(deltas):
        # Construct the path using os.path.join for proper path handling
        size_dir = os.path.join(folder, f"{nx}x{ny}")
        if amp_R != 0.0:
            f = f'data_err_vs_maxdim_{nx}x{ny}_delta={d}_amp_R={amp_R}.npz'
        else: 
            f = f'data_err_vs_maxdim_{nx}x{ny}_delta={d}.npz'
        path = os.path.join(size_dir, f)
        
        print(f"Loading data from: {path}")
        data = np.load(path)
        maxdims = data["maxdims"]

        if vs == "error":
            vs_value = data["errors"]
            plot_title = f"Energy error vs Bond Dimension for 2D TFIM, {nx}x{ny} sqr. lat."
            plt.ylabel("Energy err. $|E - E_{ref}|$", fontsize=18)
        elif vs == "max_trunc_err":
            vs_value = data["max_truncation_errors"]
            plot_title = f"Trunc. err. vs Bond Dimension for 2D TFIM, {nx}x{ny} sqr. lat."
            plt.ylabel("Max truncation error", fontsize=18)
        else:
            raise ValueError(f"Invalid vs value: {vs}")

        if amp_R != 0.0:
           plot_title += " (per.)"


        plt.title(plot_title, fontsize=20)
        plt.plot(maxdims, vs_value, 
                 label=f"$\delta$ = {d}", 
                 linestyle=line_styles[i],
                 marker=markers[i],
                 markersize=5,
                )

    plt.legend(loc="upper right", fontsize=13)

    if vs == "error":
        filename_base = "plot_err_vs_maxdim"
    elif vs == "max_trunc_err":
        filename_base = "plot_trunc_err_vs_maxdim"
    else:
        raise ValueError(f"Invalid vs value: {vs}")

    save_dir = os.path.join(folder, "imgs")
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{filename_base}_{nx}x{ny}' if amp_R == 0.0 else f'{filename_base}_{nx}x{ny}_per'
    filename = os.path.join(save_dir, filename)
    plt.savefig(filename+".png")
    plt.show()


