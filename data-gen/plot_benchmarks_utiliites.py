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

def load_data_to_plot(path_to_folder: str, init_state: str, optimal_bond_dim: int, init_linkdims: int, nx:int, ny:int, R:float, amp_R:float, alpha:int) -> Tuple[List[float], List[float]]:
    """
    Load and reconstruct data for plotting from a NumPy file.
    
    Args:
        path_to_folder (str): Path to the folder containing the data
        init_state (str): Initial state used in the simulation
        optimal_bond_dim (int): The bond dimension to use for the plot
        init_linkdims (int): Initial link dimensions used in the simulation
        
    Returns:
        Tuple[List[float], List[float]]: A tuple containing:
            - List of deltas
            - List of staggered magnetizations
    """
    # Note: path_to_folder now includes the alpha folder
    try:
        filename = os.path.join(path_to_folder, f"staggered_magnetization_{nx}x{ny}_alpha={int(alpha)}_R={R}_amp_R={amp_R}_init={init_state}_initdim={init_linkdims}.npz")
    except FileNotFoundError:
        filename = os.path.join(path_to_folder, f"staggered_magnetization_init={init_state}_initdim={init_linkdims}.npz")


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
    textstr = ' | '.join((
        r'$C_6 = %.2f$' % (physics_params['C6'],),
        r'$\alpha = %s$' % ( str(physics_params['alpha']) if 'alpha' in physics_params else 'all'),
        r'$R_0\pm\delta R= %.2f\pm%.2f$' % (physics_params['R'], physics_params['amp_R']),
    ))
    
    # Place text box in upper center
    props = dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='black', linewidth=1)
    ax.text(0.5, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center',
            bbox=props)

def plot_magnetization_phase_diagram(nx, ny, R, amp_R, alpha, path_to_folder, optimal_bond_dim, save_fig=False, physics_params=None, init_state:str="FM", init_linkdims:int=100):
    """
    Plot the heat map phase diagram.
    """
    deltas, staggered_magnetization = load_data_to_plot(path_to_folder, init_state, optimal_bond_dim, init_linkdims, nx, ny, R, amp_R, alpha)
    
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(deltas, staggered_magnetization, label=f"Bond dim. = {optimal_bond_dim} with quick start")
    
    if physics_params:
        add_physics_textbox(ax, physics_params)
    
    ax.legend(fontsize=10, loc="best")
    ax.set_title(f'Staggered magnetization of {nx}x{ny} square lattice (init={init_state}, initdim={init_linkdims})', fontsize=16)
    ax.set_ylabel('Staggered Magnetization', fontsize=13)
    ax.set_xlabel(r'Magnetic field coefficient $\Omega$', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=9)
    print(deltas)
    plt.grid(True)
    # ax.set_xticks(np.linspace(deltas[0], deltas[-1], 5))
    ax.set_xticks(deltas)
    plt.tight_layout()

    if save_fig:
        # Create imgs directory if it doesn't exist
        imgs_dir = os.path.join(path_to_folder, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)
        plt.savefig(os.path.join(imgs_dir, f"magnetization_phase_diagram_{nx}x{ny}_init={init_state}_initdim={init_linkdims}.png"))
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
                              vs:str="max_trunc_err", folder="Experiment_1", physics_params=None, init_state:str="FM", init_linkdims:int=100):
    # Create figure with two subplots
    plt.style.use('tableau-colorblind10') # 'seaborn-v0_8-darkgrid' , 'tableau-colorblind10'

    fig = plt.figure(figsize=(12, 8))  # Increased height to accommodate both plots
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.025)  # 4:1 height ratio, small gap
    ax1 = fig.add_subplot(gs[0])  # Main plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Small plot below, sharing x-axis
    
    ax2.set_xlabel("Bond dimension of DMRG", fontsize=18)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlabel("")
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, which='both', linestyle='-', alpha=0.5)
    ax2.grid(True, which='both', linestyle='-', alpha=0.5)

    # Set empty x-tick labels for ax1 while keeping grid lines
    ax1.tick_params(labelbottom=False)  # This removes the labels but keeps the tick marks and grid lines

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
            f = f'data_err_vs_maxdim_delta={d}_amp_R={amp_R}_init={init_state}_initdim={init_linkdims}.npz'
        else: 
            f = f'data_err_vs_maxdim_delta={d}_init={init_state}_initdim={init_linkdims}.npz'
        path = os.path.join(folder, f)
        
        print(f"Loading data from: {path}")
        data = np.load(path)
        maxdims = data["maxdims"]

        if vs == "error":
            vs_value = data["errors"]
            plot_title = f"Energy error vs Bond Dimension for 2D TFIM\n {nx}x{ny} sqr. lat. in {init_state} as init. state (initdim={init_linkdims})"
            fig.supylabel("Energy err. $|E - E_{ref}|$", fontsize=20)
        elif vs == "max_trunc_err":
            vs_value = data["max_truncation_errors"]
            plot_title = f"Truncation Error vs Bond Dimension for 2D TFIM\n {nx}x{ny} sqr. lat. in {init_state} as init. state (initdim={init_linkdims})"
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
    d = deltas[0]
    if amp_R != 0.0:
        f = f'data_err_vs_maxdim_delta={d}_amp_R={amp_R}_init={init_state}_initdim={init_linkdims}.npz'
    else: 
        f = f'data_err_vs_maxdim_delta={d}_init={init_state}_initdim={init_linkdims}.npz'
    path = os.path.join(folder, f)
    
    data = np.load(path)
    maxdims = data["maxdims"]
    vs_value = data["errors"] if vs == "error" else data["max_truncation_errors"]
    print(vs_value)
    if all(vs_value == 0.0):
        ax2.set_yscale("linear")
    
    ax2.plot(maxdims, vs_value,
            label=f"$\delta$ = {d}",
            linestyle=line_styles[0],
            marker=markers[-1],
            markersize=5,
            color='black')

    ax1.set_title(plot_title, fontsize=20)
    ax1.legend(loc="best", fontsize=10)
    ax2.legend(loc="best", fontsize=10)

    # Create imgs directory if it doesn't exist
    save_dir = os.path.join(folder, "imgs")
    os.makedirs(save_dir, exist_ok=True)
    
    filename_base = "plot_err_vs_maxdim" if vs == "error" else "plot_trunc_err_vs_maxdim"
    filename = f'{filename_base}_{nx}x{ny}_init={init_state}_initdim={init_linkdims}' if amp_R == 0.0 else f'{filename_base}_{nx}x{ny}_per_init={init_state}_initdim={init_linkdims}'
    filename = os.path.join(save_dir, filename)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()


def load_data_to_plot_staggered_magnetization_all_alpha(path_to_folder: str, init_state: str, optimal_bond_dim: int, init_linkdims: int, nx:int, ny:int, R:float, amp_R:float) -> Dict[float, Tuple[List[float], List[float]]]:
    """
    Load staggered magnetization data for different alpha values.
    
    Args:
        path_to_folder (str): Base path to the folder containing alpha-specific subfolders
        init_state (str): Initial state used in the simulation (e.g., "FM", "AFM")
        optimal_bond_dim (int): The bond dimension to use for the plot
        init_linkdims (int): Initial link dimensions used in the simulation
        nx (int): Number of sites in the x direction
        ny (int): Number of sites in the y direction
        
    Returns:
        Dict[float, Tuple[List[float], List[float]]]: Dictionary where:
            - Keys are alpha values
            - Values are tuples of (deltas, magnetizations) for each alpha
    """
    # Find all alpha folders in the path
    alpha_folders = glob.glob(os.path.join(path_to_folder, "alpha_*"))
    
    if not alpha_folders:
        raise ValueError(f"No alpha folders found in {path_to_folder}")
    
    # Dictionary to store results
    alpha_data = {}
    
    # Process each alpha folder
    for alpha_folder in alpha_folders:
        try:
            # Extract alpha value from folder name
            alpha = float(alpha_folder.split("alpha_")[-1])
            
            # Construct path to the lattice size folder
            lattice_folder = os.path.join(alpha_folder, f"{nx}x{ny}")
            if not os.path.exists(lattice_folder):
                print(f"Warning: No {nx}x{ny} folder found in {alpha_folder}")
                continue
            
            # Load data for this alpha value
            deltas, magnetizations = load_data_to_plot(
                lattice_folder,  # Use the lattice folder path
                init_state=init_state,
                optimal_bond_dim=optimal_bond_dim,
                init_linkdims=init_linkdims,
                nx=nx,
                ny=ny,
                R=R,
                amp_R=amp_R,
                alpha=alpha
            )
            
            # Store in dictionary
            alpha_data[alpha] = (deltas, magnetizations)
            
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not process folder {alpha_folder}: {str(e)}")
            continue
    
    if not alpha_data:
        raise ValueError(f"No valid data found in any alpha folder for lattice size {nx}x{ny}, path: {alpha_folders}")
    
    # Sort the dictionary by alpha values
    return dict(sorted(alpha_data.items()))


def plot_phase_diagram_all_alpha(nx: int, ny: int, R:float, amp_R:float, path_to_folder: str, optimal_bond_dim: int, save_fig: bool = False, 
                               physics_params: dict = None, init_state: str = "FM", init_linkdims: int = 100):
    """
    Plot the staggered magnetization phase diagram for all alpha values.
    
    Args:
        nx (int): Number of sites in the x direction
        ny (int): Number of sites in the y direction
        path_to_folder (str): Path to the folder containing alpha-specific subfolders
        optimal_bond_dim (int): The bond dimension to use for the plot
        save_fig (bool): Whether to save the figure
        physics_params (dict): Dictionary containing physics parameters for the text box
        init_state (str): Initial state used in the simulation (e.g., "FM", "AFM")
        init_linkdims (int): Initial link dimensions used in the simulation
    """
    # Load data for all alpha values
    alpha_data = load_data_to_plot_staggered_magnetization_all_alpha(
        path_to_folder=path_to_folder,
        init_state=init_state,
        optimal_bond_dim=optimal_bond_dim,
        init_linkdims=init_linkdims,
        nx=nx,
        ny=ny,
        R=R,
        amp_R=amp_R
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6)) 

    markers = [
        'o',    # circle
        's',    # square
        '^',    # triangle up
        'D',    # diamond
    ]
    line_styles = ["-", "--", "-.", ":"]

    # Plot data for each alpha value
    for i, (alpha, (deltas, magnetizations)) in enumerate(alpha_data.items()):
        ax.plot(deltas, magnetizations, 
                label=f"α = {alpha}", 
                marker=markers[i],  # Add markers for better visibility
                markersize=4,
                linestyle=line_styles[i],
                linewidth=1.5)
    
    # Add physics parameters text box if provided
    if physics_params:
        add_physics_textbox(ax, physics_params)
    
    # Customize the plot
    ax.set_title(f'Staggered magnetization of {nx}x{ny} square lattice\n'
                f'(init={init_state}, initdim={init_linkdims}, bond dim={optimal_bond_dim})', 
                fontsize=14)
    ax.set_xlabel(r'Magnetic field coefficient $\Omega$', fontsize=12)
    ax.set_ylabel('Staggered Magnetization', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Alpha values', loc='best', fontsize=11)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        save_dir = os.path.join(path_to_folder, "agg_imgs")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"magnetization_phase_diagram_all_alpha_{nx}x{ny}_init={init_state}_initdim={init_linkdims}.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig


def plot_FM_and_AFM_error_vs_bond_dim(vs, nx=5, ny=5, folder="./Benchmark-normalized/",  deltas=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10], save_fig=True, 
                                     physics_params={'C6': 1.0, 'alpha': "NaN", 'R': 1.0, 'amp_R': 0.0}, init_linkdims=100):
    """
    Plot the error vs bond dimension for both FM and AFM states.
    AFM data is plotted in light grey to shadow the FM data.
    
    Args:
        nx (int): Number of sites in x direction
        ny (int): Number of sites in y direction
        folder (str): Path to the benchmark data folder
        vs (str): Type of plot to show: "error" or "max_trunc_err"
        deltas (list): List of deltas to plot
        optimal_bond_dim (int): Optimal bond dimension to highlight
        save_fig (bool): Whether to save the figure
        physics_params (dict): Dictionary containing physics parameters
        init_linkdims (int): Initial link dimensions used in simulation
    """
    plt.style.use('tableau-colorblind10')
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])  # Main plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Small plot below
    
    # Set up axes properties
    ax2.set_xlabel("Bond dimension of DMRG", fontsize=18)
    ax1.set_yscale("log")
    ax2.set_yscale("linear")
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlabel("")
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, which='both', linestyle='-', alpha=0.5)
    ax2.grid(True, which='both', linestyle='-', alpha=0.5)
    
    # Set empty x-tick labels for ax1 while keeping grid lines
    ax1.tick_params(labelbottom=False)  # This removes the labels but keeps the tick marks and grid lines
    
    if physics_params:
        add_physics_textbox(ax1, physics_params)
    
    # Define markers and line styles
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
    
    # Plot for both FM and AFM
    for init_state, color in [("AFM", "darkgrey"),("FM", None),]:
        # Plot all deltas on the main plot (ax1)
        for i, d in enumerate(deltas):
            if physics_params['amp_R'] != 0.0:
                f = f'data_err_vs_maxdim_delta={d}_amp_R={physics_params["amp_R"]}_init={init_state}_initdim={init_linkdims}.npz'
            else:
                f = f'data_err_vs_maxdim_delta={d}_init={init_state}_initdim={init_linkdims}.npz'
            path = os.path.join(folder, f"alpha_{int(physics_params['alpha'])}/{nx}x{ny}", f)
            
            try:
                data = np.load(path)
                maxdims = data["maxdims"]
                if vs == "error":
                        vs_value = data["errors"]
                        plot_title = f"Energy error vs Bond Dimension for 2D TFIM\n {nx}x{ny} sqr. lat. both FM and AFM (in grey) (initdim={init_linkdims})"
                        fig.supylabel("Energy err. $|E - E_{ref}|$", fontsize=20)
                elif vs == "max_trunc_err":
                        vs_value = data["max_truncation_errors"]
                        plot_title = f"Truncation Error vs Bond Dimension for 2D TFIM\n {nx}x{ny} sqr. lat. both FM and AFM (in grey) (initdim={init_linkdims})"
                        fig.supylabel("Max truncation error", fontsize=20)
                else:
                        raise ValueError(f"Invalid vs value: {vs}")
                
                # Plot on main axis
                if d != 0.0:
                    ax1.plot(maxdims, vs_value,
                        label=f"{init_state} $\delta$ = {d}" if color is None else None,  # Only label FM data
                        linestyle=line_styles[i],
                        marker=markers[i],
                        markersize=5,
                        color=color,
                        alpha=0.75 if color is not None else 1.0)
                else:
                    # Plot first delta on small plot (ax2)
                    ax2.plot(maxdims, vs_value,
                            label=f"{init_state} $\delta$ = {d}" if color is None else None,
                            linestyle=line_styles[0],
                            marker=markers[-1],
                            markersize=5,
                            color=color,
                            alpha=0.75 if color is not None else 1.0)
                           
                    print(f'Values to plot for {init_state} and delta {d} are:\n {vs_value}.')           

         

            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load data for {init_state}, delta={d}: {str(e)}")
                continue
    
    ax1.set_title(plot_title, fontsize=20)
    
    # Add legend only for FM data
    ax1.legend(loc="best", fontsize=10)
    ax2.legend(loc="best", fontsize=10)
    
    # Save figure if requested
    if save_fig:
        save_dir = os.path.join(folder, "agg_imgs")
        os.makedirs(save_dir, exist_ok=True)
        if vs == "error":
            filename = f"plot_energy_err_vs_maxdim_FM_AFM_{nx}x{ny}_initdim={init_linkdims}_alpha={physics_params['alpha']}"
        elif vs == "max_trunc_err":
            filename = f"plot_trunc_err_vs_maxdim_FM_AFM_{nx}x{ny}_initdim={init_linkdims}_alpha={physics_params['alpha']}"
        if physics_params['amp_R'] != 0.0:
            filename += "_pert"
        plt.savefig(os.path.join(save_dir, filename + ".png"), bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig

def plot_time_vs_bond_dim(
    nx: int,
    ny: int,
    R: float,
    amp_R: float,
    alpha: int,
    path_to_folder: str,
    init_state: str = "FM",
    init_linkdims: int = 100,
    save_fig: bool = False,
    physics_params: dict = None,
):
    """
    Plot DMRG time vs bond dimension for different deltas from drmg_time_*.npz file.
    Args:
        nx, ny: lattice size
        R, amp_R: lattice parameters
        alpha: interaction exponent
        path_to_folder: path to alpha_*/size/ folder
        init_state: initial state (FM/AFM)
        init_linkdims: initial link dimension
        save_fig: whether to save the figure
        physics_params: dict for physics text box
    """
    # Extract alpha from the folder path (e.g., "alpha_1" -> 1)
    folder_alpha = None
    for part in path_to_folder.split(os.sep):
        if part.startswith("alpha_"):
            folder_alpha = int(part.split("alpha_")[-1])
            break
    
    if folder_alpha is None:
        raise ValueError(f"Could not extract alpha from path: {path_to_folder}")
    
    # Use the alpha from the folder path, not the passed parameter
    actual_alpha = folder_alpha
    
    # Compose filename using the actual alpha from the folder
    filename = os.path.join(
        path_to_folder,
        f"drmg_time_{nx}x{ny}_alpha={actual_alpha}_R={R}_amp_R={amp_R}_init={init_state}_initdim={init_linkdims}.npz"
    )
    if not os.path.exists(filename):
        # Try fallback (old style)
        filename = os.path.join(
            path_to_folder,
            f"drmg_time_init={init_state}_initdim={init_linkdims}.npz"
        )
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Could not find drmg_time npz file in {path_to_folder}")

    # Load nested dict: {bond_dim: {delta: time}}
    data_dict = load_nested_dict_int_to_pairs(filename)

    # Get all bond_dims and deltas, filtering out -1.0 (total experiment time)
    bond_dims = sorted([bd for bd in data_dict.keys() if bd != -1.0])
    total_time = data_dict[-1.0][0.0]
    deltas = set()
    for bd in bond_dims:
        deltas.update(data_dict[bd].keys())
    deltas = sorted(deltas)

    # Prepare data for plotting: for each delta, get (bond_dim, time)
    times_per_delta = {d: [] for d in deltas}
    for d in deltas:
        for bd in bond_dims:
            # Some bond_dims may not have all deltas
            t = data_dict[bd].get(d, np.nan)
            times_per_delta[d].append(t)

    # Plot
    plt.style.use('tableau-colorblind10')
    plt.figure(figsize=(8,6))
    # plt.yscale("log")
    markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+', 'h', 'p']
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]

    for i, d in enumerate(deltas):
        plt.plot(bond_dims, times_per_delta[d],
                 marker=markers[i%len(markers)], label=f"$\\delta$ = {d}", linestyle=line_styles[i%len(line_styles)])
    plt.xlabel("Bond dimension", fontsize=14)
    plt.ylabel("DMRG time [s]", fontsize=14)
    plt.title(f"DMRG time vs bond dimension ({nx}x{ny}, init={init_state}, initdim={init_linkdims})\n Total time: {total_time:.2f} s", fontsize=15)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    if physics_params:
        # Update physics_params with the actual alpha
        updated_physics_params = {**(physics_params or {}), 'alpha': actual_alpha}
        add_physics_textbox(plt.gca(), updated_physics_params)
    plt.tight_layout()
    if save_fig:
        imgs_dir = os.path.join(path_to_folder, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)
        plt.savefig(os.path.join(imgs_dir, f"plot_time_vs_maxdim_{nx}x{ny}_alpha={actual_alpha}_R={R}_amp_R={amp_R}_init={init_state}_initdim={init_linkdims}.png"))
        print(f"Saved figure to {os.path.join(imgs_dir, f'plot_time_vs_maxdim_{nx}x{ny}_alpha={actual_alpha}_R={R}_amp_R={amp_R}_init={init_state}_initdim={init_linkdims}.png')}")
    plt.show()

# Optionally, a function to loop over all alpha folders and call the above for each

def plot_time_vs_bond_dim_all_alpha(
    nx: int,
    ny: int,
    path_to_folder: str,
    save_fig: bool = False,
    physics_params: dict = None,
    init_state: str = "FM",
    init_linkdims: int = 100,
    R: float = 1.0,
    amp_R: float = 0.0,
):
    """
    For each alpha_* subfolder, plot DMRG time vs bond dimension for all deltas.
    """
    import glob
    alpha_folders = glob.glob(os.path.join(path_to_folder, "alpha_*"))
    if not alpha_folders:
        raise ValueError(f"No alpha_* folders found in {path_to_folder}")
    for alpha_folder in sorted(alpha_folders):
        try:
            alpha = float(alpha_folder.split("alpha_")[-1])
            size_folder = os.path.join(alpha_folder, f"{nx}x{ny}")
            if not os.path.exists(size_folder):
                print(f"No {nx}x{ny} folder in {alpha_folder}")
                continue
            print(f"Plotting for alpha={alpha} in {size_folder}")
            plot_time_vs_bond_dim(
                nx, ny, R, amp_R, alpha, size_folder,
                init_state=init_state, init_linkdims=init_linkdims,
                save_fig=save_fig,
                physics_params={**(physics_params or {}), 'alpha': alpha, 'R': R, 'amp_R': amp_R}
            )
        except Exception as e:
            print(f"Failed for {alpha_folder}: {e}")