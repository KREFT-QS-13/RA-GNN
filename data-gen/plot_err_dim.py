import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def draw_plots_error_vs_maxdim(nx:int, ny:int, delta:list[float], amp_R:float=0.0, filename:str="plot_err_vs_maxdim", vs:str="max_trunc_err", folder="Experiment_1"):
    plt.figure(figsize=(11, 7))
    plt.style.use('ggplot') # 'seaborn-v0_8-darkgrid' , 'tableau-colorblind10'
    plt.xlabel("Bond dimension of DMRG", fontsize=18)
    plt.yscale("log")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    markers = ["o", "s", "D", "P", "X", "H", "v", "^", "<", ">"]
    for i, d in enumerate(delta):
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

    save_dir = os.path.join("imgs", folder)
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{filename_base}_{nx}x{ny}' if amp_R == 0.0 else f'{filename_base}_{nx}x{ny}_per'
    filename = os.path.join(save_dir, filename)
    plt.savefig(filename+".png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot error vs bond dimension for TFIM')
    parser.add_argument('-nx', type=int, default=4, help='Number of sites in x direction')
    parser.add_argument('-ny', type=int, default=4, help='Number of sites in y direction')
    parser.add_argument('-amp_R', type=float, default=0.0)
    parser.add_argument('-vs', type=str, default="max_trunc_err", help='Variable to plot: error or max_trunc_err')
    parser.add_argument('-folder', type=str, default="Experiment_1", help='Folder to save the plots')
    args = parser.parse_args()

    amp_R = args.amp_R
    nx, ny = args.nx, args.ny
    vs = args.vs
    folder = args.folder

    delta = [0.0, 10.0, 20.0, 25.0, 30.0, 50.0, 100.0]
    print(f"Plot for: size = {nx}x{ny} , delta = {delta}")
    draw_plots_error_vs_maxdim(nx, ny, delta, amp_R, vs=vs, folder=folder)

if __name__ == "__main__":
    main()

