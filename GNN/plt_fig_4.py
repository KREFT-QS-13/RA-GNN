"""
Filename: plt_extrapol_dffrt_trn_sizes.py
Description: Script used to produce figures in the paper. In particular, ffor the example, selecting the variable type="GNN" produces figure 4 in the paper. 
The paths to all the files produced through executing the file `GNN_training.py`, starting with "cluster_one_shots_*.h5" need to be passed in via the command line.
Author: Olivier Simard
Date: 2024-08-11
License: MIT License
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from re import search
from matplotlib.ticker import (AutoMinorLocator)
import argparse

# Enable LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # For additional LaTeX packages

dict_labels = {
    'Mg+delta+hist': r'$\#1$',
    'Mg+NN+delta+hist': r'$\#2$',
    'Mg+NN+NNN+delta+hist': r'$\#3$',
    'Mg+NN+NNN+1+delta+hist': r'$\#4$',
    'Mg+1+NN+NNN+1+delta+hist': r'$\#5$',
    'Mg+NN+NNN+X+delta+hist': r'$\#6$'
}

if __name__=='__main__':
    ORDER_CONVERSION = 1000
    MS = 3.5
    LABELSIZE = 18
    FONTSIZE = 22
    LEGENDSIZE = 18
    MARKER = 'D'
    COLORS = ['black','grey','lightgrey','goldenrod','darkorange','tomato']
    
    # compare various edge features in GNN
    TYPE = 'GNN' # TRGT, GNN, SNPT or SIZE
    figname = "./Figs/extrapolation_GNN.pdf"

    assert len(sys.argv) > 1, "Need to provide the path to the hdf5 file containing the metrics."
    parser = argparse.ArgumentParser(description='Plotting the extrapolation of the GNN.')
    parser.add_argument('--path', type=str, required=True, help='Path to the hdf5 file containing the metrics.')
    args = parser.parse_args()
    paths_to_data = [args.path]

    if TYPE == 'GNN':
        d = .01  # how big to make the diagonal lines in axes coordinates
        fig, ax = plt.subplots(nrows=3,sharex=True)
        # fig.suptitle("Metrics",y=0.97,fontsize=20)
        fig.set_size_inches(10,8)
        for a in ax:
            a.yaxis.set_minor_locator(AutoMinorLocator(n=2))
            a.yaxis.grid(True, which='major',linestyle='-')
            a.yaxis.grid(True, which='minor',linestyle='--')
            a.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            a.xaxis.grid(True, which='major',linestyle='-')
            a.xaxis.grid(True, which='minor',linestyle='--')

        for ii in range(len(ax)-1):
            ax[ii].tick_params(labelbottom=False,bottom=False,which='both')
        
       
        # Add y-axis labels for each metric
        ax[0].set_ylabel(r"$R^2$", fontsize=FONTSIZE)
        ax[1].set_ylabel("MAE (nm)", fontsize=FONTSIZE)
        ax[2].set_ylabel("MEDAE (nm)", fontsize=FONTSIZE)
        
        for aa in ax:
            aa.axvspan(2-0.02,2+0.02,color='gray',alpha=0.5)

        for cc,path_to_data in enumerate(paths_to_data):
            data_dict = {}
            training_sizes = None
            extrapol_sizes = None
            with h5py.File(path_to_data,'r') as ff:
                training_sizes = list(ff.keys())
                for kk in training_sizes: # level of training sizes
                    extrapol_sizes = list(ff.get(kk).keys())
                    for ll in extrapol_sizes: # level of extrapolation sizes
                        k3 = ff.get(kk).get(ll).keys()
                        for mets in k3: # level of metrics
                            data_dict[kk+'/'+ll+'/'+mets] = list(ff.get(kk).get(ll).get(mets))[0]
            
            print(f"extrapol sizes = {extrapol_sizes}")

            label = path_to_data.split('/')[1].replace('_','+')
            label =  'Mg+1+NN+NNN+1+delta+hist'
            label = dict_labels[label]

            xs = np.arange(len(extrapol_sizes))
            key = 'R2'
            ys = []
            for ii,kk in enumerate(training_sizes):
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            
            ax[0].plot(xs,ys,ms=MS,marker=MARKER,label=label,c=COLORS[cc])
            
            # zoom-in / limit the view to different portions of the data
            ax[0].set_ylim(.98, 1.)  # outliers only

            key = 'MAE'
            ys = []
            for kk in training_sizes:
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            ax[1].plot(xs,np.array(ys)*ORDER_CONVERSION,ms=MS,marker=MARKER,c=COLORS[cc])

            # zoom-in / limit the view to different portions of the data
            ax[1].set_ylim(.0*ORDER_CONVERSION, .01*ORDER_CONVERSION)  # most of the data

            key = 'MEDAE'
            ys = []
            for kk in training_sizes:
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            ax[2].plot(xs,np.array(ys)*ORDER_CONVERSION,ms=MS,marker=MARKER,label='median absolute error',c=COLORS[cc])

            # zoom-in / limit the view to different portions of the data
            ax[2].set_ylim(-.001*ORDER_CONVERSION, .01*ORDER_CONVERSION)  # most of the data

            # hide the spines between ax and ax2
            ax[0].spines['bottom'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[2].spines['top'].set_visible(False)
            ax[2].xaxis.tick_bottom()

            for ii in range(len(ax)):
                ax[ii].tick_params(axis='y',which='major',labelsize=LABELSIZE)
            ax[2].tick_params(axis='x',which='major',labelsize=LABELSIZE)

            for ii in range(0,len(ax)-1):
                # arguments to pass to plot, just so we don't keep repeating them
                kwargs = dict(transform=ax[ii].transAxes, color='k', clip_on=False)
                ax[ii].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                ax[ii].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax[ii+1].transAxes)  # switch to the bottom axes
                ax[ii+1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax[ii+1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            plt.xticks(xs, extrapol_sizes, rotation='vertical')

        ax[2].set_xlabel("Cluster size",fontsize=FONTSIZE)
        # Set global ylabel using fig.text()
        ax[0].legend(ncol=3, prop={'size': LEGENDSIZE}, bbox_to_anchor=(1.0, 1.5), borderaxespad=0, handletextpad=0.2)


    fig.savefig(figname)
    plt.show()