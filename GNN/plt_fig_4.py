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
    FONTSIZE = 24
    LEGENDSIZE = 18
    MARKER = 'D'
    COLORS = ['black','grey','lightgrey','goldenrod','darkorange','tomato']

    TYPE = 'GNN' # TRGT, GNN, SNPT or SIZE

    if TYPE=='TRGT': # incld_scdn vs not
        figname = "./Figs/extrapolation_trgt.pdf"
        COLORS_2 = ['darkred','navy','violet','pink']
        MARKER = '^'
    elif TYPE=='SIZE': # compare various sizes in training set
        figname = "./Figs/extrapolation.pdf"
        COLORS_2 = ['purple','pink']
    elif TYPE=='GNN': # compare various edge features in GNN
        figname = "./Figs/extrapolation_GNN.pdf"
    elif TYPE=="SNPT":
        COLORS_2 = ['black','silver','khaki','tan','deepskyblue','cyan','darkred','red']
        figname = "./Figs/extrapolation_SNPT.pdf"

    assert len(sys.argv) > 1, "Need to provide the path to the hdf5 file containing the metrics."
    paths_to_data = list(map(str,sys.argv))[1:]

    if TYPE == 'TRGT' or TYPE == 'SIZE' or TYPE=="SNPT":

        fig, ax = plt.subplots(nrows=3,sharex=True)
        # fig.suptitle("Metrics",y=0.95,fontsize=20)
        fig.set_size_inches(10,8)
        for a in ax:
            a.yaxis.set_minor_locator(AutoMinorLocator(n=2))
            a.yaxis.grid(True, which='major',linestyle='-')
            a.yaxis.grid(True, which='minor',linestyle='--')
            a.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            a.xaxis.grid(True, which='major',linestyle='-')
            a.xaxis.grid(True, which='minor',linestyle='--')

        ax[0].tick_params(labelbottom=False,bottom=False,which='both')
        
        ax[1].tick_params(labelbottom=False,bottom=False,which='both')

        if TYPE=='TRGT':
            ax[0].text(0.45,0.5,r"$R^2$",transform=ax[0].transAxes,fontsize=18)
            ax[1].text(0.45,0.5,'MAE',transform=ax[1].transAxes,fontsize=18)
            ax[2].text(0.45,0.5,'MEDAE',transform=ax[2].transAxes,fontsize=18)
        elif TYPE=='SIZE':
            ax[0].text(0.75,0.5,r"$R^2$",transform=ax[0].transAxes,fontsize=18)
            ax[1].text(0.75,0.5,'MAE',transform=ax[1].transAxes,fontsize=18)
            ax[2].text(0.75,0.5,'MEDAE',transform=ax[2].transAxes,fontsize=18)
        elif TYPE=="SNPT":
            ax[0].text(0.8,0.8,r"$R^2$",transform=ax[0].transAxes,fontsize=18)
            ax[1].text(0.8,0.8,'MAE',transform=ax[1].transAxes,fontsize=18)
            ax[2].text(0.8,0.8,'MEDAE',transform=ax[2].transAxes,fontsize=18)
            
        for aa in ax:
            if TYPE=='TRGT':
                aa.axvspan(2-0.02,2+0.02,color='gray',alpha=0.5)
                aa.axvspan(3-0.02,3+0.02,color='gray',alpha=1.0)
            elif TYPE=='SIZE':
                aa.axvspan(1-0.02,1+0.02,color=COLORS_2[0],alpha=0.5)
                aa.axvspan(2-0.02,2+0.02,color=COLORS_2[1],alpha=0.5)

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
            label = None
            if TYPE=='TRGT':
                incl_second = str(search(r'_inclscdn_([^_]+)',path_to_data).group(1))
                if incl_second=="True":
                    label = "NN+NNN ({})".format(training_sizes[0].replace('_',','))
                else:
                    label = "NN ({})".format(training_sizes[0].replace('_',','))
            elif TYPE=="SNPT":
                smpl = int(search(r'_SMPL_(\d*)',path_to_data).group(1))
                tmp = None
                if "Mg_1_" in path_to_data:
                    tmp = dict_labels['Mg+1+NN+NNN+1+delta+hist']
                else:
                    tmp = dict_labels['Mg+NN+NNN+delta+hist']
                label = "{} ({})".format(smpl,tmp)

            xs = np.arange(len(extrapol_sizes))
            key = 'R2'
            for ii,kk in enumerate(training_sizes):
                ys = []
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
                if TYPE=='SIZE':
                    label = '{}'.format(training_sizes[ii].replace('_',' '))
            ax[0].plot(xs,ys,ms=MS,marker=MARKER,label=label,c=COLORS_2[cc])
            if TYPE=='SIZE':
                ax[0].set_ylim(top=1.0,bottom=0.93)
            elif TYPE=='TRGT':
                ax[0].set_ylim(top=1.0,bottom=0.95)
            elif TYPE=='SNPT':
                ax[0].set_ylim(top=1.0,bottom=-0.1)

            key = 'MAE'
            ys = []
            for kk in training_sizes:
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            ax[1].plot(xs,np.array(ys)*1000,ms=MS,marker=MARKER,c=COLORS_2[cc])
            if TYPE=='SIZE':
                ax[1].set_ylim(bottom=0.0,top=0.02*1000)
            elif TYPE=='TRGT':
                ax[1].set_ylim(bottom=0.0,top=0.015*1000)
            elif TYPE=='SNPT':
                ax[1].set_ylim(bottom=0.0,top=0.05)

            key = 'MEDAE'
            ys = []
            for kk in training_sizes:
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            ax[2].plot(xs,np.array(ys)*1000,ms=MS,marker=MARKER,label='median absolute error',c=COLORS_2[cc])
            if TYPE=='SIZE':
                ax[2].set_ylim(bottom=0.0,top=0.015*1000)
            elif TYPE=='TRGT':
                ax[2].set_ylim(bottom=0.0,top=0.015*1000)
            elif TYPE=='SNPT':
                ax[2].set_ylim(bottom=0.0,top=0.015)

            plt.xticks(xs, extrapol_sizes, rotation='vertical')

        for ii in range(len(ax)):
            ax[ii].tick_params(axis='y',which='major',labelsize=LABELSIZE)
        ax[2].tick_params(axis='x',which='major',labelsize=LABELSIZE)
        ax[2].set_xlabel("Cluster size",fontsize=FONTSIZE)

        if TYPE=='SNPT':
            ax[0].legend(ncol=4, prop={'size': 20}, bbox_to_anchor=(1.0, 1.52), borderaxespad=0, handletextpad=0.2, handlelength=0.8)
        else:
            ax[0].legend(ncol=2, prop={'size': 20}, bbox_to_anchor=(1.0, 1.32), borderaxespad=0, handletextpad=0.2, handlelength=1.0)

    elif TYPE == 'GNN':
        d = .01  # how big to make the diagonal lines in axes coordinates
        fig, ax = plt.subplots(nrows=6,sharex=True)
        # fig.suptitle("Metrics",y=0.97,fontsize=20)
        fig.set_size_inches(10,16)
        for a in ax:
            a.yaxis.set_minor_locator(AutoMinorLocator(n=2))
            a.yaxis.grid(True, which='major',linestyle='-')
            a.yaxis.grid(True, which='minor',linestyle='--')
            a.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            a.xaxis.grid(True, which='major',linestyle='-')
            a.xaxis.grid(True, which='minor',linestyle='--')

        for ii in range(len(ax)-1):
            ax[ii].tick_params(labelbottom=False,bottom=False,which='both')
        
        ax[0].text(0.75,0.5,r"$R^2$",transform=ax[0].transAxes,fontsize=18)
        ax[2].text(0.75,0.5,'MAE',transform=ax[2].transAxes,fontsize=18)
        ax[4].text(0.75,0.5,'MEDAE',transform=ax[4].transAxes,fontsize=18)
        
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
            ax[1].plot(xs,ys,ms=MS,marker=MARKER,c=COLORS[cc])
            
            # zoom-in / limit the view to different portions of the data
            ax[0].set_ylim(.98, 1.)  # outliers only
            ax[1].set_ylim(-.01, .6)  # most of the data

            key = 'MAE'
            ys = []
            for kk in training_sizes:
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            ax[2].plot(xs,np.array(ys)*ORDER_CONVERSION,ms=MS,marker=MARKER,c=COLORS[cc])
            ax[3].plot(xs,np.array(ys)*ORDER_CONVERSION,ms=MS,marker=MARKER,c=COLORS[cc])

            # zoom-in / limit the view to different portions of the data
            ax[2].set_ylim(.04*ORDER_CONVERSION, .07*ORDER_CONVERSION)  # outliers only
            ax[3].set_ylim(.0*ORDER_CONVERSION, .01*ORDER_CONVERSION)  # most of the data

            key = 'MEDAE'
            ys = []
            for kk in training_sizes:
                for ext in extrapol_sizes:
                    dat = data_dict[kk+'/'+ext+'/'+key]
                    ys.append(dat)
            ax[4].plot(xs,np.array(ys)*ORDER_CONVERSION,ms=MS,marker=MARKER,label='median absolute error',c=COLORS[cc])
            ax[5].plot(xs,np.array(ys)*ORDER_CONVERSION,ms=MS,marker=MARKER,c=COLORS[cc])

            # zoom-in / limit the view to different portions of the data
            ax[4].set_ylim(.029*ORDER_CONVERSION, .06*ORDER_CONVERSION)  # outliers only
            ax[5].set_ylim(-.001*ORDER_CONVERSION, .01*ORDER_CONVERSION)  # most of the data

            # hide the spines between ax and ax2
            ax[0].spines['bottom'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[3].spines['top'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[5].spines['top'].set_visible(False)
            # ax[0].xaxis.tick_top()
            # ax[0].tick_params(labeltop='off')  # don't put tick labels at the top
            ax[5].xaxis.tick_bottom()

            for ii in range(len(ax)):
                ax[ii].tick_params(axis='y',which='major',labelsize=LABELSIZE)
            ax[5].tick_params(axis='x',which='major',labelsize=LABELSIZE)

            for ii in range(0,len(ax),2):
                # arguments to pass to plot, just so we don't keep repeating them
                kwargs = dict(transform=ax[ii].transAxes, color='k', clip_on=False)
                ax[ii].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                ax[ii].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax[ii+1].transAxes)  # switch to the bottom axes
                ax[ii+1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax[ii+1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            plt.xticks(xs, extrapol_sizes, rotation='vertical')

        ax[5].set_xlabel("Cluster size",fontsize=FONTSIZE)
        # Set global ylabel using fig.text()
        ax[0].legend(ncol=3, prop={'size': LEGENDSIZE}, bbox_to_anchor=(1.0, 1.5), borderaxespad=0, handletextpad=0.2)

    if TYPE=='TRGT' or TYPE=="SNPT":
        fig.text(0.04, 0.23, r"nm", va='center', rotation='vertical', fontsize=FONTSIZE)
    else:
        fig.text(0.04, 0.23, r"nm", va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.text(0.04, 0.495, r"nm", va='center', rotation='vertical', fontsize=FONTSIZE)

    fig.savefig(figname)
    plt.show()