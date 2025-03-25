import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.xxs

    :param directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"\nDirectory created: {directory_path}")
    else:
        print(f"\nDirectory {directory_path} already existed. You might have overwritten the data!")


def import_data(file_path):
    # file A loading as numpy array
    try:

        data = np.loadtxt(file_path)
        print(f"File {file_path} successfully loaded")
        return data

    except Exception as e:
        print(f"Error while loading file: {e}")


def mi_import_all_data(n, overlap, k):
    # A files
    file_path1 = f'test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/A14F3-TE10034.txt'
    data1 = import_data(file_path1)

    file_path3 = f'test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/A15F3-TE10034.txt'
    data3 = import_data(file_path3)

    file_path5 = f'test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/A16F3-TE10034.txt'
    data5 = import_data(file_path5)

    # C files
    file_path6 = f'test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/C11F3-VE.txt'
    data6 = import_data(file_path6)

    file_path8 = f'test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/C12F3-VE.txt'
    data8 = import_data(file_path8)

    file_path9 = f'test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/C13F3-VE.txt'
    data9 = import_data(file_path9)

    data = { 'A14F3': data1, 'A15F3':data3, 'A16F3': data5, 
            'C11F3':data6, 'C12F3':data8, 'C13F3':data9}
    
    return data

def plus_import_all_data(n, overlap):
    # A files
    file_path1 = f'test_vallone/n_{n}/{overlap}/mi/plus/num_data/A14F3-TE10034.txt'
    data1 = import_data(file_path1)

    file_path3 = f'test_vallone/n_{n}/{overlap}/mi/plus/num_data/A15F3-TE10034.txt'
    data3 = import_data(file_path3)

    file_path5 = f'test_vallone/n_{n}/{overlap}/mi/plus/num_data/A16F3-TE10034.txt'
    data5 = import_data(file_path5)

    # C files
    file_path6 = f'test_vallone/n_{n}/{overlap}/mi/plus/num_data/C11F3-VE.txt'
    data6 = import_data(file_path6)

    file_path8 = f'test_vallone/n_{n}/{overlap}/mi/plus/num_data/C12F3-VE.txt'
    data8 = import_data(file_path8)

    file_path9 = f'test_vallone/n_{n}/{overlap}/mi/plus/num_data/C13F3-VE.txt'
    data9 = import_data(file_path9)

    data = { 'A14F3': data1, 'A15F3':data3, 'A16F3': data5, 
            'C11F3':data6, 'C12F3':data8, 'C13F3':data9}
    
    return data

def corr_import_all_data(n, overlap):
    # A files
    file_path1 = f'test_vallone/n_{n}/{overlap}/corr/num_data/A14F3-TE10034.txt'
    data1 = import_data(file_path1)

    file_path3 = f'test_vallone/n_{n}/{overlap}/corr/num_data/A15F3-TE10034.txt'
    data3 = import_data(file_path3)

    file_path5 = f'test_vallone/n_{n}/{overlap}/corr/num_data/A16F3-TE10034.txt'
    data5 = import_data(file_path5)

    # C files
    file_path6 = f'test_vallone/n_{n}/{overlap}/corr/num_data/C11F3-VE.txt'
    data6 = import_data(file_path6)

    file_path8 = f'test_vallone/n_{n}/{overlap}/corr/num_data/C12F3-VE.txt'
    data8 = import_data(file_path8)

    file_path9 = f'test_vallone/n_{n}/{overlap}/corr/num_data/C13F3-VE.txt'
    data9 = import_data(file_path9)

    data = { 'A14F3': data1, 'A15F3':data3, 'A16F3': data5, 
            'C11F3':data6, 'C12F3':data8, 'C13F3':data9}
    
    return data

def confidence_interval(x, confidence_level):
    lower_percentile = 100 * (1 - confidence_level) / 2
    upper_percentile = 100 * (1 + confidence_level) / 2

    lower_bound = np.percentile(x, lower_percentile)
    upper_bound = np.percentile(x, upper_percentile)
    
    return lower_bound, upper_bound

def mi_single_plot(n, overlap, k, label, c, max_lag, m, ci):
    #plot 
    col = {'A': 'r', 'C': 'b'}

    plt.figure(figsize=(8,6)) 
    lag = np.arange(-max_lag, max_lag+1)

    # Grafico della curva principale
    plt.plot(lag, m, marker='.', markersize=8, label='median', color=col[c])

    # Aggiunta dell'ombreggiatura per la banda di confidenza
    plt.fill_between(lag, ci[:,0], ci[:,1], color=col[c], alpha=0.3, label='90% CI')

    plt.xlabel('lag', size=12)
    plt.ylabel('I', size=12)

    plt.grid()
    plt.title(f'n_{n}/{overlap}/{label}/k_{k}')
    plt.legend()

    file_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/plots/{label}.png"
    plt.savefig(file_path)

def plus_single_plot(n, overlap, label, c, max_lag, m, ci):
    #plot 
    col = {'A': 'r', 'C': 'b'}

    plt.figure(figsize=(8,6)) 
    lag = np.arange(-max_lag, max_lag+1)

    # Grafico della curva principale
    plt.plot(lag, m, marker='.', markersize=8, label='median', color=col[c])

    # Aggiunta dell'ombreggiatura per la banda di confidenza
    plt.fill_between(lag, ci[:,0], ci[:,1], color=col[c], alpha=0.3, label='90% CI')

    plt.xlabel('lag', size=12)
    plt.ylabel('I', size=12)

    plt.grid()
    plt.title(f'n_{n}/{overlap}/{label}/plus')
    plt.legend()

    file_path = f"test_vallone/n_{n}/{overlap}/mi/plus/plots/{label}.png"
    plt.savefig(file_path)

def corr_single_plot(n, overlap, label, c, max_lag, m, ci):
    #plot 
    col = {'A': 'r', 'C': 'b'}

    plt.figure(figsize=(8,6)) 
    lag = np.arange(-max_lag, max_lag+1)

    # Grafico della curva principale
    plt.plot(lag, m, marker='.', markersize=8, label='median', color=col[c])

    # Aggiunta dell'ombreggiatura per la banda di confidenza
    plt.fill_between(lag, ci[:,0], ci[:,1], color=col[c], alpha=0.3, label='90% CI')

    plt.xlabel('lag', size=12)
    plt.ylabel('|R|', size=12)

    plt.grid()
    plt.title(f'n_{n}/{overlap}/{label}')
    plt.legend()

    file_path = f"test_vallone/n_{n}/{overlap}/corr/plots/{label}.png"
    plt.savefig(file_path)


def f_conf(max_lag, matrix):
    confidence_level = 0.90
    ci = np.zeros((2*max_lag+1, 2))
    for j in range(2*max_lag+1):
        ci[j,:] = confidence_interval(matrix[:,j], confidence_level)

    return ci


def mi_plots(n, overlap, k, max_lag):
        directory_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/plots"
        ensure_directory_exists(directory_path)  
        
        data = mi_import_all_data(n, overlap, k)

        alabel = ['A14F3', 'A15F3', 'A16F3']
        clabel = ['C11F3', 'C13F3', 'C12F3']

        am = np.zeros(2*max_lag+1)
        aci = np.zeros((2*max_lag+1, 2))
        for label in alabel:
            matrix = data[label]
            m = np.median(matrix, axis=0)
            am += m
            ci = f_conf(max_lag, matrix)
            aci += ci
            mi_single_plot(n, overlap, k, label, 'A', max_lag, m, ci)

        am /= len(alabel)
        aci /= len(alabel)


        cm = np.zeros(2*max_lag+1)
        cci = np.zeros((2*max_lag+1, 2))
        for label in clabel:
            matrix = data[label]
            m = np.median(matrix, axis=0)
            cm += m
            ci = f_conf(max_lag, matrix)
            cci += ci
            mi_single_plot(n, overlap, k, label, 'C', max_lag, m, ci)

        cm /= len(clabel)
        cci /= len(clabel)

        # plot average A files 
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, am, marker='.', markersize=8, label='average A', color='r')
        plt.fill_between(lag, aci[:,0], aci[:,1], color='r', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('I', size=12)
        plt.title(f'n_{n}/{overlap}/A/k_{k}')
        plt.legend()
        plt.grid()

        file_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/plots/A.png"
        plt.savefig(file_path)

        # plot average C files
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, cm, marker='.', markersize=8, label='average C', color='b')
        plt.fill_between(lag, cci[:,0], cci[:,1], color='b', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('I', size=12)
        plt.title(f'n_{n}/{overlap}/C/k_{k}')
        plt.grid()
        plt.legend()

        file_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/plots/C.png"
        plt.savefig(file_path)

        # plot average A and C 
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, am, marker='.', markersize=8, label='average A', color='r')
        plt.fill_between(lag, aci[:,0], aci[:,1], color='r', alpha=0.3, label='average 90% CI')
        plt.plot(lag, cm, marker='.', markersize=8, label='average C', color='b')
        plt.fill_between(lag, cci[:,0], cci[:,1], color='b', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('I', size=12)
        plt.title(f'n_{n}/{overlap}/AC/k_{k}')
        plt.grid()
        plt.legend()

        file_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/plots/AC.png"
        plt.savefig(file_path)

def plus_plots(n, overlap, max_lag):
        directory_path = f"test_vallone/n_{n}/{overlap}/mi/plus/plots"
        ensure_directory_exists(directory_path)  
        
        data = plus_import_all_data(n, overlap)

        alabel = ['A14F3', 'A15F3', 'A16F3']
        clabel = ['C11F3', 'C13F3', 'C12F3']

        am = np.zeros(2*max_lag+1)
        aci = np.zeros((2*max_lag+1, 2))
        for label in alabel:
            matrix = data[label]
            m = np.median(matrix, axis=0)
            am += m
            ci = f_conf(max_lag, matrix)
            aci += ci
            plus_single_plot(n, overlap, label, 'A', max_lag, m, ci)

        am /= len(alabel)
        aci /= len(alabel)


        cm = np.zeros(2*max_lag+1)
        cci = np.zeros((2*max_lag+1, 2))
        for label in clabel:
            matrix = data[label]
            m = np.median(matrix, axis=0)
            cm += m
            ci = f_conf(max_lag, matrix)
            cci += ci
            plus_single_plot(n, overlap, label, 'C', max_lag, m, ci)

        cm /= len(clabel)
        cci /= len(clabel)

        # plot average A files 
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, am, marker='.', markersize=8, label='average A', color='r')
        plt.fill_between(lag, aci[:,0], aci[:,1], color='r', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('I', size=12)
        plt.title(f'n_{n}/{overlap}/A/plus')
        plt.legend()
        plt.grid()

        file_path = f"test_vallone/n_{n}/{overlap}/mi/plus/plots/A.png"
        plt.savefig(file_path)

        # plot average C files
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, cm, marker='.', markersize=8, label='average C', color='b')
        plt.fill_between(lag, cci[:,0], cci[:,1], color='b', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('I', size=12)
        plt.title(f'n_{n}/{overlap}/C/plus')
        plt.grid()
        plt.legend()

        file_path = f"test_vallone/n_{n}/{overlap}/mi/plus/plots/C.png"
        plt.savefig(file_path)

        # plot average A and C 
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, am, marker='.', markersize=8, label='average A', color='r')
        plt.fill_between(lag, aci[:,0], aci[:,1], color='r', alpha=0.3, label='average 90% CI')
        plt.plot(lag, cm, marker='.', markersize=8, label='average C', color='b')
        plt.fill_between(lag, cci[:,0], cci[:,1], color='b', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('I', size=12)
        plt.title(f'n_{n}/{overlap}/AC/plus')
        plt.grid()
        plt.legend()

        file_path = f"test_vallone/n_{n}/{overlap}/mi/plus/plots/AC.png"
        plt.savefig(file_path)

def corr_plots(n, overlap, max_lag):
        directory_path = f"test_vallone/n_{n}/{overlap}/corr/plots"
        ensure_directory_exists(directory_path)  
        
        data = corr_import_all_data(n, overlap)

        alabel = ['A14F3', 'A15F3', 'A16F3']
        clabel = ['C11F3', 'C13F3', 'C12F3']

        am = np.zeros(2*max_lag+1)
        aci = np.zeros((2*max_lag+1, 2))
        for label in alabel:
            matrix = data[label]
            m = np.median(matrix, axis=0)
            am += m
            ci = f_conf(max_lag, matrix)
            aci += ci
            corr_single_plot(n, overlap, label, 'A', max_lag, m, ci)

        am /= len(alabel)
        aci /= len(alabel)


        cm = np.zeros(2*max_lag+1)
        cci = np.zeros((2*max_lag+1, 2))
        for label in clabel:
            matrix = data[label]
            m = np.median(matrix, axis=0)
            cm += m
            ci = f_conf(max_lag, matrix)
            cci += ci
            corr_single_plot(n, overlap, label, 'C', max_lag, m, ci)

        cm /= len(clabel)
        cci /= len(clabel)

        ### plots ### 

        # plot average A files 
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, am, marker='.', markersize=8, label='average A', color='r')
        plt.fill_between(lag, aci[:,0], aci[:,1], color='r', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('|R|', size=12)
        plt.title(f'n_{n}/{overlap}/A')
        plt.legend()
        plt.grid()

        file_path = f"test_vallone/n_{n}/{overlap}/corr/plots/A.png"
        plt.savefig(file_path)

        # plot average C files
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, cm, marker='.', markersize=8, label='average C', color='b')
        plt.fill_between(lag, cci[:,0], cci[:,1], color='b', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('|R|', size=12)
        plt.title(f'n_{n}/{overlap}/C')
        plt.grid()
        plt.legend()

        file_path = f"test_vallone/n_{n}/{overlap}/corr/plots/C.png"
        plt.savefig(file_path)

        # plot average A and C 
        plt.figure(figsize=(8,6))
        lag = np.arange(-max_lag, max_lag+1)
        plt.plot(lag, am, marker='.', markersize=8, label='average A', color='r')
        plt.fill_between(lag, aci[:,0], aci[:,1], color='r', alpha=0.3, label='average 90% CI')
        plt.plot(lag, cm, marker='.', markersize=8, label='average C', color='b')
        plt.fill_between(lag, cci[:,0], cci[:,1], color='b', alpha=0.3, label='average 90% CI')
        plt.xlabel('lag', size=12)
        plt.ylabel('|R|', size=12)
        plt.title(f'n_{n}/{overlap}/AC')
        plt.grid()
        plt.legend()

        file_path = f"test_vallone/n_{n}/{overlap}/corr/plots/AC.png"
        plt.savefig(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots for the approach of Vallone et al (2016 ).")
    parser.add_argument("--n", type=int, required=True, help="Value for parameter n")
    parser.add_argument("--overlap", choices=['no', 'half'], required=True, help="Specify overlap choice")
    parser.add_argument("--coef", choices=['corr', 'mi', 'plus'], required=True, help="Specify coefficient to analyze data: 'corr', 'mi' and 'plus")
    parser.add_argument("--k", type=int, default=25, help="Value for parameter k")
    
    args = parser.parse_args()
    # n
    n = args.n
    # overlap
    h = args.overlap
    if h == 'half':
        overlap = 'half_overlap'
    elif h == 'no':
        overlap = 'no_overlap'
    # coefficient
    coef = args.coef
    # k for MI algorithm
    k = args.k

    max_lag = 30

    if coef == 'mi':
        mi_plots(n, overlap, k, max_lag)
    elif coef == 'plus':
        plus_plots(n, overlap, max_lag)
    elif coef == 'corr':
        corr_plots(n, overlap, max_lag)