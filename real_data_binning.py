from real_data_mi import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/')))
from core.mutual_information_binning import mutual_information_binning_adaptive


def real_data_binning(data_dict, name_dict, n, h, k_bins):
    for name in data_dict.keys():
        start = time.time()
        # data
        data_array = data_dict[name]

        # number of windows
        i = 0
        while (i*h + n) <= len(data_array):
            i+=1
        i_max = i  # actually we are interested in i-1, but for the subquent for cycle is reasonable to save i_max=i
        logging.info(f'\n number of windows for {name} = {i_max}')
        # mi computation
        mi_array = np.zeros(i_max)

        for j in range(i_max): # j=0,...,i_max-1
            # compute mi for each window
            mi_array[j] = mutual_information_binning_adaptive(data_array[j*h: (j*h + n)], num_bins=k_bins)[0]


        # save reuslts
        directory_path = f'data/real_data/n_{n}/h_{h}/binning/k_bins_{k_bins}/{name_dict}'
        ensure_directory_exists(directory_path)
        file_name = f"mi_{name}.txt"
        file_path = os.path.join(directory_path, file_name)
        np.savetxt(file_path, mi_array)

        end = time.time() - start 
        logging.info(f'\n run time for {name} = {end} sec \n \n')


if __name__ == "__main__":

    setup_logger()
    logging.getLogger('core.mutual_information_binning').setLevel(logging.WARNING)
    logging.getLogger('utils.core_utils').setLevel(logging.WARNING)

    # directory path
    directory_path = "data/real_data/raw_data"

    # import raw data
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}
    
    parser = argparse.ArgumentParser(description="Process real data with given parameters.")
    parser.add_argument("--n", type=int, default=5000, help="Value for parameter n (default: 5000)")
    parser.add_argument("--overlap", choices=['no', 'half'], required=True, help="Specify overlapping mode: 'no' or 'half'")
    parser.add_argument("--k", type=int, default=24, help="Value for parameter k_bins (default: 24)")
    
    args = parser.parse_args()
    n = args.n
    h = n if args.overlap == 'no' else n // 2
    k_bins = args.k
    logging.info(f"\nParameters: n={n}, h={h}, k_bins(nominal)={k_bins}")
        
    real_data_binning(raw_data['A'], 'A', n, h, k_bins)
    real_data_binning(raw_data['C'], 'C', n, h, k_bins)


