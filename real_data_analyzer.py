import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp



def two_sample_ks(mi_A, mi_C, alpha):
    # Two-sample Kolmogorov-Smirnov test
    test_statistics, p_value = ks_2samp(mi_A, mi_C)

    # Results
    print("Statistic KS:", test_statistics)
    print("p-value:", p_value)

    if p_value < alpha:
        print(f"The two samples have significantly different distributions (at the {int(alpha*100)} % significance level).")
    else:
        print("There is not enough evidence to assert that the two samples come from different distributions.")


def import_real_data(file_path_A, file_path_C):
    # file A loading as numpy array
    try:

        mi_A = np.loadtxt(file_path_A)
        
    except Exception as e:
        print(f"Error while loading file: {e}")

    # file C loading as numpy array
    try:
        
        mi_C = np.loadtxt(file_path_C)
        
    except Exception as e:
        print(f"Error while loading file: {e}")

    return mi_A, mi_C


if __name__ == "__main__":
    # choose data to analyze
    file_path_C = 'data/real_data/n_5000/h_2500/k_1/C/mi_C.txt'
    file_path_A = 'data/real_data/n_5000/h_2500/k_1/A/mi_A.txt'

    mi_A, mi_C = import_real_data(file_path_A, file_path_C)

    plt.figure(figsize=(9,6))
    plt.hist(mi_A, density=True, bins=int(np.sqrt(len(mi_A))), label='A')
    plt.hist(mi_C, density=True, bins=int(np.sqrt(len(mi_C))), label='C', alpha=0.6)

    plt.legend(fontsize=15)
    plt.ylabel('density', size=15)
    plt.xlabel('MI', size=15)
    # choose the file name
    plt.savefig('test.pdf')
    plt.show()

    # Two-sample Kolmogorov-Smirnov test
    alpha = 0.05
    two_sample_ks(mi_A, mi_C, alpha)
