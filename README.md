# MN_module8

**Numerical Methods for Nonlinear Physics - Module 8. Pisa University**

This repository contains code and resources developed for **Module 8** of the course "Numerical Methods for Physics". The project focuses on statistical analysis, mutual information estimation, and synthetic data generation.

---

## **Features**
- **Data Generation**: Tools for generating univariate and multivariate synthetic datasets.
- **Mutual Information Calculation**: Implementation of algorithms to calculate mutual information (MI) with k-nearest neighbors.
- **Data Checking Utilities**: Scripts for validating and visualizing datasets.

---

## **Project Structure**

# Repository Structure: MN_module8

**Top-Level Directories**  
- `core/` : Scripts for Mutual Information (MI) calculation.  
- `sampling/` : Scripts for synthetic data generation.  
- `utils/` : Utility scripts and helper functions.  
- `tests/` : Unit tests for core functionalities.  
- `old_repo_architecture/` : Archived repository structure files (until 12/12/2024).  

---

## **Directory Breakdown**

### `core/`  
Scripts for computing Mutual Information (MI):  
- `calculate_mi_algorithm1.py` : MI calculation using Algorithm 1 from Kraskov et al.  
- `calculate_mi_algorithm1_entropies_sum.py` : Summing entropies for MI estimation.  
- `calculate_mi_algorithm2.py` : MI calculation using Algorithm 2 from Kraskov et al.   

### `sampling/`  
Scripts for generating synthetic datasets:  
- `generate_data.py` : Main script for data generation.  
- `generate_multivariate_data.py` : Generates bivariate datasets.  
- `generate_univariate_data.py` : Generates a dataset composed by two univariate correlated series.  

### `utils/`  
Utility scripts to support core tasks:  
  
- `config_utils.py`: Handles configuration files 
- `decorators.py`:
- `io_utils.py`: handles reading, writing, managing data files efficiently and manages input/output operations.
- `math_utils.py`: Contains helper functions for mathematical operations
- `mutual_information_utils.py`: Utility functions to support MI computation algorithms
- `plot_utils.py` : Functions for visualizing data (e.g., line charts, scatter plots, histograms).
- `pre_processing_utils.py`: Pre-processes data (normalization, cleaning) before analysis.

 

### `tests/`  
Unit tests to ensure code reliability:  



### **Other Files**  
- `calculate_mi_algorithm1.py`: Top-level MI script
- `calculate_mi_algorithm1_entropies_sum.py`: Top-level MI script
- `calculate_mi_algorithm2.py`: Top-level MI script
- `checking_data.py` : Validates and checks data integrity.
- `generate_data.py` : Top-level data generation script
- `generate_multivariate_data.py ` : Top-level data generation script
- `generate_univariate_data.py` : Top-level data generation script
- `config.json`: Configuration file
- `.gitignore` : Specifies files and directories ignored by Git.  
- `LICENSE` : Project licensing information (GPL-3.0).  
- `README.md` : Main documentation for the repository.  


---