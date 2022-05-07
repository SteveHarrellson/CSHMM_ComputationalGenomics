# CSHMM_ComputationalGenomics
The code listed in this repo is a final course project for CBMFW4791 Computational Genomics at Columbia University, taught by professor Itsik Pe'er. The goal of this project was to use a variety of HMM architectures to predict protein secondary structure from primary structure emissions.  The dataset used is from the cap7 protein structure classification competition provided by (https://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt).  

There are two python codes in this repository.
The first is Pre-processing_HMM.py. This code tests multiple pre-processing techniques on the emissions of the HMM. You can either run it using the lookback pre-processing or the palindromic pre-processing.
The second file is Sliding_window.py. For a value of k=1, this code runs the traditional HMM architecture on the dataset. For a k>1, it runs a pre-processing of the secondary structure such that the trailing k secondary structure labels ending at position t were used to express the hidden state of position t.

To run each of these codes, simply run them in the same folder as the downloaded datasets specified in the data folder of this repo.
No additional libraries are required for this project. The codes do not require additional computing power.
