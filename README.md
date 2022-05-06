# CSHMM_ComputationalGenomics

# Introduction
The code listed in this repo is a final course project for CBMFW4791 computational genomics at Columbia university, taught by professor Itsik Pe'er.  The goal of this project was to use a variety of HMM architectures to predict protein secondary structure from primary structure emissions.  We first collected data from the cap7 protein structure classification competition provided by (https://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt).  


We developed a traditional HMM architecture and verified that it predicted protein secondary structure at a level consistent with what is found in the literature.  The code to run this traditional HMM is 'traditional_hmm.ipynb'

We modified this approach by performing a pre-processing of the dataset to encode context into amino acid emissions.  There were two approaches to encoding this information.  One is called the 'fixed lookback' approach (Fixed_Lookback.ipynb).  This attempts to model the structure of alpha helicies by encoding information about the amino acid residue emitted four emissions in the past.  This is relevant because in alpha helix structure, there is a four fold amino acid periodicity, where an amino acid is spatially adjacent to an amino acid four emissions in the past.  The second is called the 'palindromic lookback' approach.  This attempts to model beta sheet structure by encoding emission information centered around a hinge point.  To run each of these codes, simply run them in the same folder as the downloaded datasets specified in the data folder of this repo.

The final and most successful approach was to encode context on the hidden states instead of the emissions (DeBruijn_state_context.ipynb).  This was inspired by De Bruijn graph decoding, where we cycle through groups of hidden states.  To run this code, download and run it in the same folder as teh datasets specified in the data folder of this repo.



