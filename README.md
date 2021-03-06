# CSHMM_ComputationalGenomics
Authors: Pendo Abbo (pa2451), Mujahed Darwaza (md4028), Steven Harrelson (sh3379)

The code listed in this repo is a final course project for CBMFW4791 Computational Genomics at Columbia University, taught by professor Itsik Pe'er. The goal of this project was to use a variety of HMM architectures to predict protein secondary structure from primary structure emissions. 

The dataset used is from the cap7 protein structure classification competition provided by (https://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt). More information can be found in the _Data_ file. Sample data has been provided in the _train_sample.npy_ and _test_sample.npy_ files.

There are two main scripts. To run each of these codes, simply run them in the same folder as the downloaded datasets specified in the data folder of this repo.

**pre-processing_hmm.py** Tests multiple pre-processing techniques on the emissions of the HMM. You can either run it using the lookback pre-processing or the palindromic pre-processing. Lines 475-482 should be modified if one wishes to use palindromic pre-processing instead of lookback. The "encode_lookback_context" should be replaced by "encode_palindromic_context" and the initial padding should go up to size instead of size-1.
   
    #to run on sample data change lines 443 and 444 to the following:
        cb513 = np.load('test_sample.npy')
        cb6133filtered = np.load('train_sample.npy')


**sliding_window_hmm.py** For a value of k=1, this code runs the traditional HMM architecture on the dataset. For a k>1, it runs a pre-processing of the secondary structure such that the trailing k secondary structure labels ending at position t were used to express the hidden state of position t. The code can be run using the following command:
    
    python3 sliding_window.py <k> <eval>
    
    #k: window size, integer
    
    #eval: inidicates the evaluation set. input 'dev' to break out 10% of training data to be used as evaluation set. input 'test' to use test set for evaluation
    
    #to run on sample data change lines 450 and 451 to the following:
        cb513 = np.load('test_sample.npy')
        cb6133filtered = np.load('train_sample.npy')

Not that the decoding times for the sliding window approach as the windodw size, k, expands.

<img width="166" alt="image" src="https://user-images.githubusercontent.com/88948596/167264499-13cf031e-aecc-42e3-b71e-0ec871d676bc.png">

**System Requirements**

Python3

_*No additional computing power required._

**Required Libraries**

numpy, pandas, pickle, sklearn, tqdm
