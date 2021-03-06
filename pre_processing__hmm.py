# -*- coding: utf-8 -*-
"""Pre-processing _HMM.py

# Prediciting Q3 protein secondary structure using HMM with pre-processed emission sequence

Two functions have been coded to pre-process the emission sequence. Using this code, one can either use the lookback encoder to mimic an alpha helix interaction of the amino acids, or the palindromic encoder for a beta sheet modelling.
"""

import gzip
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
import random

def get_data(arr, residue_list, q8_list, columns, r, f, bounds=None):
    
    """
    This function retrieves and formats data from the CB6133_filtered and CB531 datasets [1][2]
    Codes is slighlty modified from code provided by [3][4]
    
    [1] Jian Zhou and Olga G. Troyanskaya. Deep supervised and convolutional generative stochastic network for
        protein s
    [2] Jian Zhou and Olga G. Troyanskaya. CB6133 dataset.
        https://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt, 2014.
    [3] Iddo Drori et al. High Quality Prediction of Protein Q8 Secondary Structure by
        Diverse Neural Network Architectures. arXiv preprint arXiv:1811.07143, 2018
    [4] https://github.com/idrori/cu-ssp/blob/master/model_1/model_1.py
    """
    
    if bounds is None: bounds = range(len(arr))
    
    data = [None for i in bounds]
    for i in bounds:
        seq, q8, q3, q2, profiles = '', '', '', '', []
        for j in range(r):
            jf = j*f
            
            # Residue convert from one-hot to decoded
            residue_onehot = arr[i,jf+0:jf+22]
            residue = residue_list[np.argmax(residue_onehot)]

            # Q8 one-hot encoded to decoded structure symbol
            residue_q8_onehot = arr[i,jf+22:jf+31]
            residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

            if residue == 'NoSeq': break      # terminating sequence symbol

            nc_terminals = arr[i,jf+31:jf+33] # nc_terminals = [0. 0.]
            sa = arr[i,jf+33:jf+35]           # sa = [0. 0.]
            profile = arr[i,jf+35:jf+57]      # profile features
            
            seq += residue # concat residues into amino acid sequence

            #encode q3 structure
            if residue_q8 in 'GHI':
                q3 += 'H'
                q2 += 'A'
            elif residue_q8 in 'TBSL':
                q3 += 'C'
                q2 += 'X'
            elif residue_q8 in 'E':
                q3 += 'E'
                q2 += 'X'
            else:
                q3 += 'Z'
                q2 += 'Z'
            
            q8  += residue_q8 # concat secondary structure into secondary structure sequence
            profiles.append(profile)
        
        data[i] = [str(i+1), len(seq), seq, np.array(profiles), q8, q3, q2]
    
    return pd.DataFrame(data, columns=columns)


def encode_sequence(sequence, code):
    
    """
    Provided an input sequence and a code, returns the encoding of the sequence
    """
    
    encoded_seq = []
    
    for x in sequence:
        try:
            idx = code[x]
            encoded_seq.append(idx)
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return encoded_seq


def format_dataset(df, emission_code, state_code, exp_col="q3_expected"):
    
    """
    Provided a dataframe which contains the amino sequences and the hidden sequence,
    this function encodes those sequences according to the provided codes
    and return them
    
    *exp_col specifies if want to encode the q8, q3, or q2 hidden sequence
    """
    
    assert ('id' in df.columns and 'len' in df.columns and 'input' in df.columns and exp_col in df.columns)
    
    formattedDF = pd.DataFrame(columns=['id','len','input','expected'])

    for i in range(len(df)):
        
        sid = df.iloc[i].id
        slen = df.iloc[i].len
        enc_input = encode_sequence(df.iloc[i].input, emission_code)
        enc_expected = encode_sequence(df.iloc[i][exp_col], state_code)
        
        assert (len(enc_input) == len(enc_expected))
        
        formattedDF = formattedDF.append({'id':sid, 'len':slen, 'input':enc_input, 'expected':enc_expected}, ignore_index=True)

    return formattedDF

def estimate_transition_matrix(df, state_code):
    """
    Given a dataframe that has the data for the amino sequences and their corresponding hidden sequence,
    we use the data to compute the MLEs of the emission probablities.
    
    ex. estimated P(emission=A|state=H) = count(emission=A,state=H) / sum_over_all_emission count(emission, state=H)
    
    *implemented a pseudocount of +1 for cases where we have 0 observations of a certain (emission,state) combo
    """
    
    n_states = len(state_code)
    
    #using pseudocount of +1
    counts = np.ones(shape=(n_states, n_states), dtype=float)
    
    for i in range(len(df)):
        
        state_seq = df.iloc[i].expected
        seq_len = len(df.iloc[i].expected)
        
        for j in range(seq_len - 1):
            
            x = state_seq[j]
            y = state_seq[j+1]
            
            counts[x,y] += 1
    
    #transform counts to probability by normalizing of row sums
    row_sums = np.sum(counts, axis=1)
    T = counts / row_sums.reshape((-1,1))
    
    return T


def estimate_emission_matrix(df, state_code, emission_code):
    """
    Given a dataframe that has the data for the amino sequences and their corresponding hidden sequence,
    we use the data to compute the MLEs of the transition probablities.
    
    ex. estimated P(state_{t+1}=E|state_{t}=H) = count(state_{t}=H, state_{t+1}=E) / sum_over_all_states count(state_{t}=H, state_{t+1})
    
    *implemented a pseudocount of +1 for cases where we have 0 observations of a certain (state,state) combo
    """
    
    n_states = len(state_code)
    n_emissions = len(emission_code)
    
    #using pseudocount of +1
    #Steve Contribution
    #Change counts matrix to be n_states x n_emissions x n_emissions
    #Each state has a n_emissions x n_emissions context-dependent matrix associated with it
    counts = np.ones(shape=(n_states, n_emissions, n_emissions), dtype=float)
    
    for i in range(len(df)):
        
        state_seq = df.iloc[i].expected
        #emission_seq now of form: [(1,4),(2,5),...]
        #store context in y
        #store new state in z
        emission_seq = df.iloc[i].input
        #df['len'] no longer reflects true length
        seq_len = len(df.iloc[i].input)
        
        for j in range(seq_len):
            
            x = state_seq[j]
            y = emission_seq[j][0]
            z = emission_seq[j][1]
            
            counts[x,y,z] += 1

    #transform counts to probability by normalizing of row sums
    #not sure how pendo's way works, implementing my own here
    #index y is 'context', normalize by this row
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            counts[i,j,:] = counts[i,j,:]/np.sum(counts[i,j,:])
    
    E = counts
    
    #row_sums = np.sum(counts, axis=1)
    #print(row_sums.shape)
    #E = counts / row_sums.reshape((-1,3))

    return E

def start_distribution(df,state_code):
    """
    Given a dataframe that has the data for the amino sequences and their corresponding hidden sequence,
    we use the data to compute the MLEs of the start distribution.
    
    ex. estimated P(state=H) = count(state=H) / sum_over_all_states count(state)
    
    *implemented a pseudocount of +1 for cases where we have 0 observations of a certain state
    """
    
    n_states = len(state_code)
    
    #using pseudocount of +1
    counts = np.array([1.0] * n_states)
    
    for i in range(len(df)):
        
        state_seq = df.iloc[i].expected
        seq_len = len(df.iloc[i].expected)
        
        for j in range(seq_len):
            
            x = state_seq[j]
            
            counts[x] += 1
    
    #transform counts to probability by normalizing of row sums
    total = sum(counts)
    pi = counts / total
    
    return pi

def viterbi_decoding(T,E,pi,seq):
    """
    This functions performs viterbi decoding to get the predicted hidden sequence,
    given the input emission sequence as well as the transition matrix, emission matrix,
    and the start distribution
    """
    
    #sequence length
    N = len(seq)
    #num of states
    M = T.shape[0]
    
    assert (M == len(pi))
    
    #V will store viterbi values
    V = np.zeros(shape=(M, N), dtype=float)
    
    #P will store prev state from which we transitioned into state m and time n to achieve the max value of V[m,n]
    #(i.e. pointer to help us reconstruct sequence after predicting most probable path in viterbi graph)
    P = np.empty(shape=(M, N))
    P[:] = np.NaN
    
    #populate viterbi matrix
    for n in range(N):
        
        #get current emissions
        e = seq[n]
        
        for m in range(M):
            
            #initilize viterbi value for current timestep given state m to be -infty
            maxV = float("-inf")
            prev = np.NaN
            
            #get log prob. of emission given state m
            emiss_logp = np.log(E[m,e[0],e[1]])
            
            #start of sequence
            if n == 0:
                start_logp = np.log(pi[m])
                maxV = emiss_logp + start_logp
                
            else:
            
                #solve for max value for V[m,n]
                for i in range(M): 

                    #get previous timestep viterbi value for state i (which should be a log prob)
                    prev_vit = V[i, n-1]

                    #get log prob of transition from state i to m
                    trans_logp = np.log(T[i,m])

                    #update viterbi value for current timestep given state m
                    curV = prev_vit + trans_logp + emiss_logp

                    if curV > maxV:
                        maxV = curV
                        prev = i
        
            V[m,n] = maxV
            P[m,n] = prev
    
    #initialize with state with highest probability at end of sequence
    best_path = [np.argmax(V[:,-1])]
    
    #work backwards to reconstruct sequence
    for n in range(N-1,0,-1):
        
        #determine where we are
        cur_state = best_path[0]

        #find state from which we came that yielded highest probability to current state at current time
        prev_state = int(P[cur_state,n])

        #prepend previous state
        best_path = [prev_state] + best_path
    
    return best_path


def getPredictions(df, T, E, pi):
    
    """
    get predictions for all sequences in specified dataset using provided HMM
    """
    
    results = pd.DataFrame(columns=['input','predicted','expected'])
    
    for i in range(len(df)):
        
        #change amino_seq input
        amino_seq = df.iloc[i].input
        pred_seq = viterbi_decoding(T,E,pi,amino_seq)
        exp_seq = df.iloc[i].expected
        
        #Steve commented this out because we are padding our start with new symbols
        #assert(len(amino_seq) == len(pred_seq) and len(amino_seq) == len(exp_seq))
        
        results = results.append({'input':amino_seq, 'predicted':pred_seq, 'expected':exp_seq}, ignore_index=True)
    
    return results

def HMMaccuracy(df,q=3):
    
    """
    Compute accurcay of HMM given a dataframe the has the input emission sequences,
    the predicted hidden sequences, and the actual hidden sequences
    
    *q specifies whether the predicion was made for q2, q3, or q8 protein structure
    """
    
    #row represents expected state
    #col represents predicted state
    counts = np.zeros(shape=(q,q), dtype=int)
    
    for i in range(len(df)):

        #get predicted and expected hidden sequence from dataframe
        pred = df.iloc[i].predicted
        exp = df.iloc[i].expected
        
        #assert (len(pred) == len(exp))
        
        for j in range(len(pred)):
            
            x = exp[j]
            y = pred[j]
            counts[x,y] += 1
    
    rowSum = np.sum(counts, axis=1)
    colSum = np.sum(counts, axis=0)
    
    #true positive (negative) / total predicted positive (negative)
    precision = np.array([counts[i,i] / colSum[i] for i in range(q)])
    
    #true positive (negative) / total actual positive (negative)
    recall = np.array([counts[i,i] / rowSum[i] for i in range(q)])
    
    accuracy = 0
    for i in range(q):
        accuracy += counts[i,i]
    accuracy = accuracy / sum(rowSum)
    
    
    return accuracy, precision, recall, counts

#Pre-processing functions for emissions

def encode_lookback_context(sequence,spacing = 4):
  '''Encode context by keeping track of the i-spacing residue'''
  encoded_sequence = []
  for i in range(spacing,len(sequence)):
      encoded_sequence.append((sequence[i-spacing],sequence[i]))
  return encoded_sequence

def encode_palindrome_context(sequence, length):
  '''
  Encode context by associating amino acids around a turn together. This function supposes that each strand is of fixed length, and then turns.
  '''
  encoded_sequence = []
  for i in range(length,len(sequence)):
    encoded_sequence.append((sequence[i-2*(i%length)],sequence[i]))
  return encoded_sequence


'Examples of how the functions work'
l = range(1,20)
'Example of lookback encoding'
#print(encode_lookback_context(l))

'Example of palindrome encoding'
#There is a turn every 5 amino acids. The second number of the tuple is always increasing at a constant rate.
#print(encode_palindrome_context(l,5))

"""Step 1: Load data"""

#seed so get consistent results for every run
random.seed(0)

cb513 = np.load(dir+'/cb513+profile_split1.npy.gz')
cb6133filtered = np.load(dir+'/cullpdb+profile_5926_filtered.npy.gz')
print("Data Loaded")
print(f"CB6133 shape: {cb6133filtered.shape}")
print(f"CB513 shape: {cb513.shape}")

"""### Step 2: Process Data"""

maxlen_seq = r = 700 # protein residues padded to 700
f = 57  # number of features for each residue

residue_list = list('ARNDCEQGHILKMFPSTWYVX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']
q3_list      = list('HCE') + ['NoSeq']
q2_list      = list('AX') + ['NoSeq']

columns = ["id", "len", "input", "profiles", "q8_expected", "q3_expected", "q2_expected"]

print("Turning data arrays into dataframes")

# train, test split
train_df = get_data(cb6133filtered, residue_list, q8_list, columns, r, f)
test_df  = get_data(cb513, residue_list, q8_list, columns, r, f)

"""### Step 3: Encode Sequences and Format DataFrames
    (a) Create codes to encode emission and hidden sequences
    (b) Apply encodings & specify hidden sequence of interest (q2, q3, q8)
"""

emission_code = {residue_list[i]:i for i in range(len(residue_list)-1)}
emission_code['$'] = 21
state_code = {q3_list[i]:i for i in range(len(q3_list)-1)}

print("emission_code:")
for k,v in emission_code.items():
    print(f"{k}:{v}", end=" ")

print("\n\nstate_code:")
for k,v in state_code.items():
    print(f"{k}:{v}", end=" ")

train_df_formatted = format_dataset(train_df, emission_code, state_code, 'q3_expected')
test_df_formatted = format_dataset(test_df, emission_code, state_code, 'q3_expected')

print("Encoding sequences")
size = 4

#make a copy and encode context
train_df_context = train_df_formatted.copy()
test_df_context = test_df_formatted.copy()


#pad beginning by '$' start symbols determined by lookback/palindrome length
train_df_context['input'] = train_df_context['input'].apply(lambda x : [21 for i in range(size-1)] + x) 
test_df_context['input'] = test_df_context['input'].apply(lambda x : [21 for i in range(size-1)] + x)

#encodes context and truncates expected vals
#replace with desired encoding function

train_df_context['input'] = train_df_context['input'].apply(lambda x :  encode_lookback_context(x,size))
test_df_context['input'] = test_df_context['input'].apply(lambda x :  encode_lookback_context(x,size))

"""### Step 4: Estimate HMM=(T, E, pi) using Our Data

"""

print("Computing initial estimates for transition and emission matrices using training data")
start = time.time()
T = estimate_transition_matrix(train_df_context, state_code)
E = estimate_emission_matrix(train_df_context, state_code, emission_code)
pi = start_distribution(train_df_context,state_code)
end = time.time()
print(f"Time to estimate T, E, pi is approx: {round((end-start)//60,4)} minutes")

"""### Step 5: Compare HMM against Prior Research [5]

    Some slight difference is expected because they were only able to train and test on CB531 whereas we will be training on CB6113 and testing on CB531.

    [5] W. Ding, D. Dai, J. Xie, H. Zhang, W. Zhang and H. Xie, "PRT-HMM: A Novel Hidden Markov Model for Protein Secondary Structure Prediction," 2012 IEEE/ACIS 11th International Conference on Computer and Information Science, 2012, pp. 207-212, doi: 10.1109/ICIS.2012.89.

    https://ieeexplore-ieee-org.ezproxy.cul.columbia.edu/stamp/stamp.jsp?tp=&arnumber=6211098

"""

print("Start Distribution (H,C,E) x (H,C,E):")
print(pi.round(decimals=4))

#compare start distribution to source
pi_source = np.array([0.3496, 0.4405, 0.2100] )
pi_delta = (pi - pi_source).round(decimals=4)
pi_delta

print("Transition Matrix (H,C,E) x (H,C,E):")
print(T.round(decimals=4))

#compare transition matrix to source
T_source = np.array( \
    [[0.8937, 0.1036, 0.0027], \
     [0.0810, 0.8297, 0.0893], \
     [0.0091, 0.1801, 0.8108 ]]
    )

T_delta = (T - T_source).round(decimals=4)
T_delta

print("Emissions Matrix (H,C,E) x (H,C,E):")
print(E.round(decimals=4))

"""### Step 6: Compute HMM Prediction Performance on Train Data

---

    We can compare against performance of traditional HMM from [5] as sanity check:
        Overall Accuracy: 44.38%
        Helix Accuracy (H): 90.46%
        Beta-Sheet Accuracy (E): 4.56%
        Coil Accuracy (C): 28.05%
        
     *Some slight difference is expected because they were only able to train and test on CB531 whereas we will be training on CB6113 and testing on CB531.
    
"""

#make predictions
start = time.time()
train_predictions = getPredictions(train_df_context, T, E, pi)
train_acc, train_prec, train_rec, train_cnts = HMMaccuracy(train_predictions, q=3)
end = time.time()
print(f"Time predict on training data is approx: {round((end-start),2)} seconds")

#Our HMM performance

print(f"Accuracy: {round(train_acc,4)} \n")
print(f"Precision (H,C,E):\n\t {train_prec.round(decimals = 4)} \n")
print(f"Recall (H,C,E):\n\t {train_rec.round(decimals = 4)} \n")
print("Counts (H,C,E) x (H,C,E):")
print(train_cnts)

"""### Step 7: Compute HMM Performance on Test Data

    We can compare against performance of traditional HMM from [5] as sanity check:
            Overall Accuracy: 44.38%
            Helix Accuracy (H): 90.46%
            Beta-Sheet Accuracy (E): 4.56%
            Coil Accuracy (C): 28.05%
        
     *Some slight difference is expected because they were only able to train and test on CB531 whereas we will be training on CB6113 and testing on CB531.
"""

start = time.time()
test_predictions = getPredictions(test_df_context, T, E, pi)
test_acc, test_prec, test_rec, test_cnts = HMMaccuracy(test_predictions, q=3)
end = time.time()
print(f"Time predict on test data is approx: {round((end-start),2)} seconds")

print(f"Accuracy: {round(test_acc,4)} \n")
print(f"Precision (H,C,E):\n\t {test_prec.round(decimals = 4)} \n")
print(f"Recall (H,C,E):\n\t {test_rec.round(decimals = 4)} \n")
print("Counts (H,C,E) x (H,C,E):")
print(test_cnts)
