###This code builds the structure for the sliding-window states HMM.
###For a length k=1, this is equivalent to a traditional HMM.

import gzip
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
import random
from itertools import product
from tqdm import tqdm
from collections import defaultdict
import sys

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
            elif residue_q8 in 'E':
                q3 += 'E'
                q2 += 'X'
            else:
                q3 += 'C'
                q2 += 'X'
            
            q8  += residue_q8 # concat secondary structure into secondary structure sequence
            profiles.append(profile)
        
        data[i] = [str(i+1), len(seq), seq, np.array(profiles), q8, q3, q2]
    
    return pd.DataFrame(data, columns=columns)


def create_context_code(symbols, context):
    
    context_encode = {}
    context_decode = {}
    
    i=0
    for comb in product(symbols, repeat=context):
        
        x = ''.join(comb)
        context_encode[x] = i
        context_decode[i] = x
        
        i += 1
        
    
    for j in range(1,context):
        for comb in product(symbols, repeat=(context-j)):
            x = '$'*j + ''.join(comb)
            context_encode[x] = i
            context_decode[i] = x

            i += 1
    
    return context_encode, context_decode
    

def encode_sequence(sequence, code, context):
    
    """
    Provided an input sequence and a code, returns the encoding of the sequence
    """
    
    encoded_seq = []
    
    if context > 1:
        seq = '$' * (context-1)
        seq = seq + sequence
    else:
        seq = sequence
        
    for i in range(context, len(seq)+1):
        
        x = seq[i-context:i]
    
        try:
            idx = code[x]
            encoded_seq.append(idx)
        except Exception as e:
            print(f"Error: {e}")
            break
    
    assert len(encoded_seq) == len(sequence)
    
    return encoded_seq


def decode_sequence(sequence, code):
    
    decoded_seq = ''
    
    for i in range(len(sequence)):
        
        x = sequence[i]
        x = code[x]
        
        decoded_seq += x[-1]
        
    
    return decoded_seq
        

def format_dataset(df, emission_code, state_code, emission_context, state_context, exp_col="q3_expected"):
    
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
        enc_input = encode_sequence(df.iloc[i].input, emission_code, emission_context)
        enc_expected = encode_sequence(df.iloc[i][exp_col], state_code, state_context)
        
        assert (len(enc_input) == len(enc_expected))
        
        formattedDF = formattedDF.append({'id':sid, 'input':enc_input, 'expected':enc_expected}, ignore_index=True)

    return formattedDF

def estimate_transition_matrix(df, state_code, possible_priors):
    """
    Given a dataframe that has the data for the amino sequences and their corresponding hidden sequence,
    we use the data to compute the MLEs of the emission probablities.
    
    ex. estimated P(emission=A|state=H) = count(emission=A,state=H) / sum_over_all_emission count(emission, state=H)
    
    *implemented a pseudocount of +1 for cases where we have 0 observations of a certain (emission,state) combo
    """
    
    n_states = len(state_code)
    
    counts = np.zeros(shape=(n_states, n_states), dtype=int)
    
    #using pseudocount of +1 for only those transitions that are actually possible
    for y, priors in possible_priors.items():
        for x in priors:
            counts[x,y] = 1
    
    for i in range(len(df)):
        
        state_seq = df.iloc[i].expected
        seq_len = len(state_seq)
        
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
    counts = np.ones(shape=(n_states, n_emissions), dtype=int)
    
    for i in range(len(df)):
        
        state_seq = df.iloc[i].expected
        emission_seq = df.iloc[i].input
        seq_len = len(emission_seq)
        
        for j in range(seq_len):
            
            x = state_seq[j]
            y = emission_seq[j]
            
            try:
                counts[x,y] += 1
            except:
                print(x,y)

    #transform counts to probability by normalizing of row sums
    row_sums = np.sum(counts, axis=1)
    E = counts / row_sums.reshape((-1,1))
    
    return E

def start_distribution(df, state_symbols, state_decode):
    """
    Given a dataframe that has the data for the amino sequences and their corresponding hidden sequence,
    we use the data to compute the MLEs of the start distribution.
    
    ex. estimated P(state=H) = count(state=H) / sum_over_all_states count(state)
    
    *implemented a pseudocount of +1 for cases where we have 0 observations of a certain state
    """
    
    n_states = len(state_symbols)
    
    #using pseudocount of +1
    counts = np.array([1.0] * n_states)
    
    for i in range(len(df)):
        
        state_seq = df.iloc[i].expected
        seq_len = len(state_seq)
        
        for j in range(seq_len):
            
            x = state_seq[j]
            x = state_decode[x][-1]
            idx = state_symbols[x]
            
            counts[idx] += 1
    
    #transform counts to probability by normalizing of row sums
    total = sum(counts)
    pi = counts / total
    
    return pi

def viterbi_decoding(T,E,pi,seq,state_symbols, state_decode, possible_priors):
    """
    This functions performs viterbi decoding to get the predicted hidden sequence,
    given the input emission sequence as well as the transition matrix, emission matrix,
    and the start distribution
    """
    
    #sequence length
    N = len(seq)
    #num of states
    M = T.shape[0]
    
    assert (M == len(state_decode) and len(pi) == len(state_symbols))
    
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
            emiss_logp = np.log(E[m,e])
            
            #start of sequence
            if n == 0:
                
                sym = state_decode[m][-1]
                sym_idx = state_symbols[sym]
                start_logp = np.log(pi[sym_idx])
                maxV = emiss_logp + start_logp
                
            else:
            
                #solve for max value for V[m,n]
                #only want to consider those states from which it is possible to transition into m
                #this is done to save time, but can iterate over all possible M to get same results
                for i in possible_priors[m]: 

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


def getPredictions(df, T, E, pi, state_symbols, state_decode, possible_priors):
    
    """
    get predictions for all sequences in specified dataset using provided HMM
    """
    
    results = pd.DataFrame(columns=['input','predicted','expected'])
    
    for i in tqdm(range(len(df)), desc="Decoding sequences"):
        
        amino_seq = df.iloc[i].input
        pred_seq = viterbi_decoding(T,E,pi,amino_seq,state_symbols, state_decode, possible_priors)
        exp_seq = df.iloc[i].expected
        
        assert(len(amino_seq) == len(pred_seq) and len(amino_seq) == len(exp_seq))
        
        results = results.append({'input':amino_seq, 'predicted':pred_seq, 'expected':exp_seq}, ignore_index=True)
    
    return results


def HMMaccuracy(df,state_decode, state_symbols):
    
    """
    Compute accurcay of HMM given a dataframe the has the input emission sequences,
    the predicted hidden sequences, and the actual hidden sequences
    
    *q specifies whether the predicion was made for q2, q3, or q8 protein structure
    """
    
    #row represents expected state
    #col represents predicted state
    q = len(state_symbols)
    counts = np.zeros(shape=(q,q), dtype=int)
    
    for i in range(len(df)):

        #get predicted and expected hidden sequence from dataframe
        pred = df.iloc[i].predicted
        pred = decode_sequence(pred, state_decode)
        exp = df.iloc[i].expected
        exp = decode_sequence(exp, state_decode)
        
        assert (len(pred) == len(exp))
        
        for j in range(len(pred)):
            
            x = state_symbols[exp[j]]
            y = state_symbols[pred[j]]
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
       
       
if __name__ == '__main__':

    state_context = int(sys.argv[1])
    emission_context = 1
    evaluation = sys.argv[2]
    log = open(f'log_{state_context}_{evaluation}', 'w')

    ### Step 1: Load Data

    #seed so get consistent results for every run
    random.seed(0)

    print("Stage 1: Loading data...")
    log.write("Stage 1: Loading data... \n")
    cb513 = np.load('cb513+profile_split1.npy.gz')
    cb6133filtered = np.load('cullpdb+profile_5926_filtered.npy.gz')
    log.write("Data Loaded \n")
    log.write(f"CB6133 shape: {cb6133filtered.shape} \n")
    log.write(f"CB513 shape: {cb513.shape} \n")


    ### Step 2: Process Data and split into train, eval sets

    print("Stage 2: Processing data and splitting into train and eval")
    log.write("\nStage 2: Processing data and splitting into train and eval \n")

    maxlen_seq = r = 700 # protein residues padded to 700
    f = 57  # number of features for each residue

    residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
    q8_list      = list('LBEGIHST') + ['NoSeq']
    q3_list      = list('HCE') + ['NoSeq']
    q2_list      = list('AX') + ['NoSeq']

    columns = ["id", "len", "input", "profiles", "q8_expected", "q3_expected", "q2_expected"]

    # train, eval split
    if evaluation == 'dev':
        # break out 10% of train data to be used as eval set
        train_df, eval_df = train_test_split(get_data(cb6133filtered, residue_list, q8_list, columns, r, f), test_size=0.1, random_state=11)
    else:
        train_df = get_data(cb6133filtered, residue_list, q8_list, columns, r, f)
        eval_df  = get_data(cb513, residue_list, q8_list, columns, r, f)


    ### Step 3: Encode Sequences and Format DataFrames
        #(a) Create codes to encode emission and hidden sequences
        #(b) Apply encodings & specify hidden sequence of interest (q2, q3, q8)

    print("Stage 3: Encoding sequences and formatting dataframes")
    log.write("\nStage 3: Encoding sequences and formatting dataframes \n")
    emission_encode, emission_decode = create_context_code(residue_list[:-1], emission_context)
    state_encode, state_decode = create_context_code(q3_list[:-1], state_context)

    state_symbols, _ = create_context_code(q3_list[:-1], 1)

    log.write("emission_code: \n")
    for k,v in emission_encode.items():
        log.write(f"{k}:{v} ")

    log.write("\n\nstate_code: \n")
    for k,v in state_encode.items():
        log.write(f"{k}:{v} ")
        
    log.write("\n\nstate_symbols: \n")
    for k,v in state_symbols.items():
        log.write(f"{k}:{v} ")

    possible_priors = defaultdict(list)
    if state_context > 1:
        for key1, idx1 in state_encode.items():
            for key2, idx2 in state_encode.items():
                if key1[1:] == key2[0:-1]:
                    possible_priors[idx2].append(idx1)
    else:
        priors = list(state_decode.keys())
        for key, idx in state_encode.items():
            possible_priors[idx] = priors

    log.write("\n Encoding sequences \n")
    train_df_formatted = format_dataset(train_df, emission_encode, state_encode, emission_context, state_context, 'q3_expected')
    eval_df_formatted = format_dataset(eval_df, emission_encode, state_encode, emission_context, state_context, 'q3_expected')


    ### Step 4: Estimate HMM=(T, E, pi) using Our Data

    log.write("\nStage 4: Computing initial estimates for transition and emission matrices using training data \n")
    print("Stage 4: Computing initial estimates for transition and emission matrices using training data")
    start = time.time()
    T = estimate_transition_matrix(train_df_formatted, state_encode, possible_priors)
    E = estimate_emission_matrix(train_df_formatted, state_encode, emission_encode)
    pi = start_distribution(train_df_formatted,state_symbols, state_decode)
    end = time.time()
    log.write(f"Time to estimate T, E, pi is approx: {round((end-start)//60,4)} minutes \n")

    # View parameters
    log.write("Start Distribution (H,C,E): \n")
    log.write(str(pi.round(decimals=4)))
    log.write("\n")

    log.write("Transition Matrix (state x state):")
    log.write(str(T.shape))
    log.write(str(T.round(decimals=4)))
    log.write("\n")

    log.write("Emissions Matrix (state x emission):")
    log.write(str(E.shape))
    log.write(str(E.round(decimals=4)))
    log.write("\n")


    ### Step 5: Compute HMM Performance on Eval Data

    print("\nStage 5: Computing HMM Performance on Eval Data")
    log.write("Stage 5: Computing HMM Performance on Eval Data \n")

    start = time.time()
    eval_predictions = getPredictions(eval_df_formatted, T, E, pi, state_symbols, state_decode, possible_priors)
    eval_acc, eval_prec, eval_rec, eval_cnts = HMMaccuracy(eval_predictions, state_decode, state_symbols)
    end = time.time()
    log.write(f"Time predict on eval data is approx: {round((end-start),2)} seconds \n")
    print(f"Time predict on eval data is approx: {round((end-start),2)} seconds")

    log.write(f"Accuracy: {round(eval_acc,4)} \n")
    print(f"Accuracy: {round(eval_acc,4)} \n")

    log.write(f"Precision (H,C,E):\n\t {eval_prec.round(decimals = 4)} \n")
    print(f"Precision (H,C,E):\n\t {eval_prec.round(decimals = 4)} \n")

    log.write(f"Recall (H,C,E):\n\t {eval_rec.round(decimals = 4)} \n")
    print(f"Recall (H,C,E):\n\t {eval_rec.round(decimals = 4)} \n")

    log.write("Counts (H,C,E) x (H,C,E): \n")
    print("Counts (H,C,E) x (H,C,E):")

    log.write(str(eval_cnts))
    print(eval_cnts)



