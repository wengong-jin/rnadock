import ast
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

def visualize_match(sequence, pred_lst, true_lst):
    assert len(sequence) == len(pred_lst) and len(sequence) == len(true_lst)
    print(np.sum(np.expand_dims(np.array(pred_lst), axis=1)))
    print(np.sum(np.expand_dims(np.array(true_lst), axis=1)))
    print(len(sequence))
    arr = np.concatenate((np.expand_dims(np.array(pred_lst), axis=1), np.expand_dims(np.array(true_lst), axis=1)), axis=1).T
    fig, ax = plt.subplots(figsize=(90,5))
    sns.heatmap(arr, annot=False, linewidth = .5, yticklabels=["pred","true"], xticklabels=sequence).figure.savefig("visualizations/chart.png")

def get_contiguous_predicted_sites(sequence, pred_lst):
    pred_arr = np.where(np.array(pred)>0.19, 1, 0)
    change_indices = np.where(np.diff(pred_arr) != 0)[0] + 1 # all indices where prev elem is diff than this elem
    if pred_arr[0] == 1:
        change_indices = np.insert(change_indices, 0, 0) # if first elem is 1, add 0 to beginning
    if pred_arr[-1] == 1:
        change_indices = np.append(change_indices, len(pred_arr)) # same for last elem

    start_end_indices = change_indices.reshape(-1, 2)
    contiguous_chunks = [np.array(list(range(start, end))) for start, end in start_end_indices if pred_arr[start] == 1 and end-start >= 5] # chunk must be >= size 5
    
    sites = []
    seq_arr = np.array(list(sequence))
    for chunk in contiguous_chunks:
        sites.append("".join(list(seq_arr[chunk])))
    
    return sites

def generate_substrings(strings):
    substrings = []
    for s in strings:
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                if len(s[i:j]) >= 5:
                    substrings.append(s[i:j])
    return substrings

def most_frequent_words(lst, n):
    dct = {}
    word_counts = Counter(lst)
    top_n = word_counts.most_common(n)
    for word, count in top_n:
        dct[word] = count
    return dct

def plot_histogram(data):
    labels = list(data.keys())
    frequencies = list(data.values())

    fig, ax = plt.subplots(figsize=(30,15))

    ax.bar(labels, frequencies)
    ax.set_xlabel('Sequences')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of K-mers')
    plt.xticks(rotation=90)

    plt.savefig("visualizations/kmer_hist.png")

if __name__ == "__main__":
    df = pd.read_csv("charts_rna_binary_noligand_fullrun/visualization_info.csv")

    contiguous_predicted_seqs = []

    for i in range(len(df)):
        seq, pred, true = (df.iloc[i][0], df.iloc[i][1], df.iloc[i][2])
        pred = ast.literal_eval(pred)
        true = ast.literal_eval(true)

        contiguous_predicted_seqs.extend(get_contiguous_predicted_sites(seq, pred))
    
    kmers = generate_substrings(contiguous_predicted_seqs)
    kmer_dict = most_frequent_words(kmers, 50)
    plot_histogram(kmer_dict)

    
        
