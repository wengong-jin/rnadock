import ast
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

def visualize_match(sequence, pred_lst, true_lst, idx):
    assert len(sequence) == len(pred_lst) and len(sequence) == len(true_lst)
    arr = np.concatenate((np.expand_dims(np.array(pred_lst), axis=1), np.expand_dims(np.array(true_lst), axis=1)), axis=1).T
    fig, ax = plt.subplots(figsize=(20,5))
    sns.heatmap(arr[:,100:200], annot=False, linewidth = .5, yticklabels=["pred","true"], xticklabels=sequence[100:200]).figure.savefig(f"visualizations/heatmap_example{idx}.png", dpi=1200)

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

def generate_substrings(strings, prot_labels):
    substrings = []
    labels = []
    for k in range(len(strings)):
        s = strings[k]
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                if len(s[i:j]) >= 5:
                    substrings.append(s[i:j])
                    labels.append(prot_labels[k])
    return substrings, labels

def most_frequent_words(lst, n, prot_labels):
    dct = {}
    represented_prots = []
    words = []
    word_counts = Counter(lst)
    top_n = word_counts.most_common(n)
    for word, count in top_n:
        dct[word] = count
        indexes = [i for i in range(len(lst)) if lst[i] == word]
        for idx in indexes:
            represented_prots.append(prot_labels[idx])
            words.append(word)
    return dct, represented_prots, words

def plot_histogram(data):
    labels = list(data.keys())
    frequencies = list(data.values())
    
    filtered_labels = []
    filtered_frequencies = []
    for i in range(len(labels)):
        dont_add = False
        for j in range(len(labels)):
            if i == j:
                pass
            else:
                if labels[i] in labels[j] and frequencies[i] <= frequencies[j]:
                    dont_add = True
        if not dont_add:
            filtered_labels.append(labels[i])
            filtered_frequencies.append(frequencies[i])

    fig, ax = plt.subplots(figsize=(8,5))
    d = {"Sequences": filtered_labels, "Frequencies": filtered_frequencies}
    df = pd.DataFrame(d)

    sns.set_palette("magma")
    fig = sns.barplot(data=df, x="Sequences", y="Frequencies", palette="rocket",ax=ax)
    ax.set_xlabel('Sequences')
    ax.set_ylabel('Frequency')
    ax.set_title('Most Common K-mers in Predicted Binding Sites [Validation]')
    ax.set_xticklabels(filtered_labels, rotation=90)
    plt.tight_layout()
    plt.savefig("visualizations/kmer_hist_val_pred_3.png", dpi=1200)

if __name__ == "__main__":
    df = pd.read_csv("charts_rna_binary_noligand_fullrun/visualization_pred_info.csv")

    contiguous_predicted_seqs = []
    protein_identity = []

    for i in range(len(df)):
        seq, pred, true = (df.iloc[i][0], df.iloc[i][1], df.iloc[i][2])
        pred = ast.literal_eval(pred)
        true = ast.literal_eval(true)
        
        # visualize_match(seq, pred, true, i)
        # if i == 17:
        #     break

        contiguous_predicted_seqs.extend(get_contiguous_predicted_sites(seq, pred))
        protein_identity.extend(len(contiguous_predicted_seqs) * [i])
    
    kmers, prot_labels = generate_substrings(contiguous_predicted_seqs, protein_identity)
    kmer_dict, represented_prots, words = most_frequent_words(kmers, 200, prot_labels)
    plot_histogram(kmer_dict)