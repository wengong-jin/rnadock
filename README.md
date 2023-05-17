# Identification of Novel RNA-Binding Motifs via Frame Averaging

Understanding the mechanisms of RNA-protein binding is critical for the discovery of new RBPs and development of RBP-targeting therapeutics. Here, we train a model to predict RNA-binding motifs given protein structure and sequence information. We show that an SE(3)-invariant frame averaging architecture robustly learns whether a residue binds to RNA. The majority of common k-mers in predicted binding sites are not seen in the training set, indicating that the model has identified novel RNA-binding motifs. Additionally, we show that the residue classification model can be used to accurately identify RBPs from a set of proteins.

To run code:

1. Create conda environment from rnadock.yml
2. To retrain residue classification model run
```
python models/frame.py
```
3. To run classification model from trained residue classification model checkpoint run
```
python models/rbp_identification.py
```
