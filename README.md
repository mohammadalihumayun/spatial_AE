# Spatial_AE

This repository contains python implementation for an unsupervised, speech representation learning Auto-Encoder (AE), evaluated over automatic KeyWord Spotting (KWS). The proposed AE is constrained with spatial position of datapoints. Please cite the relevant paper for the proposed model as follows:
```
Humayun MA, Yassin H, Abas PE. 2021. Spatial position constraint for unsupervised learning of speech representations. PeerJ Computer Science 7:e650  
https://doi.org/10.7717/peerj-cs.650
```
The implementaion has following three main code files, for the simulations and evaluation of the AE model.

* `dlmodels.py`: contains modules for implementation of the proposed and benchmark deep learning models   
* `simulations.py`: file to run the experiments and save the results
* `display_results.py`: file to plot the results graphically and save as figures


# Running the simulations

You need to run `simulations.py` file to run the experiments and to save their results.

## Select input and output data directories
Before running the `simulations.py` file, select the input/output data directories, the number of samples to use from dataset, and the number of iterations to test the model.

* Set following variables to select the directory to load input speech dataset and the directory, to save the results:
```
resultspath='x:/results_directory' # folder to save the results as csv and plots
corpuspath='x:/googlecorpus_directory' # folder for speech dataset
```

* Set following variables to select the number of samples for unsupervised autoencoder training, the size of training/test sets for supervised KWS evaluation, and the number of repeatations of train/test cycles for the models:

```
unlabeled_set_size=12000 # number of samples for autoencoder training
lbl_trai_size=3000 # size of training set for supervised KWS evaluation
lbl_test_size=6000 # size of test set for supervised KWS evaluation
num_simulations=10 # iterations for training and testing runs of the models
```


**Note:** Our experiments have used Speech_commands, i.e. crowdsourced keywords dataset by google. The dataset can be downloaded from the following URL  
https://www.tensorflow.org/datasets/catalog/speech_commands


## Experiments for evaluation

The code loads audio files with corresponding filenames and sample rates, and extracts their spectral features.
The features are then segregated into unlabaled set for unsupervised AE training and the labeled set for  supervised KWS evaluation.
The mean anchor vector and corresponding postion scalars are computed using cosine distances in the features space.
The deeplearning models are imported as modules from `dlmodels.py` file. 
Training and evaluation of the proposed as well as benchmark model are repeated for the selected number of iterations over randomly sampled, different training and test sets. 
The simulation results are saved as follows, in the results directory:  
* AE representation for KWS train/test sets as `ae_representation.npy`    
* MFCC features for KWS train/test sets as `MFCC_features.npy`  
* Classification scores as csv file named `results.csv`   

`results.csv` stores the results in following format:  


|MFCC WER|MFCC precision|MFCC recall|MFCC avg precision|MFCC avg recall|AE WER|AE precision|AE recall|AE avg precision|AE avg recall|
|---|---|---|---|---|---|---|---|---|---|


# Visualization of results
You need to run `display_results.py` file to load the results from the results directory, and display them graphically. Running the file plots the following graphs and saves them as figures in the 'results' folder:

* Precision for keywords
* Recall for keywords
* Precision difference between the proposed AE representation and MFCC
* Recall difference between the proposed AE representation and MFCC
* 2D PCA projections for the proposed AE representation
* 2D PCA projections for MFCC
* Computational complexities for complete distance matrix and the proposed position estimate

# Contact information

The model has been developed in Faculty of Integrated Sciences, Universiti Brunei Darussalam, Jalan Tungku Link, BE 1410, Brunei.

For any queries, please contact:

Mohammad Ali Humayun
mohammadalihumayun@gmail.com  
Pg Emeroylariffion Abas
emeroylariffion.abas@ubd.edu.bn  
