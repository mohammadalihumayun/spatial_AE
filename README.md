# Spatial_AE

This repository contains python implementation for an unsupervised, speech representation learning Auto-Encoder AE, model evaluated over automatic KeyWord Spotting KWS.
The relevant paper for the proposed model is currently under review, and we will update its information, if accepted.  

The implementaion has following three code files, for the simulations and evaluation of the AE model.

`dlmodels.py` library that contains the modeules that implement the proposed and benchmark deep learning models in tensorflow.keras   
`simulations.py` file to run the experiments  
`display_results.py` file to plot the results graphically  

For any queries, please contact:  
mohammadalihumayun@gmail.com

# Running the simulations

You need to run `simulations.py` file to run the experiments and to save their results.

## Select input and output data directories
Before running the `simulations.py` file, select the input output data directories and the number of samples to use from dataset.    
Set following variables to select the directory to load input speech dataset and the directory, to save the results:
```
resultspath='x:/results_directory' # folder to save the results as csv and plots
corpuspath='x:/googlecorpus_directory' # folder for speech dataset
```

Set following variables to select the number of samples for unsupervised autoencoder training, and size of training/test sets for supervised KWS evaluation.

```
unlabeled_set_size=12000 # number of samples for autoencoder training
lbl_trai_size=3000 # size of training set for supervised KWS evaluation
lbl_test_size=6000 # size of test set for supervised KWS evaluation
```


**Note:** Our experiments have used Speech_commands, i.e. crowdsourced keywords dataset by google. The dataset can be downloaded from the following URL  
https://www.tensorflow.org/datasets/catalog/speech_commands


## Experiments for evaluation

The code loads audio files with corresponding filenames and sample rates using scipy and extracts their spectral features using librosa.
The dataset is then segregated into unlabaled set for unsupervised AE training and the labeled set for  supervised KWS evaluation.
The mean anchor vector and corresponding postion scalars are computed using cosine distances in features space using scipy.
The deeplearning models are implemented in tensorflow.kers and are imported as modules from `dlmodels.py` file. 
multiple iterations are run for training and evaluation of the proposed as well as benchmark model and the results are saved as csv file named 'results.csv' in the earlier selected results directory. The 'results.csv' stores the results in following format


|wercm|cmpresi|cmrecal|cmavpresi|cmavrecal|wer_v_p|appresi|aprecal|apavpresi|apavrecal|
|---|---|---|---|---|---|---|---|---|---|

## Visualization of results
The spectral feature space and the features extracted by the proposed model are compressed by PCA, scaled and means are plotted as scatter plot for visualization  
Precision and recall values for all classes are loaded from results.csv from the results path and the values are plotted and saved as figures in the results directory
