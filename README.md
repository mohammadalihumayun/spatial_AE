# Spatial_AE
Following steps represent the flow of experiments in terms of the code in main.py file

## Input and output data directories
First of all the directory to save the results and the directory to load the input speech dataset are selected  
Speech_commands, i.e. Google Dataset. has been used for out experiments which can be downloaded from the following URL  
https://www.tensorflow.org/datasets/catalog/speech_commands

## Data preperation
Select the number of samples for unsupervised autoencoder training, and size of training and test sets for supervised KWS evaluation .

```
resultspath='x:/results_directory' # folder to save the results as csv and plots
corpuspath='x:/googlecorpus_directory' # folder for speech dataset
unlabeled_set_size=12000 # number of samples for autoencoder training
lbl_trai_size=3000 # size of training set for supervised KWS evaluation
lbl_test_size=6000 # size of test set for supervised KWS evaluation
```

Audio files are loaded with corresponding filenames and sample rates using scipy  
Spectral features are extracted from loaded audio files using librosa  
Unlabaled set for unsupervised AE training and labeled set for   supervised KWS evaluation are segregated  

## Experiments for evaluation
Mean anchor vector and postion scalars are computed using cosine distances in features space using scipy  
The proposed and the benchmark models based on tf.keras are imported from the module dlmodels  
Training and evaluation for proposed as well as benchmark model are performed for multiple iterations using the loaded models  
The results are saved as csv file named results.csv in the earlier selected results directory  

## Visualization of results
The spectral feature space and the features extracted by the proposed model are compressed by PCA, scaled and means are plotted as scatter plot for visualization  
Precision and recall values for all classes are loaded from results.csv from the results path and the values are plotted and saved as figures in the results directory
