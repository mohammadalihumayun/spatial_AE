import scipy.io.wavfile
import os
import numpy as np
import librosa
from scipy.spatial.distance import cdist
import numpy as np
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Dense, Dropout, PReLU, Input, BatchNormalization,Conv1D, Flatten, MaxPooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1,l2,l1_l2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_recall_fscore_support as prf
import csv
from dlmodels import bmcnn, discrim,ae_pos_abx



resultspath='x:/results_directory' # folder to save the results as csv and plots
corpuspath='x:/googlecorpus_directory' # folder for speech dataset
unlabeled_set_size=12000 # number of samples for autoencoder training
lbl_trai_size=3000 # size of training set for supervised KWS evaluation
lbl_test_size=6000 # size of test set for supervised KWS evaluation
num_simulations=10 # iterations for training and testing runs of the models
lbl_trai_size=lbl_trai_size*2 # selecting twice as dataset will be split to half after shuffling for each iteration while supervised KWS training



## load audio files with filenames and sample rates

wavelist=[]
sratlist=[]
filelist=[]
for keywordpath in os.listdir(corpuspath):
    try:
        for audiofile in os.listdir(corpuspath+'/'+keywordpath):
            srate,audio=scipy.io.wavfile.read(corpuspath+'/'+keywordpath+'/'+audiofile)
            wavelist.append(audio)
            sratlist.append(srate)
            filelist.append(keywordpath+'/'+audiofile)
    except:
        pass

# extract spectral features from speech

rawa=np.zeros((len(wavelist),16000))
for i in range(len(wavelist)):
  rawa[i][0:len(wavelist[i])]=wavelist[i]
mfca=[[] for _ in range(len(rawa))]
lpsa=[[] for _ in range(len(rawa))]

for i in range(len(rawa)):
 mfca[i]=librosa.feature.mfcc(y=rawa[i], sr=sratlist[i])
 lpsa[i]=librosa.feature.melspectrogram(y=rawa[i], sr=sratlist[i], n_fft=1025)
mfca=np.array(mfca)
lpsa=np.array(lpsa)

f=open(corpuspath+'/testing_list.txt','r')
testlist=[line.rstrip() for line in f]
f.close()
f=open(corpuspath+'/validation_list.txt','r')
valilist=[line.rstrip() for line in f]
f.close()

# segregate unlabaled set for unsupervised AE training

unlabaledmfc=mfca[np.in1d(filelist,testlist+valilist, invert=True)]
unlabaledlps=lpsa[np.in1d(filelist,testlist+valilist, invert=True)]

# segregate training and test sets for supervised KWS evaluation

labeltrailps=lpsa[np.in1d(filelist,valilist)]
labeltraimfc=mfca[np.in1d(filelist,valilist)]
labeltestlps=lpsa[np.in1d(filelist,testlist)]
labeltestmfc=mfca[np.in1d(filelist,testlist)]
labeltraifle=np.array(filelist)[np.in1d(filelist,valilist)]
labeltestfle=np.array(filelist)[np.in1d(filelist,testlist)]

labeltrai=[[] for _ in range(len(labeltraifle))]
labeltest=[[] for _ in range(len(labeltestfle))]

for i in range(len(labeltraifle)):
 labeltrai[i]=labeltraifle[i].split('/')[0]

for i in range(len(labeltestfle)):
 labeltest[i]=labeltestfle[i].split('/')[0]

labeltest=np.array(labeltest)
labeltrai=np.array(labeltrai)
labelunique=np.unique(labeltrai)

trailabelids=np.zeros(len(labeltrai))
testlabelids=np.zeros(len(labeltest))
for i in range(len(labeltrai)):
  trailabelids[i]=np.where(labeltrai[i]==labelunique)[0][0]

for i in range(len(labeltest)):
  testlabelids[i]=np.where(labeltest[i]==labelunique)[0][0]

trailabelids=trailabelids.astype('int')
testlabelids=testlabelids.astype('int')



# retain number of selected samples

np.random.seed(1)
ridcs=np.arange(len(unlabaledmfc))
np.random.shuffle(ridcs)
ridcs=ridcs[:unlabeled_set_size]
ridcs1=np.arange(len(labeltraimfc))
np.random.shuffle(ridcs1)
ridcs1=ridcs1[0:lbl_trai_size]
ridcs2=np.arange(len(labeltestmfc))
np.random.shuffle(ridcs2)
ridcs2=ridcs2[0:lbl_test_size]


ae_inp_mfc=unlabaledmfc[ridcs]
ae_pos_lps=unlabaledlps[ridcs]

kws_trai_mfc=labeltraimfc[ridcs1]
kws_trai_label=labeltrai[ridcs1]
kws_trai_labelids=trailabelids[ridcs1]

kws_test_mfc=labeltestmfc[ridcs2]
kws_test_label=labeltest[ridcs2]
kws_test_labelids=testlabelids[ridcs2]




# flatten the feature vectors for transforamtion by AE model

kws_test_mfcflat=kws_test_mfc.reshape(len(kws_test_mfc),(len(kws_test_mfc[0])*len(kws_test_mfc[0][0])))
kws_trai_mfcflat=kws_trai_mfc.reshape(len(kws_trai_mfc),(len(kws_trai_mfc[0])*len(kws_trai_mfc[0][0])))
ae_inp_mfcflat=ae_inp_mfc.reshape(len(ae_inp_mfc),(len(ae_inp_mfc[0])*len(ae_inp_mfc[0][0])))
ae_pos_lpsflat=ae_pos_lps.reshape(len(ae_pos_lps),len(ae_pos_lps[0])*len(ae_pos_lps[0][0]))



# computation of mean anchor vector and postion scalars

pfeat=ae_pos_lpsflat # ae_pos_lpsflat, ae_inp_mfcflat
#pfeat=ae_inp_mfcflat[:,5*32:(10)*32] # ae_pos_lpsflat, ae_inp_mfcflat[:,5*32:(10)*32]
acr=np.mean(pfeat,axis=0)
acr=np.reshape(acr,(1,-1))
d=cdist(pfeat,acr, metric='cosine')#correlation, cosine


# training and evaluation for proposed as well as benchmark model

for i in range(num_simulations):
 wercm, cmpresi, cmrecal, cmavpresi, cmavrecal=bmcnn()
 wer_v_p, embeding_v_p, embeding_t_p,hist_p, appresi, aprecal, apavpresi, apavrecal =ae_pos_abx(ae_inp_mfcflat,[ae_inp_mfcflat,d],500,'Adamax',['mse','mse'],0.0,kws_trai_mfcflat,kws_trai_labelids,kws_test_mfcflat,kws_test_labelids,5,0.00,0.0)
 with open(resultspath+'/results.csv', 'a', newline='') as csvfile:
     res = csv.writer(csvfile)
     res.writerow([wercm,cmpresi,cmrecal,cmavpresi,cmavrecal,wer_v_p,appresi,aprecal,apavpresi,apavrecal])



## Print the accuracies

print('Benchmark CNN accuracy using MFCC: ',int(1-wercm)*100,' %')
print('Accuracy using the proposed AE representation: ',int(1-wer_v_p)*100,' %')

## saving the representations for visualization


trainlabeluniqueids=np.unique(kws_trai_labelids)
ltl=len(trainlabeluniqueids)-0
trainlabelidx=[[]]*ltl
for i in range(ltl):
 trainlabelidx[i]=np.where(trainlabeluniqueids[i]==kws_trai_labelids)
 

#mrkr=[".",",","o","v","^","<",">","s","p","P","H","x","X","D","d",".",",","o","v","^","<",">","s","p","P","H","x","X","D","d"]


posm=[[]]*ltl
mfcm=[[]]*ltl
for i in range(ltl):
    posm[i]=np.mean(embeding_t_p[trainlabelidx[i][0][:100]],axis=0)
    mfcm[i]=np.mean(ae_inp_mfcflat[trainlabelidx[i][0][:100]],axis=0)
    
    
posm=np.array(posm)
mfcm=np.array(mfcm)

np.save(resultspath+'/ae_representation.npy',posm)
np.save(resultspath+'/MFCC_features.npy',mfcm)
np.save(resultspath+'/keyword_labels.npy',np.array(kws_trai_label))

