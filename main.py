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



resultspath='f:/testcorpus' # folder to save the results as csv and plots
corpuspath='x:/googlecorpus' # folder for speech dataset
unlabeled_set_size=12000 # number of samples for autoencoder training
lbl_trai_size=3000 # size of training set for supervised KWS evaluation
lbl_test_size=6000 # size of test set for supervised KWS evaluation
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

for i in range(10):
 wercm, cmpresi, cmrecal, cmavpresi, cmavrecal=bmcnn()
 wer_v_p, embeding_v_p, embeding_t_p,hist_p, appresi, aprecal, apavpresi, apavrecal =ae_pos_abx(ae_inp_mfcflat,[ae_inp_mfcflat,d],500,'Adamax',['mse','mse'],0.0,kws_trai_mfcflat,kws_trai_labelids,kws_test_mfcflat,kws_test_labelids,5,0.00,0.0)
 with open(resultspath+'/results.csv', 'a', newline='') as csvfile:
     res = csv.writer(csvfile)
     res.writerow([wercm,cmpresi,cmrecal,cmavpresi,cmavrecal,wer_v_p,appresi,aprecal,apavpresi,apavrecal])

## pca plots for visualization


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


X=mfcm
X = (X - X.min())/(X.max() - X.min())
pca = skPCA(n_components=2)
transformed = pca.fit_transform(X)
plt.figure()
for i in range(0,len(X)):
 plt.scatter(transformed[i][0],transformed[i][1])#,label=np.unique(trainlabel)[i], marker=mrkr[i],s=50)#, color=colors[i],marker=mrkr[i],s=50)#mrkr[i]
 plt.annotate(np.unique(kws_trai_label)[i], (transformed[i][0],transformed[i][1]))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title('PCA-projection_MFCC_mean-of-points')
#plt.legend()
plt.savefig('MFCC_30',dpi=300)
plt.show()
plt.close()


X= posm
X = (X - X.min())/(X.max() - X.min())
pca = skPCA(n_components=2)
transformed = pca.fit_transform(X)
plt.figure()
for i in range(0,len(X)):
 plt.scatter(transformed[i][0],transformed[i][1],s=50)#,label=np.unique(trainlabel)[i], marker=mrkr[i],s=50)#, color=colors[i], marker=mrkr[i],s=50)#
 plt.annotate(np.unique(kws_trai_label)[i], (transformed[i][0],transformed[i][1]))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title('PCA-projection_Transform_with-Position-constraint_mean-of-points')
plt.savefig('AE_PROJECTION_30',dpi=300)
plt.show()
plt.close()



# precision and recall value display based on, vectors for all iterations, extracted from results.csv


datasetfile=resultspath+'/results.csv'
with open(datasetfile, newline='', encoding='ANSI') as f:
     reader = csv.reader(f)
     data = list(reader)

# 1,2 6,7
cmpresi=np.sum(np.array([x.replace('[','').replace(']','').split() for x in np.array(data)[:,1]]).astype('float'),axis=0)
cmrecal=np.sum(np.array([x.replace('[','').replace(']','').split() for x in np.array(data)[:,2]]).astype('float'),axis=0)
appresi=np.sum(np.array([x.replace('[','').replace(']','').split() for x in np.array(data)[:,6]]).astype('float'),axis=0)
aprecal=np.sum(np.array([x.replace('[','').replace(']','').split() for x in np.array(data)[:,7]]).astype('float'),axis=0)

# sample numbers in the dataset from the corpus paper

key=['bed','bird','cat','dog','down','eight','five','four','go','happy','house','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','wow','yes','zero']
samp=np.array([2014,2064,2031,2128,3917,3787,4052,3728,3880,2054,2113,3801,2100,3934,3941,3745,3845,3890,3778,3998,2022,3860,3872,3727,1759,3880,3723,2123,4044,4052])

# precision recall plots

X=samp
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (0.2 +0.2) -0.2

plt.figure()
plt.plot(cmpresi/10,label='MFCC_CNN',marker='.')
plt.plot(appresi/10,label='AE-REP_DNN',marker='.')
plt.title('Precision')
plt.xlabel('Keyword ID')
plt.legend()
plt.savefig(resultspath+'/precision',dpi=300)
plt.show()
plt.figure()
plt.plot(cmrecal/10,label='MFCC_CNN',marker='.')
plt.plot(aprecal/10,label='AE-REP_DNN',marker='.')
plt.title('Recall')
plt.xlabel('Keyword ID')
plt.legend()
plt.savefig(resultspath+'/recall',dpi=300)
plt.show()
plt.figure()
plt.plot(X_scaled,label='Samples_scaled',marker='.',linewidth=0.5)
plt.plot((appresi/10)-(cmpresi/10),label='Score_difference',marker='.')
plt.title('Precision_Difference')
plt.xlabel('Keyword ID')
plt.legend()
plt.savefig(resultspath+'/precision_difference',dpi=300)
plt.show()
plt.figure()
plt.plot(X_scaled,label='Samples_scaled',marker='.',linewidth=0.5)
plt.plot((aprecal/10)-(cmrecal/10),label='Score_difference',marker='.')
plt.title('Recall_difference')
plt.xlabel('Keyword ID')
plt.legend()
plt.savefig(resultspath+'/recall_difference',dpi=300)
plt.show()