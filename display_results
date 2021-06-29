## pca plots for visualization


posm=np.load(resultspath+'/ae_representation.npy')
mfcm=np.load(resultspath+'/MFCC_features.npy')
kws_trai_label=np.load(resultspath+'/keyword_labels.npy')

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
