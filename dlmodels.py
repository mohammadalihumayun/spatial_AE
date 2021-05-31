
# benchmark CNN for supervised KWS

def bmcnn():
 pt=5
 inputdim1=kws_trai_mfc.shape
 outputdim1=to_categorical(kws_trai_labelids).shape[1]
 input_flat1 = Input(shape=(inputdim1[1:]))
 h=Conv1D(64, kernel_size=32,strides=2, padding='same')(input_flat1)#
 h=BatchNormalization()(h)
 h=MaxPooling1D()(h)# 
 h=Flatten()(h) 
 output_layer1 = Dense(outputdim1, activation='softmax')(h)
 emodel1 = Model(input_flat1,output_layer1)
 emodel1.compile(optimizer='Adamax', loss='categorical_crossentropy')
 emodel1.summary()
 es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,min_delta=0, patience=pt)
 hist= emodel1.fit(kws_trai_mfc,to_categorical(kws_trai_labelids),epochs=100,batch_size=32,shuffle=True,validation_split=0.5,callbacks=[es],verbose=1) 
 prediction=emodel1.predict(kws_test_mfc)
 print('acc',sum(np.argmax(prediction,axis=1)==kws_test_labelids)/len(prediction))
 print('wer',1-(sum(np.argmax(prediction,axis=1)==kws_test_labelids)/len(prediction)))
 wer=1-(sum(np.argmax(prediction,axis=1)==kws_test_labelids)/len(prediction))
 presi=prf(np.argmax(prediction,axis=1),kws_test_labelids)[0]
 recal=prf(np.argmax(prediction,axis=1),kws_test_labelids)[1]
 avpresi=prf(np.argmax(prediction,axis=1),kws_test_labelids,average='weighted')[0]
 avrecal=prf(np.argmax(prediction,axis=1),kws_test_labelids,average='weighted')[1]
 return wer, presi, recal, avpresi, avrecal


# feedforward DNN for supervised KWS

def discrim(indata,outdata1,werval,werlabelval1,epc,dvspt,dpt):
 outdata=to_categorical(outdata1)
 werlabelval=to_categorical(werlabelval1)
 oact='softmax' 
 hact='relu'
 spl=len(indata)
 inputdim=len(indata[0])
 outputdim=len(outdata[0])
 input_flat1 = Input(shape=(inputdim,))
 hidden_layer1 = Dense(int((outputdim+inputdim)/2))(input_flat1)
 h2=BatchNormalization()(hidden_layer1)
 h3=PReLU(alpha_initializer='zeros')(h2)
 h4=Dropout(0.1)(h3)
 output_layer1 = Dense(outputdim, activation=oact)(h4)
 emodel1 = Model(input_flat1,output_layer1)
 emodel1.compile(optimizer='Nadam', loss='categorical_crossentropy')
 emodel1.summary()
 #plot_model(emodel1, to_file='dmodel.png')
 es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=dpt)
 hist= emodel1.fit(indata,outdata,epochs=epc,batch_size=32,shuffle=True,validation_split=dvspt,callbacks=[es])
 flat_outa = emodel1.predict(werval)
 flatword = np.argmax((flat_outa),axis=1)
 outword = np.argmax((werlabelval),axis=1)
 flaterror=0
 for j in range(0, len(werval)):
  if flatword[j] == outword[j]:
   flaterror = flaterror
  else:
   flaterror=flaterror+1
 
 percenterror=(flaterror*100)/len(werval)
 presi=prf(flatword,werlabelval1)[0]
 recal=prf(flatword,werlabelval1)[1]
 avpresi=prf(flatword,werlabelval1,average='weighted')[0]
 avrecal=prf(flatword,werlabelval1,average='weighted')[1]
 return percenterror, hist, presi, recal, avpresi, avrecal

# proposed AE with position constraint

def ae_pos_abx(indata,outdata,epc,opt,lss,avspt,discval,disclabelval,werval,werlabelval,apt,l1r,l2r):
 #cp = ModelCheckpoint(fp)
 #cbl = [cp]
 inputdim=len(indata[0])
 inputlayer = Input(shape=(inputdim,))
 outputdim1=len(outdata[0][0])
 outputdim2=len(outdata[1][0])
 L=Dense(300,kernel_regularizer=l2(0.0))(inputlayer)
 #L=Reshape((300,))(L)
 L= BatchNormalization()(L)
 L=PReLU(activity_regularizer=l1(0.0))(L)
 L=Dropout(0.1)(L)
 L=Dense(150,kernel_regularizer=l2(0.0))(L)
 L= BatchNormalization()(L)
 #L=PReLU(activity_regularizer=l1(0.0))(L)
 #L=Dropout(0.1)(L)
 emb_layer=Dense(75)(L)
 #L=PReLU(activity_regularizer=l1(0.0))(L)
 #emb_layer=BatchNormalization(name='embedding')(L)
 houtlayer1 = Dense(300)(emb_layer)
 houtlayer1a=BatchNormalization()(houtlayer1)
 #houtlayer1b=PReLU()(houtlayer1a)
 houtlayer1bd=Dropout(0.1)(houtlayer1a)
 houtlayer2 = Dense(50,kernel_regularizer=l2(0.000))(emb_layer)
 houtlayer2a= BatchNormalization()(houtlayer2)
 houtlayer2b=PReLU(activity_regularizer=l1(0.0))(houtlayer2a)
 #houtlayer2c=Dropout(0.1)(houtlayer2b)
 #houtlayer2d = Dense(25,kernel_regularizer=l2(0.000))(houtlayer2b)
 #output
 outputlayer1 = Dense(outputdim1, name='primary-output__MFCC-reconstruction')(houtlayer1bd)
 outputlayer2 = Dense(outputdim2, name='secondary-task__geometric-position',kernel_regularizer=l1_l2(l1=l1r, l2=l2r))(houtlayer2b)#,kernel_constraint=MinMaxNorm(min_value=0.5,max_value=1.5, axis=0)
 embedmodel = Model(inputlayer,emb_layer)
 regenmodel = Model(inputlayer,[outputlayer1,outputlayer2])#,outputlayer3,outputlayer4,outputlayer5
 regenmodel.compile(optimizer=opt, loss=lss)
 regenmodel.summary()
 #plot_model(regenmodel, to_file='model.png')
 es = EarlyStopping(monitor='loss', mode='min',verbose=1, patience=apt)
 #es = EarlyStopping(monitor='loss', mode='min',verbose=1,min_delta=0, patience=1)
 aehist=regenmodel.fit(indata,outdata,epochs=epc,batch_size=32,shuffle=True,validation_split=avspt,callbacks=[es])
 #aehist=regenmodel.fit(indata,outdata,epochs=epc,batch_size=32,shuffle=True,callbacks=[es])
 embeddingtrain=embedmodel.predict(indata)
 plt.plot(embeddingtrain[0])
 embeddingval=embedmodel.predict(discval)
 werembval=embedmodel.predict(werval)
 wer,hist, presi, recal, avpresi, avrecal=discrim(embeddingval,disclabelval,werembval,werlabelval,500,0.5,5)
 print('wer ',wer)
 return wer, embeddingval, embeddingtrain,aehist, presi, recal, avpresi, avrecal
