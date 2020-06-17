# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:07:51 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:48:45 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:33:41 2020

@author: Admin
"""
from keras import optimizers
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from load_datasets import *
from utils import margin_loss, margin_loss_hard, CustomModelCheckpoint
from deepcaps import DeepCapsNet, DeepCapsNet28, BaseCapsNet
import os
import imp
#################### my test ##################
def test(eval_model, data):

    (x_train, y_train), (x_test, y_test) = data

    # uncommnt and add the corresponding .py and weight to test other models
    # m1 = imp.load_source('module.name', args.save_dir+"/deepcaps.py")
    # _, eval_model = m1.DeepCapsNet28(input_shape=x_test.shape[1:], n_class=10, routings=3)
    eval_model.load_weights(args.save_dir+"/best_weights_2x.h5")
    a1, b1 = eval_model.predict(x_test)
    p1 = np.sum(np.argmax(a1, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    print('Test acc:', p1)
    return a1, b1

class args:
    numGPU = 1
    epochs = 30
    batch_size = 128
    lr = 0.001
    lr_decay = 0.96
    lam_recon = 0.4
    r = 3
    routings = 3
    shift_fraction = 0.1
    debug = False
    digit = 5
    save_dir = 'model/CIFAR10/13'
    t = False
    w = None
    ep_num = 0
    dataset = "my_data"


seeslice=np.zeros((26924,5))
def makemap(img):
    w=127
    h=212 
    s=(w,h)
    whole_t=np.zeros(s)#1,2,3,4
    core_t=np.zeros(s)#1,3,4
    enhanced_t=np.zeros(s)#4
    for i in range(0,w):
        for j in range(0,h):
            if (img[i,j]==1 or img[i,j]==2 or img[i,j]==3 or img[i,j]==4):
                whole_t[i,j]=1
            if(img[i,j]==1 or img[i,j]==4 or img[i,j]==3):
                core_t[i,j]=1
            if(img[i,j]==4):
                enhanced_t[i,j]=1     
    plt.imshow(whole_t) 
    plt.show()         
    return whole_t,core_t,enhanced_t
tpw=0
fnw=0
fpw=0
tpc=0
fnc=0
fpc=0
tpe=0
fne=0
fpe=0
def dice_score_core(pred,gt):
    global tpc
    global fnc
    global fpc
    
    print('****cccccccccc',tpc,fnc,fpc)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j]==gt[i,j] and pred[i,j]==1:
                tpc=tpc+1
            if gt[i,j]==1 and pred[i,j]==0:
                fnc=fnc+1
            if gt[i,j]==0 and pred[i,j]==1:
                fpc=fpc+1
    print('cccccccccc',tpc,fnc,fpc)
def dice_score_whole(pred,gt):
    global tpw
    global fnw
    global fpw
    print('*****WWWWWWWWWWWW',tpw,fnw,fpw)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j]==gt[i,j] and pred[i,j]==1:
                tpw=tpw+1
            if gt[i,j]==1 and pred[i,j]==0:
                fnw=fnw+1
            if gt[i,j]==0 and pred[i,j]==1:
                fpw=fpw+1
    print('WWWWWWWWWWWW',tpw,fnw,fpw)
def dice_score_enhanced(pred,gt):
    global tpe
    global fne
    global fpe
    print('*****eeeeeeeee',tpe,fne,fpe) 
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j]==gt[i,j] and pred[i,j]==1:
                tpe=tpe+1
            if gt[i,j]==1 and pred[i,j]==0:
                fne=fne+1
            if gt[i,j]==0 and pred[i,j]==1:
                fpe=fpe+1
    # if (2*tp+fp+fn==0):
    #     d=1
    # else:
    #     d=(2*tp)/(2*tp+fp+fn) 
    print('eeeeeeeee',tpe,fne,fpe)    
    

def mytest(path):
    global tpc
    global fnc
    global fpc
    global tpw
    global fnw
    global fpw
    global tpe
    global fne
    global fpe
    
    tpw=0
    fnw=0
    fpw=0
    tpc=0
    fnc=0
    fpc=0
    tpe=0
    fne=0
    fpe=0

    p = os.listdir(path)
    p.sort(key=str.lower)
    
    arr = []
    for i in range(len(p)):
      if(i != 4):
        img = sitk.ReadImage(path+'/'+p[i])
        arr.append(sitk.GetArrayFromImage(img))
      else:
        img = sitk.ReadImage(path+'/'+p[i])
        Y_labels = sitk.GetArrayFromImage(img) 
    
    data = np.zeros((Y_labels.shape[1],Y_labels.shape[0],Y_labels.shape[2],4))
    for i in range(Y_labels.shape[1]):
        data[i,:,:,0] = arr[0][:,i,:]
        data[i,:,:,1] = arr[1][:,i,:]
        data[i,:,:,2] = arr[2][:,i,:]
        data[i,:,:,3] = arr[3][:,i,:]
        
    s=np.random.randint(60,180,30)
    # s=[80,70,65,120,140]
    
    for i in range(0,30):
        
        d=test_model_gen(28, data, Y_labels,s[i])
        x_test=d[0]
        y_test=d[2]
        xTest=x_test.astype('float32')
        yTest = np.zeros((d[2].shape[0],5))
        # a1,b1=test(eval_model, ((xTest,yTest), (xTest,yTest)))
        yTest = np.zeros((d[2].shape[0],5))
        for j in range(yTest.shape[0]):
          yTest[j,d[2][j]] = 1
        a1,b1=test(eval_model, ((xTest,yTest.astype('float32')), (xTest,yTest.astype('float32'))))
        
        x=data[s[i]]
        gtt=x[:,:,0]
        plt.imshow(gtt)
        plt.show()
        #
        new_y=np.zeros((26924,1))
        m=0
        for j in range(0,26924):
            if a1[j,0]>=a1[j,1] and a1[j,0]>=a1[j,2]  and a1[j,0]>=a1[j,3] and a1[j,0]>=a1[j,4]:
                # print(a1[j,0],a1[j,1],a1[j,2],a1[j,3],a1[j,4])
                # print('0000000000000')
                m=0
            elif a1[j,1]>=a1[j,0]  and a1[j,1]>=a1[j,2]  and a1[j,1]>=a1[j,3] and a1[j,1]>=a1[j,4] :
                m=1
                # print(a1[j,0],a1[j,1],a1[j,2],a1[j,3],a1[j,4])
                # print('1111111111')
            elif a1[j,2]>=a1[j,0]  and a1[j,2]>=a1[j,1]  and a1[j,2]>=a1[j,3] and a1[j,2]>=a1[j,4] :
                m=2
                # print(a1[j,0],a1[j,1],a1[j,2],a1[j,3],a1[j,4])
                # print('22222222222')
            elif a1[j,3]>=a1[j,0]  and a1[j,3]>=a1[j,1]  and a1[j,3]>=a1[j,2] and a1[j,3]>=a1[j,4] :
                m=3
                # print(a1[j,0],a1[j,1],a1[j,2],a1[j,3],a1[j,4])
                # print('3333333333')
            elif a1[j,4]>=a1[j,0]  and a1[j,4]>=a1[j,1]  and a1[j,4]>=a1[j,2] and a1[j,4]>=a1[j,3] :
                m=4
                # print(a1[j,0],a1[j,1],a1[j,2],a1[j,3],a1[j,4])
                # print('4444444444444')
            # else: 
            #     m=0
            new_y[j,0]=m
        #
        wid=127
        h=212    
        y_test=y_test.reshape(wid,h)
        w,c,e=makemap(y_test)
        new_y= new_y.reshape(wid,h)
        w1,c1,e1=makemap(new_y)
        dice_score_whole(w1,w)
        dice_score_core(c1,c)
        dice_score_enhanced(e1,e)
        
        
    d_w=0
    d_c=0
    d_e=0
    
    print("*****slice*******"+str(i))
    print('eeeeeeeee',tpe,fne,fpe)   
    if (2*tpc+fpc+fnc==0):
        d_c=1
    else:
        d_c=(2*tpc)/(2*tpc+fpc+fnc) 
    if (2*tpw+fpw+fnw==0):
        d_w=1
    else:
        d_w=(2*tpw)/(2*tpw+fpw+fnw) 
    if (2*tpe+fpe+fne==0):
        d_e=1
    else:
        d_e=(2*tpe)/(2*tpe+fpe+fne) 
    print(d_w,d_c,d_e)
    
    return d_w,d_c,d_e



def test_model_gen(input_dim,Data,y,slice_no):
  x=Data[slice_no]
  max0=np.max(x[:,:,0])
  max1=np.max(x[:,:,1])
  max2=np.max(x[:,:,2])
  max3=np.max(x[:,:,3])
 
  min0=np.min(x[:,:,0])
  min1=np.min(x[:,:,1])
  min2=np.min(x[:,:,2])
  min3=np.min(x[:,:,3])

  if (max0-min0 != 0):
      x[:,:,0]=(x[:,:,0]-min0)/(max0-min0)
  if (max1-min1 != 0):
      x[:,:,1]=(x[:,:,1]-min1)/(max1-min1)
  if (max2-min2 != 0):
    x[:,:,2]=(x[:,:,2]-min2)/(max2-min2)
  if (max3-min3 != 0):
    x[:,:,3]=(x[:,:,3]-min3)/(max3-min3)

    
    
    
  X1 = []
  X2 = []
  Y = []
  num=0
  for i in range(int((input_dim)/2),y.shape[0]-int((input_dim)/2)):#32-123
    for j in range(int((input_dim)/2),y.shape[2]-int((input_dim)/2)):#32-208
      #Filtering all 0 patches
        X2.append(x[i-16:i+17,j-16:j+17,:])
        X1.append(x[i-int((input_dim)/2):i+int((input_dim)/2),j-int((input_dim)/2):j+int((input_dim)/2),:])
        Y.append(y[i,slice_no,j])
        num=num+1
        
  X1 = np.asarray(X1)
  X2 = np.asarray(X2)
  Y = np.asarray(Y)
  d = [X1,X2,Y]
  return d


model, eval_model = DeepCapsNet28((28,28,4), n_class=5, routings=args.routings)
test_path='D:/haj/dataset/test/'
fold = os.listdir(test_path)
fold.sort(key=str.lower) 
num=0
for path in fold:
    path =test_path+path
    d_w,d_c,d_e=mytest(path)
    num=num+1
    
    print('patient'+str(num)+'whole tumor',np.average(d_w))
    print('patient'+str(num)+'core tumor',np.average(d_c))
    print('patient'+str(num)+'enhanced core',np.average(d_e))
    

##############################################