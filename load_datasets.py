from keras.utils import to_categorical
import numpy as np
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt

numall=80000
X_train=np.zeros((numall,33,33, 4),dtype=np.float32)
X_train2=np.zeros((numall,15,15, 4),dtype=np.float32)

y_train=np.zeros((numall,1),dtype=np.float32)
mm=0

num1=0
num2=0
num3=0
num4=0
num0=0

   
def load_tiny_imagenet(path1,num_classes=5):
    global y_train
    global X_train
    global X_train2
    global num0
    global num1
    global num2
    global num3
    global num4
    fold = os.listdir(path1)
    fold.sort(key=str.lower) 
    firtst=fold[0]
    mylen=len(fold)
    andis=0
    while (andis<mylen):
        
        pathmy=fold[andis]
        p_rand=np.random.rand(1)
        if (p_rand>0.50):
            continue
        #print(path)
        path =path1+pathmy
        p = os.listdir(path)
        p.sort(key=str.lower)
        arr = []
        # print(path+'/'+p[1])
        #print(len(p))
        # Reading from 4 images and creating 4 channel slice-wise 
        for i in range(len(p)):
          if(i != 4):
            # print(path+'/'+p[i])
            img = sitk.ReadImage(path+'/'+p[i])#+'/'+p1[-1]
            
            arr.append(sitk.GetArrayFromImage(img))
          else:
            # print(path+'/'+p[i])
            img = sitk.ReadImage(path+'/'+p[i])#+'/'+p1[0]
            Y_labels = sitk.GetArrayFromImage(img)
        data = np.zeros((Y_labels.shape[1],Y_labels.shape[0],Y_labels.shape[2],4))
        for i in range(Y_labels.shape[1]):
          data[i,:,:,0] = arr[0][:,i,:]
          data[i,:,:,1] = arr[1][:,i,:]
          data[i,:,:,2] = arr[2][:,i,:]
          data[i,:,:,3] = arr[3][:,i,:]
        
        # Creating patches for each slice and training(slice-wise)
        s1=np.random.randint(30,200,3)
        for i in range(3):
          flag=model_gen(33,data,Y_labels,s1[i])
          
          if(flag != 0):
               cat_y= np.zeros((y_train.shape[0],5))
               for j in range(y_train.shape[0]):
                     cat_y[j,int(y_train[j])] = 1 
               cat_y=cat_y.astype(np.float32) 
               X_train=X_train.astype(np.float32) 
               X_train2=X_train2.astype(np.float32) 
               # print(np.shape(X_train),np.shape(cat_y))
               return X_train,X_train2,cat_y
               
        
        if (pathmy ==fold[-1]):
                 andis=-1
        andis=andis+1
   
def model_gen(input_dim,data,y,slice_no):
  global mm
  f=0
  x = data[slice_no]
  # print(x.any() ,'&&&&&&&&&&',np.sum(y[:,slice_no,:]) )
  
  if(x.any() != 0 ):
      # plt.imshow(x[:,:,1])
      # plt.show()
      # plt.imshow(y[:,slice_no,:])
      # plt.show()
      # print( '*****')
      max0=np.max(x[:,:,0])
      max1=np.max(x[:,:,1])
      max2=np.max(x[:,:,2])
      max3=np.max(x[:,:,3])
     
      min0=np.min(x[:,:,0])
      min1=np.min(x[:,:,1])
      min2=np.min(x[:,:,2])
      min3=np.min(x[:,:,3])
      
      if (max0-min0):
          x[:,:,0]=(x[:,:,0]-min0)/(max0-min0)
      if (max1-min1):
        x[:,:,1]=(x[:,:,1]-min1)/(max1-min1)
      if (max2-min2):
        x[:,:,2]=(x[:,:,2]-min2)/(max2-min2)
      if (max3-min3):
        x[:,:,3]=(x[:,:,3]-min3)/(max3-min3)
      
      # plt.imshow(x[:,:,0])
      # plt.show()
      for i in range(int((input_dim)/2),y.shape[0]-int((input_dim)/2)):
        for j in range(int((input_dim)/2),y.shape[2]-int((input_dim)/2)):
          #Filtering all 0 patches
            if(f==0 and loaddata( y[i,slice_no,j])) :
          
                  if(f==0 and x[i-7:i+8,j-7:j+8,:].any != 0): 
                     X_train[mm,:,:,0]=x[i-int((input_dim)/2):i+int((input_dim)/2)+1,j-int((input_dim)/2):j+int((input_dim)/2)+1,0]
                     X_train[mm,:,:,1]=x[i-int((input_dim)/2):i+int((input_dim)/2)+1,j-int((input_dim)/2):j+int((input_dim)/2)+1,1]
                     X_train[mm,:,:,2]=x[i-int((input_dim)/2):i+int((input_dim)/2)+1,j-int((input_dim)/2):j+int((input_dim)/2)+1,2]
                     X_train[mm,:,:,3]=x[i-int((input_dim)/2):i+int((input_dim)/2)+1,j-int((input_dim)/2):j+int((input_dim)/2)+1,3]
                     X_train2[mm,:,:,0]=x[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1,0]
                     X_train2[mm,:,:,1]=x[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1,1]
                     X_train2[mm,:,:,2]=x[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1,2]
                     X_train2[mm,:,:,3]=x[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1,3]
                     
                     y_train[mm]=y[i,slice_no,j]
                     # print(X_train[mm,16,16,0],x[i,j,0],X_train2[mm,7,7,0])
                      #print(mm)
                     mm=mm+1
                     # print('&&&^^^^',num0,num1,num2,num3,num4)
                     if (num0==16*numall/20 and num1==numall/20 and num2==numall/20 and num3==numall/20 and num4==numall/20):
                         mm=0
                         f=1
                         break
  # print('fff',f)                 
  return f     

def loaddata(y):
    global num0
    global num1
    global num2
    global num3
    global num4
    global numall
    n=np.random.rand(1)
    qq=np.random.rand(1)
    flag=0
    if (y==0):
       if (num0<(16*numall/20)and n>0.997 and qq>0.6):
           num0=num0+1
           flag=1
    elif (y==1):
        if (num1<(numall/20)):
            num1=num1+1
            flag=1
    elif (y==2):
        if(num2<(numall/20)and n>0.8 and qq>0.7):
            num2=num2+1
            flag=1
    elif (y==3):
       if (num3<(numall/20)and n>0.2 and qq>0.35):
            num3=num3+1
            #num3=num3+#
            flag=1
    elif (y==4):
        if (num4<(numall/20)and n>0.6 and qq>0.39):
            num4=num4+1
            flag=1
    return flag

def resize(data_set, size):
    X_temp = []
    import scipy
    for i in range(data_set.shape[0]):
        resized = np.resize(data_set[i], (size, size))
        X_temp.append(resized)
    X_temp = np.array(X_temp, dtype=np.float32) / 255.
    return X_temp