# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:17:03 2020

@author: Admin
"""
from keras import optimizers
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from load_datasets import *
from utils import margin_loss, margin_loss_hard, CustomModelCheckpoint
import os
import imp
from twowaydeep import DeepCapsNetTwoPath
import matplotlib.pyplot as plt

def train(model, data, hard_training, args):
    # unpacking the data
    (x_train, y_train), (x_train2),(x_test, y_test), (x_test2) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log' + appendix + '.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', batch_size=args.batch_size, histogram_freq=int(args.debug), write_grads=False)
    checkpoint1 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_1' + appendix + '.h5', monitor='val_capsnet_acc', 
                                        save_best_only=False, save_weights_only=True, verbose=1)

    checkpoint2 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_2' + appendix + '.h5', monitor='val_capsnet_acc',
                                        save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * 0.5**(epoch // 10))

    if(args.numGPU > 1):
        parallel_model = multi_gpu_model(model, gpus=args.numGPU)
    else:
        parallel_model = model

    if(not hard_training):
        parallel_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[margin_loss, 'mse'], loss_weights=[1, 0.4], metrics={'capsnet': "accuracy"})
    else:
        parallel_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[margin_loss_hard, 'mse'], loss_weights=[1, 0.4], metrics={'capsnet': "accuracy"})

    # Begin: Training with data augmentation
    def train_generator(x1,x2, y, batch_size, shift_fraction=args.shift_fraction):
       
        train_datagen = ImageDataGenerator()  # shift up to 2 pixel for MNIST
        genX1 = train_datagen.flow(x1, y,  batch_size=batch_size, seed=1)
        genX2 = train_datagen.flow(x2, y, batch_size=batch_size, seed=1)
        while True:
            x1_batch, y_batch = genX1.next()
            x2_batch, y_batch = genX2.next()
            yield ([x1_batch, x2_batch,y_batch], [y_batch, x1_batch])

    
   
    parallel_model.fit_generator(generator=train_generator(x_train,x_train2, y_train, args.batch_size, args.shift_fraction),
                                  steps_per_epoch=int(y_train.shape[0] / args.batch_size), epochs=args.epochs,
                                  validation_data=[[x_test,x_test2, y_test], [y_test, x_test]], callbacks=[lr_decay, log, checkpoint1, checkpoint2],
                                  initial_epoch=int(args.ep_num),
                                  shuffle=True)


    
    
    parallel_model.save(args.save_dir + '/trained_model_multi_gpu.h5')
    model.save(args.save_dir + '/trained_model.h5')

    return parallel_model

def test(eval_model, data):

    (x_train, y_train), (x_test, y_test) = data

    # uncommnt and add the corresponding .py and weight to test other models
    # m1 = imp.load_source('module.name', args.save_dir+"/deepcaps.py")
    # _, eval_model = m1.DeepCapsNet28(input_shape=x_test.shape[1:], n_class=10, routings=3)
    eval_model.load_weights(args.save_dir+"/best_weights_2.h5")
    a1, b1 = eval_model.predict(x_test)
    p1 = np.sum(np.argmax(a1, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    print('Test acc:', p1)
    return a1, b1


class args:
    numGPU = 1
    epochs = 20
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

os.makedirs(args.save_dir, exist_ok=True)
try:
    os.system("cp deepcaps.py " + args.save_dir + "/deepcaps.py")
except:
    print("cp deepcaps.py " + args.save_dir + "/deepcaps.py")


num=70000
# load data
if(args.dataset == "CIFAR100"):
    (x_train, y_train), (x_test, y_test) = load_cifar100()
    x_train = resize(x_train, 64)
    x_test = resize(x_test, 64)
elif(args.dataset == "CIFAR10"):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    x_train = resize(x_train, 64)
    x_test = resize(x_test, 64)
elif(args.dataset == "MNIST"):
    (x_train, y_train), (x_test, y_test) = load_mnist()
elif(args.dataset == "FMNIST"):
    (x_train, y_train), (x_test, y_test) = load_fmnist()
elif(args.dataset == "SVHN"):
    (x_train, y_train), (x_test, y_test) = load_svhn()
    x_train = resize(x_train, 64)
    x_test = resize(x_test, 64)
else:
    s,t,v= load_tiny_imagenet("D:/haj/dataset/HGG/", 5)
    ind = np.arange(s.shape[0])
    np.random.shuffle(ind)
    s_=s[ind,:,:,:]
    t_=t[ind,:,:,:]
    v_=v[ind,:]
    x_train1=s_[0:num,:,:,:]
    x_train2=t_[0:num,:,:,:]
    y_train1 =v_[0:num,:]
    
    x_test1=s_[num:,:,:,:]
    x_test2=t_[num:,:,:,:]
    y_test1 =v_[num:,:]



#model, eval_model = DeepCapsNet(input_shape=x_train1.shape[1:], n_class=y_train1.shape[1], routings=args.routings)  # for 64*64
model, eval_model = DeepCapsNetTwoPath(input_shape1=x_train1.shape[1:],input_shape2=x_train2.shape[1:], n_class=y_train1.shape[1], routings=args.routings)  #for 28*28

# plot_model(model, show_shapes=True,to_file=args.save_dir + '/model.png')



################  training1  #################  
appendix = ""
# model.load_weights(args.save_dir + '/best_weights_2' + appendix + '.h5')

train(model=model, data=((x_train1, y_train1),(x_train2), (x_test1, y_test1), (x_test2)), hard_training=False, args=args)

model.load_weights(args.save_dir + '/best_weights_2' + appendix + '.h5')
appendix = "x"
train(model=model, data=((x_train1, y_train1),(x_train2), (x_test1, y_test1), (x_test2)), hard_training=True, args=args)
#############################################


################  training2  #################  

# s,v= load_tiny_imagenet("D:/haj/dataset/HGG/", 5)
# x_train1=s[0:num,:,:,:]
# y_train1 =v[0:num,:]
# x_test1=s[num:,:,:,:]
# y_test1 =v[num:,:]
# appendix = ""
# model.load_weights(args.save_dir + '/best_weights_2' + appendix + '.h5')
# train(model=model, data=((x_train1, y_train1), (x_test1, y_test1)), hard_training=False, args=args)

# appendix = "x"
# model.load_weights(args.save_dir + '/best_weights_2' + appendix + '.h5')

# train(model=model, data=((x_train1, y_train1), (x_test1, y_test1)), hard_training=True, args=args)
#############################################



################# testing  ################# 


# test(eval_model, ((x_train1, y_train1), (x_test1, y_test1)))

###########################################


################# show testing  #################  

# new_x_test,new_y_test=load_test_tiny_imagenet()
# model, eval_model = DeepCapsNet(input_shape=new_x_test.shape[1:], n_class=new_y_test.shape[1], routings=args.routings)
# xTest=new_x_test.astype('float32')
# yTest=new_y_test.astype('float32')
# w=91
# h=176
# a1,b1=test(eval_model, ((xTest,yTest), (xTest,yTest)))

# new_y=np.zeros((16016,1))
# m=0
# for i in range(0,16016):
#     if a1[i,0]>=a1[i,1] and a1[i,0]>=a1[i,2]  and a1[i,0]>=a1[i,3] and a1[i,0]>=a1[i,4] :
#         m=0
#     elif a1[i,1]>=a1[i,0]  and a1[i,1]>=a1[i,2]  and a1[i,1]>=a1[i,3] and a1[i,1]>=a1[i,4] :
#         m=1
#     elif a1[i,2]>=a1[i,0]  and a1[i,2]>=a1[i,1]  and a1[i,2]>=a1[i,3] and a1[i,2]>=a1[i,4] :
#         m=2
#     elif a1[i,3]>=a1[i,0]  and a1[i,3]>=a1[i,1]  and a1[i,3]>=a1[i,2] and a1[i,3]>=a1[i,4] :
#         m=3
#     elif a1[i,4]>=a1[i,0]  and a1[i,4]>=a1[i,1]  and a1[i,4]>=a1[i,2] and a1[i,4]>=a1[i,3] :
#         m=4
#     new_y[i,0]=m
    
# print(new_y)
# y2 =new_y
# pred=y2.reshape(w,h)
# plt.imshow(pred)
# plt.show()

##############################################

