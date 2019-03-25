from __future__ import print_function
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
from keras.callbacks import EarlyStopping
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
#import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

batch_size = 80
num_classes = 2
epochs = 150
calc_times=10

# Input image dimensions
img_rows, img_cols = 50, 50

# Set the number of images of class1 and class2 (sample 1 and sample 2).
class1_num=400
class2_num=400
class1_train_num=320
class2_train_num=320
class1_test_num=80
class2_test_num=80

#Output file to show accuracies of classification averaged over 10 calculaion with the same sample.
f=open('pair_cnn_acc.txt','w')

#Classifications are paerformed from sample1 to sample 18 (ID of each sample is shown in image files).
for i_label in range (1,18):
    for j_label in range (i_label+1,19):

        x_class1=np.array([[[[0.0 for i in range(img_rows)] for j in range(img_cols)]]for k in range(class1_num)])
        x_class2=np.array([[[[0.0 for i in range(img_rows)] for j in range(img_cols)]]for k in range(class2_num)])
        t_train=np.array([],dtype=int)
        t_test=np.array([],dtype=int)
        acc_each_cnn=np.array([],dtype=float)
        
        #Input of training images.
        label1=str(i_label)
        label2=str(j_label)
        dir1="./"+label1+"/*.jpg"
        dir2="./"+label2+"/*.jpg"
        files_1_train=glob.glob(dir1)
        files_2_train=glob.glob(dir2)


        name_num=0
        for file_1_train in files_1_train:
            im=Image.open(file_1_train)
            rgb_im=im.convert('RGB')
            size=rgb_im.size
    
            for x in range(size[0]):
                for y in range(size[1]):
                    r,g,b=rgb_im.getpixel((x,y))
                    x_class1[name_num][0][x][y]=r/255.0
            name_num+=1

        name_num=0
        for file_2_train in files_2_train:
            im=Image.open(file_2_train)
            rgb_im=im.convert('RGB')
            size=rgb_im.size
    
            for x in range(size[0]):
                for y in range(size[1]):
                    r,g,b=rgb_im.getpixel((x,y))
                    x_class2[name_num][0][x][y]=r/255.0
            name_num+=1

        for i in range(0, class1_train_num):
            t_train=np.append(t_train,0)
    
        for i in range(0, class2_train_num):
            t_train=np.append(t_train,1)

        for i in range(0, class1_test_num):
            t_test=np.append(t_test,0)
        
        for i in range(0, class2_test_num):
            t_test=np.append(t_test,1)
            
        t_train = keras.utils.to_categorical(t_train, num_classes)
        t_test = keras.utils.to_categorical(t_test, num_classes)

        calc_num=0.0
        while calc_num<calc_times:
            #Split 400 images into 320 trainig images and 80 test images.
            x_train_class1,x_test_class1=train_test_split(x_class1,test_size=0.2,random_state=None)
            x_train_class2,x_test_class2=train_test_split(x_class2,test_size=0.2,random_state=None)
            x_train=np.append(x_train_class1,x_train_class2,axis=0)
            x_test=np.append(x_test_class1,x_test_class2,axis=0)

            print(x_train.shape," ",x_test.shape)
            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
                x_test = x_class1.reshape(x_test.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1)

            #Augumentation of trainig images.
            datagen = ImageDataGenerator(rotation_range=180.0,horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')


            #Construction of CNN archtecture.
            model = Sequential()
            model.add(Conv2D(30,kernel_size=(5, 5),
                             activation='relu',
                             input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(80,activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
    
            model.summary()
            model.fit_generator(datagen.flow(x_train, t_train, batch_size=batch_size),
                                steps_per_epoch=10,
                                nb_epoch=epochs,
                                verbose=1,
                                validation_data=(x_test, t_test))

            score=model.evaluate(x_test,t_test,batch_size=160)
            #To remove the error of leraning (the case that the lerning does not procees), the resutls does not count when the accuracy is completely 0.50.
            if abs(score[1]-0.5)>0.001:
                acc_each_cnn=np.append(acc_each_cnn,score[1])
                calc_num+=1.0
        
    
        print(acc_each_cnn)
        #Calculation of average of accuracy over 10 calclations.
        ave_acc=0.0
        for i in range(0,calc_times):
            ave_acc+=acc_each_cnn[i]

        ave_acc/=(1.0*calc_times)
        print(ave_acc)
        f.write(str(i_label)+" "+str(j_label)+" "+str(ave_acc)+"\n")

f.close()
