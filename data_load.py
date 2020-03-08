# -*- coding: utf-8 -*-

from keras.datasets import cifar10, cifar100
from sklearn.model_selection import train_test_split
import numpy as np
import keras 
import pandas as pd
import os
import imageio
import numpy as np
import shutil
import zipfile
import wget
from PIL import Image
from sklearn.utils import shuffle

path = os.path.join(os.path.dirname(os.path.realpath('__file__')))
os.chdir(path)

#   Datasets experimental load. With Train-Validation-Test Split.

def get_dataset(ds_name, normalize = True, ratio = 0.15):   
    ''' Get specified dataset with Train-Validation-Test Split.
        ds_name (String)    -> Name of the dataset.
        normalize (Boolean) -> Wherer the dataset is returned normalized or not.
        ration (Float)      -> Equal dataset percentage assigned to Validation and Test.
    '''
    print("Dataset:",ds_name)  
    if ds_name == 'cifar10':
      x_train, x_val, x_test, y_train, y_val, y_test = get_cifar10(ratio)
      normalize = True
    elif ds_name == 'cifar100':
      x_train, x_val, x_test, y_train, y_val, y_test = get_cifar100(ratio)
      normalize = True
    elif ds_name == 'omniglot':   
      x_train, x_val, x_test, y_train, y_val, y_test = get_omniglot(ratio)
      
      x_train=np.expand_dims(x_train, axis=3)
      x_val=np.expand_dims(x_val, axis=3)
      x_test=np.expand_dims(x_test, axis=3)
            
    elif ds_name == 'cifar10_proto_inst':   
      x_train, x_val, x_test, y_train, y_val, y_test = get_cifar10_proto_inst(ratio)
    
    elif ds_name == 'cifar10_proto_class':    
      x_train, x_val, x_test, y_train, y_val, y_test = get_cifar10_proto_class(ratio)
      y_train = np.arange(len(y_train)).tolist()
      y_val = np.arange(len(y_train),len(y_train)+len(y_val)).tolist()
      y_test = np.arange(len(y_train)+len(y_val),len(y_train)+len(y_val)+len(y_test)).tolist()
      
    elif ds_name == 'cifar100_proto_inst':   
      x_train, x_val, x_test, y_train, y_val, y_test = get_cifar100_proto_inst(ratio)
    
    elif ds_name == 'cifar100_proto_class':    
      x_train, x_val, x_test, y_train, y_val, y_test = get_cifar100_proto_class(ratio)
      y_train = np.arange(len(y_train)).tolist()
      y_val = np.arange(len(y_train),len(y_train)+len(y_val)).tolist()
      y_test = np.arange(len(y_train)+len(y_val),len(y_train)+len(y_val)+len(y_test)).tolist()
      
    elif ds_name == 'omniglot_proto_inst':   
      x_train, x_val, x_test, y_train, y_val, y_test = get_omniglot_proto_inst(ratio)
        
    elif ds_name == 'omniglot_proto_class':   
      x_train, x_val, x_test, y_train, y_val, y_test = get_omniglot_proto_class(ratio)
      y_train = np.arange(len(y_train)).tolist()
      y_val = np.arange(len(y_train),len(y_train)+len(y_val)).tolist()
      y_test = np.arange(len(y_train)+len(y_val),len(y_train)+len(y_val)+len(y_test)).tolist()
          
    if 'proto' in ds_name:
      num_classes = int(np.max([np.max(y_train),np.max(y_val),np.max(y_test)])+1)
      y_train = keras.utils.to_categorical(y_train, num_classes)
      y_val = keras.utils.to_categorical(y_val, num_classes)
      y_test = keras.utils.to_categorical(y_test, num_classes)

    if normalize:
        x_train, x_val, x_test = normalize_data(ds_name, x_train, x_val, x_test)
  
    return x_train, x_val, x_test, y_train, y_val, y_test
    
     
#   Complimentary datasets load functions.

def train_val_test_splitter(x, y, ratio, random_state=999):
    ''' Split dataset(X,Y) into Train-Validation-Test chunks.
        ratio (Float) -> Equal dataset percentage assigned to Validation and Test.
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=999)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio/(1-ratio), random_state=999)
    return x_train, x_val, x_test, y_train, y_val, y_test


def normalize_data(ds_name, x_train, x_val, x_test):
    ''' Normalize data for Cifar-10 and Cifar-100 datasets.
    '''
    if ds_name == 'cifar10' or ds_name == 'cifar100':
          x_train = x_train/255
          x_val = x_val/255
          x_test = x_test/255
    return x_train, x_val, x_test

def unzip():
    ''' Unzip downloaded files containing Omniglot dataset.
    '''
    with zipfile.ZipFile('omniglot-master.zip', 'r') as zip_ref:
      zip_ref.extractall()
    with zipfile.ZipFile('omniglot-master/python/images_evaluation.zip', 'r') as zip_ref:
      zip_ref.extractall()
    with zipfile.ZipFile('omniglot-master/python/images_background.zip', 'r') as zip_ref:
      zip_ref.extractall()
  
def parse_images(data):
    ''' Read raw image files and transform data into numpy arrays.
        Returns list of NumPy arrays.
    '''
    images = []
    for img in data:
      im = imageio.imread(img)
      images.append(im)
    return images

def parse_images_resize(data, width = 28, height = 28):
    ''' Read raw image files, resize them to width*height size,
        and transform data into numpy arrays.
        Return list of NumPy arrays.
    '''
    images = []
    for img in data:
      im = Image.open(img)
      im = im.resize((width,height), Image.ANTIALIAS)
      im = np.asarray(im, dtype="int32" )
      images.append(im)
    return images

def clean():
    ''' Delete unnecessary extra files downloaded in base folder. 
    '''
    os.remove('omniglot-master.zip')
    shutil.rmtree('omniglot-master')
    shutil.rmtree('images_evaluation')
    shutil.rmtree('images_background')


#   Normal datasets load functions.

def get_cifar10(ratio):
    ''' Get Cifar-10 dataset with Train-Validation-Test Split.
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
          
    return train_val_test_splitter(X, y, ratio, random_state=999)

def get_cifar100(ratio):
    ''' Get Cifar-100 dataset with Train-Validation-Test Split.
    '''
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
      
    return train_val_test_splitter(X, y, ratio, random_state=999)  

def get_omniglot(ratio):
    ''' Get Omniglot dataset with standard Train-Validation-Test Split.
    '''
    url = 'https://github.com/brendenlake/omniglot/archive/master.zip'
    wget.download(url)
    unzip()

    count = 0
    alphabets, letters, labels = [], [], []   
    for file in os.listdir("images_background"):
        alphabets.append(os.path.join("images_background", file))
    for file in os.listdir("images_evaluation"):
        alphabets.append(os.path.join("images_evaluation", file))
      
    for alpha in alphabets:
      for file in os.listdir(alpha+'/'):
        path = os.path.join(alpha, file)
        for f in os.listdir(path):
          letters.append(path+'/'+f)
          labels.append(int(count))
        count += 1

    images = parse_images_resize(letters)
    clean()
    return  train_val_test_splitter(np.array(images), np.array(labels), ratio, random_state=999) 


#   Prototypical Networks Model adapted load functions.

def get_cifar10_proto_inst(ratio):
    ''' Get Cifar-10 dataset with INSTANCE-BASED Train-Validation-Test Split.
        Adapted for experiments with Prototypical Networks.
    ''' 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    x_train = np.zeros([10, int(len(x)*(1-2*ratio)/10),32,32,3], dtype=np.float32)
    x_val = np.zeros([10, int(len(x)*(ratio)/10),32,32,3], dtype=np.float32)
    x_test = np.zeros([10, int(len(x)*(ratio)/10),32,32,3], dtype=np.float32)

    for cl in np.sort(np.unique(y)):
        x_train[cl] = x[np.where(y.T[0]==cl)[0][:int(len(x)*(1-2*ratio)/10)]]
        x_val[cl] = x[np.where(y.T[0]==cl)[0][int(len(x)*(1-2*ratio)/10):int(len(x)*(1-ratio)/10)]]
        x_test[cl] = x[np.where(y.T[0]==cl)[0][int(len(x)*(1-ratio)/10):]]
                    
    y_train = [i for i in range(10)]
    y_val = [i for i in range(10)]
    y_test = [i for i in range(10)]
              
    return x_train, x_val, x_test, y_train, y_val, y_test
  
def get_cifar10_proto_class(ratio):
    ''' Get Cifar-10 dataset with CLASS-BASED Train-Validation-Test Split.
        Adapted for experiments with Prototypical Networks.
    '''  
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    X_all , y_all =[] , []
    for i in range(np.max(y)+1):
        data=X[np.where(y==i)[0]]
        X_all.append(data)
        y_all.append(np.repeat(i, 1000))
    
    return train_val_test_splitter(np.array([X_all])[0], np.array([y_all])[0], ratio, random_state=999)

def get_cifar100_proto_inst(ratio):
    ''' Get Cifar-100 dataset with INSTANCE-BASED Train-Validation-Test Split.
        Adapted for experiments with Prototypical Networks.
    '''  
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    x_train = np.zeros([100, int(len(X)*(1-2*ratio)/100),32,32,3], dtype=np.float32)
    x_val = np.zeros([100, int(len(X)*(ratio)/100),32,32,3], dtype=np.float32)
    x_test = np.zeros([100, int(len(X)*(ratio)/100),32,32,3], dtype=np.float32)

    for cl in np.sort(np.unique(y)):
        x_train[cl] = X[np.where(y.T[0]==cl)[0][:int(len(X)*(1-2*ratio)/100)]]
        x_val[cl] = X[np.where(y.T[0]==cl)[0][int(len(X)*(1-2*ratio)/100):int(len(X)*(1-ratio)/100)]]
        x_test[cl] = X[np.where(y.T[0]==cl)[0][int(len(X)*(1-ratio)/100):]]
                    
    y_train = [i for i in range(100)]
    y_val = [i for i in range(100)]
    y_test = [i for i in range(100)]
              
    return x_train, x_val, x_test, y_train, y_val, y_test
  
def get_cifar100_proto_class(ratio):
    ''' Get Cifar-100 dataset with CLASS-BASED Train-Validation-Test Split.
        Adapted for experiments with Prototypical Networks.
    '''  
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    X_all , y_all =[] , []
    for i in range(np.max(y)+1):
        data=X[np.where(y==i)[0]]
        X_all.append(data)
        y_all.append(np.repeat(i, 1000))

    return train_val_test_splitter(np.array([X_all])[0], np.array([y_all])[0], ratio, random_state=999)

def get_omniglot_proto_inst(ratio):
    ''' Get Omniglot dataset with INSTANCE-BASED Train-Validation-Test Split.
        Adapted for experiments with Prototypical Networks.
    '''  
    url = 'https://github.com/brendenlake/omniglot/archive/master.zip'
    wget.download(url)
    unzip() 

    count = 0
    alphabets, letters, labels = [], [], [] 
    new_letters, new_labels = [], [] 
    for file in os.listdir("images_background"):
        alphabets.append(os.path.join("images_background", file))
    for file in os.listdir("images_evaluation"):
        alphabets.append(os.path.join("images_evaluation", file))
      
    for alpha in alphabets:
      for file in os.listdir(alpha+'/'):
        path = os.path.join(alpha, file)
        for f in os.listdir(path):
          letters.append(path+'/'+f)
          labels.append(int(count))
        new_letters.append(letters)
        new_labels.append(labels)
        count += 1

    images = parse_images_resize(new_letters[0])   
    X=np.array(images)
    y=np.array(labels)
    x_train = np.zeros([1623, int(20*(1-2*ratio)),28,28], dtype=np.float32)
    x_val = np.zeros([1623, int(20*(ratio)),28,28], dtype=np.float32)
    x_test = np.zeros([1623, int(20*(ratio)),28,28], dtype=np.float32)

    for cl in np.sort(np.unique(y)):
          x_train[cl] = X[np.where(y==cl)[0][:int(20*(1-2*ratio))]]
          x_val[cl] = X[np.where(y==cl)[0][int(20*(1-2*ratio)):int(20*(1-ratio))]]
          x_test[cl] = X[np.where(y==cl)[0][int(20*(1-ratio)):]]          
    y_train = [i for i in range(1623)]
    y_val = [i for i in range(1623)]
    y_test = [i for i in range(1623)]

    clean()
    return  x_train, x_val, x_test, y_train, y_val, y_test  

def get_omniglot_proto_class(ratio):
    ''' Get Omniglot dataset with CLASS-BASED Train-Validation-Test Split.
        Adapted for experiments with Prototypical Networks.
    '''  
    url = 'https://github.com/brendenlake/omniglot/archive/master.zip'
    wget.download(url)
    unzip()  

    count = 0
    alphabets, letters, labels = [], [], [] 
    new_letters, new_labels = [], [] 
    for file in os.listdir("images_background"):
        alphabets.append(os.path.join("images_background", file))
    for file in os.listdir("images_evaluation"):
        alphabets.append(os.path.join("images_evaluation", file))
      
    for alpha in alphabets:
      for file in os.listdir(alpha+'/'):
        path = os.path.join(alpha, file)
        for f in os.listdir(path):
          letters.append(path+'/'+f)
          labels.append(int(count))
        new_letters.append(letters)
        new_labels.append(labels)
        count += 1

    images = parse_images_resize(new_letters[0])   
    images_reshaped = np.array(images).reshape((1623,20, 28,28))
    labels_2 = np.arange(1623)

    clean()
    return  train_val_test_splitter(images_reshaped, labels_2, ratio, random_state=999) 


#   Standard Cifar-10 and Cifar-100 datasets load.

def load_cifar10():
    ''' Get Cifar-10 dataset with standard Train-Test Split.
    '''  
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = np.max(y_test)+1

    x_train = x_train/255
    x_test = x_test/255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def load_cifar100():
    ''' Get Cifar-100 dataset with standard Train-Test Split.
    '''  
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    num_classes = np.max(y_test)+1

    x_train = x_train/255
    x_test = x_test/255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

