from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})

import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
import warnings

import flwr as fl
import sys
import pickle
import os
import tensorflow as tf
from scipy.optimize import curve_fit
from cnn import *
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np 

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# client id
CLIENT_ID = int(sys.argv[1])


# #############################################################################
# Initialize global variable
# #############################################################################

global train, val, val_data_dict, data_num_array, add_data_dict, num_class, loss_output, slice_num, slice_desc
# #############################################################################
# Load Fashion-MNIST dataset
# #############################################################################

def shuffle(data, label):
  shuffle = np.arange(len(data))
  np.random.shuffle(shuffle)
  data = data[shuffle]
  label = label[shuffle]
  return data, label

def collect_init_data():
  with open('./dataset/train.pkl','rb') as f:
    x_train = pickle.load(f)
  with open('./dataset/train_label.pkl','rb') as f:
    y_train = pickle.load(f)

  label_tags = {
      0: 'T-Shirt', 
      1: 'Trouser', 
      2: 'Pullover', 
      3: 'Dress', 
      4: 'Coat', 
      5: 'Sandal', 
      6: 'Shirt',
      7: 'Sneaker', 
      8: 'Bag', 
      9: 'Ankle Boot'
  }
  num_class = len(label_tags)

  initial_data_array = []
  val_data_dict = []
  add_data_dict = []

  val_data_num = 500


  for i in range(num_class):
      data_num = 4000
      initial_data_array.append(data_num)
      idx = np.argmax(y_train,axis=1) == i
      val_data_dict.append((x_train[idx][data_num:data_num+val_data_num],y_train[idx][data_num:data_num+val_data_num]))
      add_data_dict.append((x_train[idx][data_num+val_data_num:], y_train[idx][data_num+val_data_num:]))

      if i == 0:
          train_data = x_train[idx][:data_num]
          train_label = y_train[idx][:data_num]
          val_data = x_train[idx][data_num:data_num+val_data_num]
          val_label = y_train[idx][data_num:data_num+val_data_num]
      else:
          val_data = np.concatenate((val_data, x_train[idx][data_num:data_num+val_data_num]), axis=0)
          val_label = np.concatenate((val_label, y_train[idx][data_num:data_num+val_data_num]), axis=0)

      if i == CLIENT_ID:
          train_data = x_train[idx][:data_num]
          train_label = y_train[idx][:data_num]
  train_data, train_label =shuffle(train_data,train_label)
  train = (train_data, train_label)
  val = (val_data, val_label)

  slice_desc = []
  a = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
  for i in range(num_class):
    slice_desc.append('Slice: %s, Number of data: %d' % (a[i], initial_data_array[i]))
    print(slice_desc[i])

  return train, val, val_data_dict, add_data_dict, num_class, slice_desc

train, val, val_data_dict, add_data_dict, num_class, slice_desc = collect_init_data()

# #############################################################################
# Initialize local model 
# #############################################################################
#  define model
model = Sequential()
#conv2d
model.add(Conv2D(4, kernel_size=(3,3), padding = 'same', activation= 'relu', input_shape=(28,28,1)) )
#maxpooling
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#conv2d
model.add(Conv2D(8, kernel_size=(3,3), padding = 'same', activation= 'relu', input_shape=(28,28,1)) )
#maxpooling
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#flatten
model.add(Flatten())
#fc
model.add(Dense(10,activation='softmax'))
model.compile(loss ='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# to make loss of each class
def check_num(labels):
  
    slice_num = dict()
    for j in range(num_class):
        idx = np.argmax(labels, axis=1) == j
        slice_num[j] = len(labels[idx])
        
    return slice_num

# #############################################################################
# train initial data
# #############################################################################

def train_on_subset(model, num_k,num_iter,epoch):
    """ 
    Given the slices, we generate for each slice k random subsets of data to fit a power-law curve.
    
    Args: 
        num_k: Subsets of the train set with different sizes
        num_iter: Number of training times
    """
    batch_size = train[0].shape[0]
    
    for i in range(num_class):
      #1차원 배열 각 차원 요소는 loss output 기록
      loss_output.append([0])
      # 2차원배열 크기는 1,n이며, n의 요소는 slic_num 크기 기록
      slice_num.append([])
    
    loss_dict=dict()
    min_loss = 100
    for i in range(num_iter):
        
        # for epoch 
        hist = model.fit(train[0][:num_k[CLIENT_ID]], train[1][:num_k[CLIENT_ID]],batch_size=batch_size,epochs = epoch,verbose=0,validation_data = (val[0],val[1]))
        slice_num= check_num(train[1][:num_k[CLIENT_ID]])
        
        v_loss = hist.history['val_loss']
        
        for e in range(epoch):
          loss= v_loss[e]
          if min_loss > loss:
            min_loss = loss
            for j in range(num_class):
              loss_dict[j]  = loss
              
        for j in range(num_class):
          loss_output[j] += (loss_dict[j] / num_iter)
          if i == 0:
              slice_num.append(slice_num[j])
    return model.get_weights()

# to split data by class
def collect_data(num_examples):
  train_data = train[0]
  train_label = train[1]
  for i in range(num_class):
      train_data = np.concatenate((train_data, add_data_dict[i][0][:num_examples[i]]), axis=0)
      train_label = np.concatenate((train_label, add_data_dict[i][1][:num_examples[i]]), axis=0)      
      add_data_dict[i]= (np.concatenate((add_data_dict[i][0][num_examples[i]:], add_data_dict[i][0][:num_examples[i]]), axis=0), 
                        np.concatenate((add_data_dict[i][1][num_examples[i]:], add_data_dict[i][1][:num_examples[i]]), axis=0))
  
  train = (train_data, train_label)
  
  
# #############################################################################
# train with data index from server's optimizer
# #############################################################################

def train_after_collect_data(mode,num_examples,num_iter,epoch):
  """ Trains the model after we collect num_examples of data
  
  Args:
      num_examples: Number of examples to collect per slice 
      num_iter: Number of training times
  """
  batch_size = train[0].shape[0]
  
  collect_data(num_examples)
  loss_dict=dict()
  min_loss = 100
  for i in range(num_iter):
        
        # for epoch 
        hist = model.fit(train[0], train[1],batch_size=batch_size,epochs = epoch,verbose=0,validation_data = (val[0],val[1]))
        slice_num= check_num(train[1])
        
        v_loss = hist.history['val_loss']
        
        
        for e in range(epoch):
          loss= v_loss[e]
          if min_loss > loss:
            min_loss = loss
            for j in range(num_class):
              loss_dict[j]  = loss
              
        for j in range(num_class):
          loss_output[j] += (loss_dict[j] / num_iter)
          if i == 0:
              slice_num.append(slice_num[j])        
          else:
              loss_output[j][-1] += (loss_dict[j] / num_iter)
  return model.get_weights()

# #############################################################################
# loss estimator
# #############################################################################

def fit_learning_curve(slice_desc,show_figure):
    """ 
    Fits the learning curve, assuming power-law form 
    
    Args:
        slice_desc: Array for slice description (e.g., Slice: White_Male)
        show_figure: If True, show learning curves for all slices
        
    Returns:
        A: Exponent of power-law equation
        B: of power-law equation
        estimate_loss: estimated loss on given data using power-law equation
    """
    def weight_list(weight):
        w_list = []
        for i in weight:
            w_list.append(1/(i**0.5))
        return w_list        

    def power_law(x, a, b):
        return (b*x**(-a))
    
    A = []
    B = []
    estimate_loss = []
    
    # loop
    xdata = np.linspace(slice_num[CLIENT_ID][0], slice_num[CLIENT_ID][-1], 1000)
    sigma = weight_list(slice_num[CLIENT_ID])
    popt, pcov = curve_fit(power_law, slice_num[CLIENT_ID], loss_output[CLIENT_ID], sigma=sigma, absolute_sigma=True)
    
    A.append(-popt[0])
    B.append(popt[1])
    estimate_loss.append(popt[1] * (data_num_array[CLIENT_ID] ** (-popt[0])))
            
    return A, B, estimate_loss
  
batch_size = 64


# #############################################################################
# flower federated learning client
# #############################################################################

# Define Flower client
class FedTunerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        print('@@@@clieint_fit')
        # parameters (num_examples, weight)
        num_examples, global_weight = parameters # load parameter and index from server
        
        if config['rnd'] == 1:
          # train initial data
          new_weights = train_on_subset(model, config['num_k'],config['num_iter'],config['epoch'])
        else:
          # train with data index from server's optimizer
          model.set_weight(global_weight)
          new_weights = train_after_collect_data(model, num_examples,config['num_iter'],config['epoch'])
        
        #fit learning curve 
        A, B, estimate_loss = fit_learning_curve(slice_desc, False)
        
        
        # return datas and estimated loss info to server
        return (CLIENT_ID, A, B, estimate_loss, new_weights), data_num_array[CLIENT_ID] , {}

    def evaluate(self, parameters, config):
        '''
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
        '''
        return
# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FedTunerClient())