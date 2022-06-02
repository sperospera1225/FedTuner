from webbrowser import get
import flwr as fl
from typing import Callable,Dict,Optional,Tuple
import numpy as np
import pickle
from cnn import *
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np 

global T, num_iter, batch_size,epoch, k, budget, Lambda, cost_func, init_train_len
T= 1
num_iter = 2
epoch =2
num_clients=10
cost_func = [1]*10 # num_class
Lambda = 0.1
budget = 4000


# #############################################################################
# Load Fashion-MNIST test dataset
# #############################################################################
def shuffle(data, label):
  shuffle = np.arange(len(data))
  np.random.shuffle(shuffle)
  data = data[shuffle]
  label = label[shuffle]
  return data, label


def collect_init_data():
  with open('./dataset/test.pkl','rb') as f:
    x_test = pickle.load(f)
  with open('./dataset/test_label.pkl','rb') as f:
    y_test = pickle.load(f)
    
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

  initial_data_array = [4000 for _ in range(num_class)]
  
  
  
  slice_desc = []
  a = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
  for i in range(num_class):
      slice_desc.append('Slice: %s, Number of data: %d' % (a[i], initial_data_array[i]))
  
  return x_test,y_test, num_class, slice_desc

# #############################################################################
# Initialize global model
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

x_test, y_test, num_class, slice_desc = collect_init_data()
val_data_dict=[]
# model
batch_size=x_test.shape[0]
#network = CNN(x_test,y_test,x_test,y_test,val_data_dict,batch_size,epoch=1,lr=0.001,num_class=num_class)



# #############################################################################
# flower federated learning server and configuration
# #############################################################################
if __name__ == "__main__":

    def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
        """Return a function which returns training configurations."""
        
        initial_k = 4000
        num_k = [ initial_k for _ in range(num_clients)]
        
        def fit_config(rnd: int) -> Dict[str, str]:
            """Return a configuration with static batch size and (local) epochs."""
            config = {
              'num_k' : num_k, 'num_iter' : num_iter, 'rnd' : rnd, 'epoch' : epoch
            }
            return config

        return fit_config


    def get_eval_fn(model):
        # model with evaluation
      def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float,float]]:
        print('@@@@@@server_eval')
        
        '''
        (CLINET_ID, A, B, estimate_loss, 불러온 글로벌 웨이트)
        
        model.set_weights(global_weights)
        model.eval(val_Dat)
        
        val_loss, test_accuracy 
        결과 pritn 
        
        # run slice tuner, CLINET_ID, A, B, estimate_loss
        결과 가지고 num_example 생성 (optimization)
        
        num_example,global_weight packaging 해서 다시 전송
        
        '''
        
        return 0, {"err":0}
      return evaluate
    


    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_available_clients=1,
        min_fit_clients =1,
        min_eval_clients = 1,
        eval_fn = get_eval_fn(model),
        on_fit_config_fn = get_on_fit_config_fn()
    )
    
    
    
    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 2},
        strategy=strategy,
    )
