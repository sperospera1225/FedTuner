# Fed-Tuner
FederatedLearning with flower and slice tuner

---------------------------------------------------------------------------
### experiments.ipynb
experiment source code for naive federated learning test with Fedtuner 

---------------------------------------------------------------------------
## Fedtuner
### server.py
server module based on flower, which consist of selective acquisition optimizer and model aggregation
### client_ms.py
client module based on flower, which consist of local model estimator and data acqusition
### cnn.py
simple cnn model for training from client and evaluation from server

---------------------------------------------------------------------------
### dataset
FashionMNIST
