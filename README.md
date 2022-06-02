# Fed-Tuner
FederatedLearning with flower and slice tuner

writed by [gggangmin](https://github.com/gggangmin/),[parkjungha](https://github.com/parkjungha/), [sperospera1225](https://github.com/sperospera1225/)

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
