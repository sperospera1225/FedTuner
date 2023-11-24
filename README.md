# Fed-Tuner
Federated Learning with flower and slice tuner

writed by [gggangmin](https://github.com/gggangmin/),[parkjungha](https://github.com/parkjungha/), [sperospera1225](https://github.com/sperospera1225/)

---------------------------------------------------------------------------
### experiments.ipynb
experiment source code for naive federated learning test with Fedtuner 
<img width="684" alt="Screenshot 2023-11-24 at 1 44 17 PM" src="https://github.com/sperospera1225/selective_data_federated_learning/assets/67995592/a5812e6a-4b15-4f7a-8e5e-073c3b678749">


---------------------------------------------------------------------------
## Fedtuner
### server.py
server module based on flower, which consist of selective acquisition optimizer and model aggregation
### client_ms.py
client module based on flower, which consist of local model estimator and data acqusition
### cnn.py
simple cnn model for training from client and evaluation from server

The detailed algorithm is as follows:
<img width="672" alt="Screenshot 2023-11-24 at 1 44 27 PM" src="https://github.com/sperospera1225/selective_data_federated_learning/assets/67995592/353bc6d2-eb69-4610-87af-df9b600dc660">

---------------------------------------------------------------------------
### dataset
FashionMNIST

>note: The result has been recognized by Korean Information Science Society Conference Excellence Paper Award.
