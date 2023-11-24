# Fed-Tuner
Federated Learning with flower and slice tuner
--------------------------------------------------------------------------

## Fedtuner
### server.py
server module based on flower, which consist of selective acquisition optimizer and model aggregation
### client_ms.py
client module based on flower, which consist of local model estimator and data acqusition
### cnn.py
simple cnn model for training from client and evaluation from server
### experiments.ipynb
experiment source code for naive federated learning test with Fedtuner 

The detailed algorithm is as follows:
<img width="672" alt="Screenshot 2023-11-24 at 1 44 27 PM" src="https://github.com/sperospera1225/selective_data_federated_learning/assets/67995592/353bc6d2-eb69-4610-87af-df9b600dc660">

### dataset
FashionMNIST

The result was acknowledged with the Excellence Paper Award at the Korean Information Science Society Conference. You can find the paper [here](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113242)
