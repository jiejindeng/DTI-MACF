# DTI-MACF




## Environment Settings

* Python == 3.6.13
* pytorch == 1.7.1
* numpy == 1.19.5
* scikit-learn == 0.24.2



## Parameter Settings

- epochs: the number of epochs to train
- lr: learning rate
- embed_dim: embedding dimension
- N: a parameter of L0, the default is the number of triples
- droprate: dropout rate
- batch_size: batch size for training

## Data Process
Datasets are available in [`datasets/`](datasets/)

The raw data is processed into bipartite graph through [`preprocess.py`](preprocess.py) as input data
~~~
python preprocess.py
~~~


## Basic Usage
Both training and testing procedures can be achived by the script[`train.py`](train.py)
~~~
python run.py 
~~~

## Hyper-parameters Tuning

There are three key hyper-parameters: *number of components*, *lr* and *embed_dim*.

- number of components: [1, 2, 3, 4, 5]
- lr: [0.005,0.001,0.0015,0.0005]
- embed_dim: [8, 16, 32, 64, 128, 256]



