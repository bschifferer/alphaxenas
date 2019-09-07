# AlphaXENAS - more efficient neural architecture search

AlphaXENAS combines multiple techniques to find neural architectures for convolutional neural networks based on refinforcement learning:
<li> NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING</b><br> Barret Zoph, Quoc V. Le 2016 [Link](https://arxiv.org/abs/1611.01578)
<li> Efficient Neural Architecture Search via Parameter Sharing</b><br> Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean 2018 [Link](https://arxiv.org/abs/1802.03268)
<li> AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search</b><br>Linnan Wang, Yiyang Zhao, Yuu Jinnai, Rodrigo Fonseca [Link](https://arxiv.org/abs/1805.07440)

It uses Monte Carlo Tree Search guided by a LSTM neural network. The CNN shares weight to improve convergence time.

## Dependency

The requirement.txt contains all dependencies - main required libraries are:

```
- tensorflow
- numpy
- scipy
- matplotlib
- papermill
```
 and can be installed via
 
```
pip install -r requirement.txt
```

Pretrained data/models are available on AWS S3. The jupyter notebook will download and unzip the data.

<a href='https://s3-eu-west-1.amazonaws.com/bsopenbucket/e6040/data.zip'>Training data</a><br>
<a href='https://s3-eu-west-1.amazonaws.com/bsopenbucket/e6040/exp.zip'>Pretrained Models</a><br>
<a href='https://s3-eu-west-1.amazonaws.com/bsopenbucket/e6040/tensorboard_log.zip'>Tensorboard Logs</a>

## Pretrained data

The pretrained data contains tensorboard visualizations, which can be viewed via

```
tensorboard --logdir='./tensorboard_log/'
```

The models contain log files in './exp/modelname/'. For example:<br>
 
<a href="./src/logs.txt">Logs.txt</a><br>

## Dataset

The experiments use CIFAR10 dataset.

## Structure
```
- alphaxenas contains all helper function and model
-- controller_coach.py defines coach
-- controller_lstm.py defines LSTM network for coach
-- controller_mcts.py defines MCTS for coach
-- child_model.py defines CNN network as child
-- data_utils.py contains data loading and data augmentation
-- model_utils.py contains helper functions
- data contains data for training
- exp contains output of experiments
- tensorboard_logs contains the logs for tensorboard
- AlphaXENAS.ipynb is main jupyter notebook for training coach and documents all results
- DenseNet.ipynb is a jupyter notebook, which implemented DenseNet (Benchmark) (as this is not the main project this is just a proof-of-concept
```

## Running the model

Training takes upto 4 days for one model, therefore papermill was used for running AlphaXENAS.ipynb 

```
- papermill AlphaXENAS.ipynb AlphaXENAS_output.ipynb 
```
