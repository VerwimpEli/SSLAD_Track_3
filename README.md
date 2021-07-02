### ICCV 2021 Workshop SSLAD Track 3 - Continual Learning

This repository provides the code required to participate in the
thirth track of the SSLAD ICCV 2021 workshop, about Continual
Learning. The code is build with the Avalanche framework, making
it easy to compare and setup very customizable benchmarks.

#### Instructions 

_Tested with python 3.9, pytorch 1.9.0, torchvision 0.10.0 and 
cuda toolkit 10.2_

Step one is to clone the Avalanche Fork 
[here](https://github.com/VerwimpEli/avalanche). Later, we will
move to the current release of Avalanche. Then, make sure
the Avalanche directory is on your PYTHONPATH.

Step two is to get a conda environement with PyTorch and
torchvision running.

Step three is to update the conda enivornment, using the 
`env.yml` file in this repo. 

Any questions, problems... can be mailed to 
eli.verwimp (at) kuleuven.be . 

#### File overview

`class_strategy.py`: Provides an empty plugin. Here, you can define
your own strategy, by implementing the necessary callbacks. Helper
methods can be ofcourse implemented as pleased.

`classification.py`: Driver code for the classification subtrack. 
There are a few things that can be changed here, such as the
model, optimizer and loss criterion. 

`haitain.py`: This file contains all code to get the challenge
and data setup, shouldn't be touched or altered.

#### Avalanche instructions

Implementing a new strategy in Avalanche works by implementing a
subclass of the `StrategyPlugin`. This plugin is passed through 
when updating the model and its callback methods are called during training
of the model. For examples of how such strategies work, see the `training/plugins` 
folder inside Avalanche for some well known methods. For a more in depth explanation, 
see the tutorials on the [avalanche website](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training).

#### Notes
* Currently PyTorch with Python 3.9 is giving a warning about leaking thread pools 
when using multiple workers. This is expected. See [this](https://github.com/pytorch/pytorch/issues/60171)
  issue.
    
* The current installation command on the PyTorch website doesn't work. You should
add the `conda-forge` channel for installing.