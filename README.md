### ICCV 2021 Workshop SSLAD Track 3 - Continual Learning

This repository provides the code required to participate in the
thirth track of the SSLAD ICCV 2021 workshop, about Continual
Learning. The code is build with the Avalanche framework, making
it easy to compare and setup very customizable benchmarks.

#### Install instructions 

*Tested with python 3.9, pytorch 1.9.0, torchvision 0.10.0 and 
cuda toolkit 10.2* 

1. Clone the Avalanche Fork [here](https://github.com/VerwimpEli/avalanche) \
`git clone https://github.com/VerwimpEli/avalanche.git`
2. Create a (new) conda environment with PyTorch and torchvision installed. \
`conda create -n sslad python=3.9` \
`conda activate sslad` \
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c conda-forge`
3. Update your conda environment with the provided .yml file. \
`conda env update --file env.yml`
4. Then, make sure the Avalanche directory is on your PYTHONPATH. \
`export PYTHONPATH=[Avalanche directory]:$PYTHONPATH`

*Any questions, problems... can be mailed to 
eli.verwimp(at)kuleuven.be, or by creating a git issue.* 

### Avalanche instructions

Implementing a new strategy in Avalanche works by implementing a
subclass of the `StrategyPlugin`. This plugin is passed through 
when updating the model and its callback methods are called during training
of the model. For examples of how such strategies work, see the `training/plugins` 
folder inside Avalanche for some well known methods. For a more in depth explanation, 
see the tutorials on the [avalanche website](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training).

**Don't hesitate to contact me at eli.verwimp(at)kuleuven.be if you don't immediately
know how to implement your method in Avalanche. It's easy, but learning a new framework 
can be challenging sometimes.**

### Notes
* Currently PyTorch with Python 3.9 is giving a warning about leaking thread pools 
when using multiple workers. This is expected. See [this](https://github.com/pytorch/pytorch/issues/60171)
  issue.
    
* The current installation command on the PyTorch website doesn't work. You should
add the `conda-forge` channel for installing.
