# MIRAGE Pytorch for deep learning experiment

PyTorch deep learning template for clear experimental records.


## Features
* Clear folder structure for making custom functions.
* `config.yaml`  used to configure training parameters.
* `exp_config.yaml` stores the parameters for each experiment.
* `Checkpoint.pth` saved the model weights, best metrics and optimizer.
* `history.pth` records the training process.


## Usage
1. Edit the `config.yaml` file: Set directory path and training parameters.
2.  Run the code.
```
python main.py -c <config file path>
```
* Or run code with image preprocessing.
```
python main.py -c <config file path> -p true
```
* Or run the code in debug mode.
```
python main.py -c <config file path> -d true
```
