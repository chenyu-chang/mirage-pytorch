# MIRAGE Pytorch for deep learning experiment

PyTorch deep learning template structure for clear experimental records.


## Features
* Clear folder structure for making custom functions.
* `config.yaml`  used to configure training parameters.
* `exp_config.yaml` stores the parameters for each experiment.
* `Checkpoint.pth` saved the model weights, best metrics and optimizer.
* `history.pth` records the training process.

## Folder Structure
  ```
  mirage-pytorch/
  │
  ├── main.py - main script to start training
  ├── config.yaml - holds configuration for training
  │
  ├── data/ - custom datasets and data preprocessing
  │   ├── get_sample_list.py
  │   ├── resize3D.py
  │   └── spect_dataset.py
  │
  ├── model/ - models
  │   ├── swintransformer3D.py
  │   └── vit.py
  │
  ├── trainer/ - custom training function
  │   └── multilabel_classification.py - train and test
  │
  └── utils/ - small utility functions
      ├── make_dir.py
      ├── plot_lr.py
      ├── plot_roc.py
      └── plot_trainning_curve.py
  ```

## Usage
1. Edit the `config.yaml` file: Set directory path and training parameters.
2.  Run the code.
```
python main.py
```
* Or run code with image preprocessing.
```
python main.py -p true
```
* Or run the code in debug mode.
```
python main.py -d true
```

### Config file format
Config files are in `.yaml` format:
```yaml
data_path: D:/SPECT/Second_year_data/SPECT_long_axis_Pre_Test_GT_noerode_0725
tmp_path: D:/SPECT/Second_year_data/.SPECT_temp1108 # Temporary directory to store preprocessed images
sample_path: D:/SPECT/Second_year_data/sample_group_0329_no_reg
save_path: D:/mirage-pytorch/exp/swin-transformer_3D
exp_name: "1108" # Can be set to ""
model: 
  name: SwinTransformer
  args: 
      in_chans: 4
      embed_dim: 48
      patch_size: !!python/tuple
      - 2
      - 2
      - 2
      window_size: !!python/tuple
      - 7
      - 7
      - 7
device: cuda:0
num_workers: 0
fold: 3 # -1 means no folder
batch_size: 10
shuffle: true
optimizer: Adam
lr: 1e-05
loss: BCEWithLogitsLoss
epochs: 3
patience: 10
lr_scheduler: StepLR
class_name: !!python/tuple
- LAD
- LCX
- RCA
```
