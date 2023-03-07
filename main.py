import sys
import os
import yaml
import torch
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import make_dir, plot_trainning_curve, plot_roc, plot_lr, setup_seed
from data import SPECT, resize3D, get_spect_sample_list, SPECT_LAD, SPECT_LCX, SPECT_RCA
from models import SwinTransformer, ViT, ChannelAttentionSwinTransformer, eca_resnet10
from prefetch_generator import BackgroundGenerator
import argparse


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



def main(args):
    stream =  open(args.config, "r")
    config = yaml.load(stream, Loader=yaml.Loader)
    
    if config['trainer']['name'] == "binary_classification":
        from trainer import binary_classification_train as train
        from trainer import binary_classification_test as test
    else:
        sys.exit("ConfigValueError: [trainer name] \""+config['trainer']['name']+"\" was not defined.") 
    
    
    if config['seed'] != 'None':
        setup_seed(config['seed'])
    preprocess = args.preprocess
    debug = args.debug
    save_path = make_dir(config['save_path'], config['exp_name'])
    if preprocess:
        image_path = resize3D(config['data_path'], store_path=config['tmp_path'], preprocess=preprocess)
    else:
        image_path = config['data_path']

    gt_all = []
    prob_all = []
    for fold in range(1, config['fold'] + 1):

        print()
        print('fold: ',fold)

        train_list, val_list = get_spect_sample_list(config['sample_path'], fold)

        if config['dataset']['name'] == "SPECT":
            train_dataset = SPECT(image_path, train_list, **config['dataset']['args'])
            val_dataset = SPECT(image_path, val_list, **config['dataset']['args'])
        elif config['dataset']['name'] == "SPECT_LAD":
            train_dataset = SPECT_LAD(image_path, train_list, **config['dataset']['args'])
            val_dataset = SPECT_LAD(image_path, val_list, **config['dataset']['args'])
        elif config['dataset']['name'] == "SPECT_LCX":
            train_dataset = SPECT_LCX(image_path, train_list, **config['dataset']['args'])
            val_dataset = SPECT_LCX(image_path, val_list, **config['dataset']['args'])
        elif config['dataset']['name'] == "SPECT_RCA":
            train_dataset = SPECT_RCA(image_path, train_list, **config['dataset']['args'])
            val_dataset = SPECT_RCA(image_path, val_list, **config['dataset']['args'])
        else:
            sys.exit("ConfigValueError: [dataset]") 

        
        train_data_size = len(train_dataset)
        val_data_size = len(val_dataset)
        print("train data size: {}".format(train_data_size))
        print("val data size: {}".format(val_data_size))

        if config['num_workers']>0:
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'], pin_memory=True, persistent_workers=True)
        else:
            train_loader = DataLoaderX(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'], pin_memory=True)
        
        val_loader = DataLoaderX(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        
        dataloaders={
            'train':train_loader,
            'val':val_loader
        }

        if config['model']['name'] == 'SwinTransformer':
            model = SwinTransformer(**config['model']['args'])
        elif config['model']['name'] == 'ViT':
            model = ViT()
        elif config['model']['name'] == 'ChannelAttentionSwinTransformer':
            model = ChannelAttentionSwinTransformer(**config['model']['args'])
        elif config['model']['name'] == 'eca_resnet10':
            model = eca_resnet10(**config['model']['args'])
        else:
            sys.exit("ConfigValueError: [model] \""+config['model']['name']+"\" was not defined.") 


        if config['loss'] == 'BCEWithLogitsLoss':
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            sys.exit("ConfigValueError: [loss] \""+config['loss']+"\" was not defined.") 
        # loss_fn = nn.MSELoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']))
        else:
            sys.exit("ConfigValueError: [optimizer] \""+config['optimizer']+"\" was not defined.")     

        if config['lr_scheduler'] == 'None':
            # lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = 25, max_epochs = args.epoch)
            lr_scheduler=None
        elif config['lr_scheduler'] == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
        else:
            sys.exit("ConfigValueError: [lr_scheduler] \""+config['lr_scheduler']+"\" was not defined.")    


        history = train(model, dataloaders, loss_fn, optimizer, lr_scheduler, config['epochs'],
                save_path=save_path, device=config['device'], class_name=config['class_name'],  fold=fold, patience=config['patience'], debug=debug, monitor=config['trainer']['args']['monitor'])

        plot_lr(history, save_path, fold)
        plot_trainning_curve(history, save_path, fold)

        gt, prob, _ = test(model, val_loader, checkpoint_path=save_path, device=config['device'], fold=fold, debug=debug)

        plot_roc(gt, prob, config['class_name'], save_path, fold=fold)

        gt_all.append(gt)
        prob_all.append(prob)

    gt_all = np.concatenate(gt_all)
    prob_all = np.concatenate(prob_all)
    plot_roc(gt_all, prob_all, config['class_name'], save_path, fold=-1)
    with open(os.path.join(save_path, 'exp_config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__  == '__main__':    
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-d', '--debug', type=bool, default=False)
    parser.add_argument('-p', '--preprocess', type=bool, default=False)
    parser.add_argument('-c', '--config', type=str, default="config.yaml")
    args = parser.parse_args()
    
    main(args)