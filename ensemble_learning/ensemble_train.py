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
import sys
sys.path.append("..")
from trainer import multilabel_classification_train, multilabel_classification_test
from utils import make_dir, plot_trainning_curve, plot_roc, plot_lr
from data import SPECT, get_spect_sample_list
from models import SwinTransformer, LinearEnsembleModel, ConvEnsembleModel
from prefetch_generator import BackgroundGenerator
import argparse


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



def main(config:dict, args):
    debug = args.debug
    save_path = make_dir(config['save_path'], config['exp_name'])
    image_path = config['data_path']

    gt_all = []
    prob_all = []
    for fold in range(1, config['fold'] + 1):

        print()
        print('fold: ',fold)

        train_list, val_list = get_spect_sample_list(config['sample_path'], fold)

        
        train_dataset = SPECT(image_path, train_list)
        val_dataset = SPECT(image_path, val_list)

        
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

        model1 =  SwinTransformer(window_size = [8, 8, 5],depths = [4, 4])
        checkpoint = torch.load(os.path.join('D:/mirage-pytorch/exp/swin-transformer_3D/two_layers/1110_inputsize323218_window_size885_exp0', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model1.load_state_dict(checkpoint['state_dict'])

        model2 =  SwinTransformer(window_size = [4, 4, 3],depths = [2, 2, 2])
        checkpoint = torch.load(os.path.join('D:/mirage-pytorch/exp/swin-transformer_3D/three_layers/1111_inputsize323218_window_size443_exp2', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model2.load_state_dict(checkpoint['state_dict'])


        model3 =  SwinTransformer(window_size = [2, 2, 2],depths = [2, 2, 2, 2])
        checkpoint = torch.load(os.path.join('D:/mirage-pytorch/exp/swin-transformer_3D/four_layers/1115_inputsize323218_window_size2_exp0', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model3.load_state_dict(checkpoint['state_dict'])

        model4 =  SwinTransformer(embed_dim=64, patch_size=[4, 4, 3], window_size = [4, 4, 3],depths = [2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\two_layers_blur\1118_inputsize323218_blur_window_size_443_patch_size_443_exp1', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model4.load_state_dict(checkpoint['state_dict'])

        model5 =  SwinTransformer(embed_dim=64, patch_size=[4, 4, 3], window_size = [2, 2, 2],depths = [2, 2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\three_layers_blur\1118_inputsize323218_blur_window_size_2_patch_size_443_exp0', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model5.load_state_dict(checkpoint['state_dict'])

        if config['model']['name'] == 'LinearEnsembleModel':
            model = LinearEnsembleModel([model1, model2, model3, model4, model5], class_num=3)
        elif config['model']['name'] =='ConvEnsembleModel':
            model = ConvEnsembleModel([model1, model2, model3, model4, model5], class_num=3)
        else:
            sys.exit("ConfigValueError: [model][name] \""+config['model']['name']+"\" was not defined.")     
        
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True    



        if config['loss'] == 'BCEWithLogitsLoss':
            loss_fn = nn.BCEWithLogitsLoss()
        elif config['loss'] == 'MSELoss':
            loss_fn = nn.MSELoss()
        else:
            sys.exit("ConfigValueError: [loss] \""+config['loss']+"\" was not defined.") 
        # loss_fn = nn.MSELoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=0.1)
        else:
            sys.exit("ConfigValueError: [optimizer] \""+config['optimizer']+"\" was not defined.")     

        if config['lr_scheduler'] == 'None':
            # lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = 25, max_epochs = args.epoch)
            lr_scheduler=None
        elif config['lr_scheduler'] == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
        else:
            sys.exit("ConfigValueError: [lr_scheduler] \""+config['lr_scheduler']+"\" was not defined.")    


        history = multilabel_classification_train(model, dataloaders, loss_fn, optimizer, lr_scheduler, config['epochs'],
                save_path=save_path, device=config['device'], class_name=config['class_name'],  fold=fold, patience=config['patience'], debug=debug)

        plot_lr(history, save_path, fold)
        plot_trainning_curve(history, save_path, fold)

        gt, prob, _ = multilabel_classification_test(model, val_loader, checkpoint_path=save_path, device=config['device'], fold=fold, debug=debug)

        plot_roc(gt, prob, config['class_name'], save_path, fold=fold)

        gt_all.append(gt)
        prob_all.append(prob)

    gt_all = np.concatenate(gt_all)
    prob_all = np.concatenate(prob_all)
    plot_roc(gt_all, prob_all, config['class_name'], save_path, fold=-1)
    with open(os.path.join(save_path, 'exp_config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__  == '__main__':
    stream =  open("config.yaml", "r")
    config = yaml.load(stream, Loader=yaml.Loader)
    
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    
    main(config, args)