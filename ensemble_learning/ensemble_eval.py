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
from utils import make_dir, plot_roc
from data import SPECT, get_spect_sample_list
from models import SwinTransformer, AverageEnsembleModel
from prefetch_generator import BackgroundGenerator
import argparse
from tqdm import tqdm


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def multilabel_classification_test(model:nn.Module, dataloaders:DataLoader, device:str, fold:int=-1, debug:bool=False):
    
    print('eval :')
    model=model.to(device)
    m = nn.Sigmoid().to(device)
    model.eval()
    with torch.no_grad():
        c=0

        gt =[]
        prob=[]
        for inputs, targets in tqdm(dataloaders):
            c +=1
            if c>10 and debug:
                break

            inputs = inputs.to(device)
            outputs = m(model(inputs))

            gt.append(targets)
            prob.append(outputs)

        gt = torch.cat(gt).cpu().numpy()
        prob = torch.cat(prob).cpu().numpy()
        pred = np.copy(prob)
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

    return gt, prob, pred



def main(config:dict, args):
    debug = args.debug
    save_path = make_dir(config['save_path'], config['exp_name'])
    image_path = config['data_path']

    gt_all = []
    prob_all = []
    for fold in range(1, config['fold'] + 1):

        print()
        print('fold: ',fold)

        _, val_list = get_spect_sample_list(config['sample_path'], fold)
        val_dataset = SPECT(image_path, val_list)
        val_data_size = len(val_dataset)
        print("val data size: {}".format(val_data_size))

        val_loader = DataLoaderX(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        
        model1 =  SwinTransformer(window_size = [8, 8, 5],depths = [2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\two_layers_blur\1116_inputsize323218_blur_window_size_885_exp1', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model1.load_state_dict(checkpoint['state_dict'])

        model2 =  SwinTransformer(window_size = [4, 4, 3],depths = [2, 2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\three_layers_blur\1116_inputsize323218_blur_window_size_443_exp0', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model2.load_state_dict(checkpoint['state_dict'])


        model3 =  SwinTransformer(window_size = [2, 2, 2],depths = [2, 2, 2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\four_layers_blur\1116_inputsize323218_blur_window_size2_exp0', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model3.load_state_dict(checkpoint['state_dict'])

        model4 =  SwinTransformer(embed_dim=64, patch_size=[4, 4, 3], window_size = [4, 4, 3],depths = [2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\two_layers_blur\1118_inputsize323218_blur_window_size_443_patch_size_443_exp1', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model4.load_state_dict(checkpoint['state_dict'])

        model5 =  SwinTransformer(embed_dim=64, patch_size=[4, 4, 3], window_size = [2, 2, 2],depths = [2, 2, 2])
        checkpoint = torch.load(os.path.join(r'D:\mirage-pytorch\exp\swin-transformer_3D\three_layers_blur\1118_inputsize323218_blur_window_size_2_patch_size_443_exp0', 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
        model5.load_state_dict(checkpoint['state_dict'])

        model = AverageEnsembleModel([model1, model2, model3, model4, model5])

        gt, prob, _ = multilabel_classification_test(model, val_loader, device=config['device'], fold=fold, debug=debug)

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