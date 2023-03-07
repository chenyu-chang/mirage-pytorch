import torch
import torch.nn as nn
import time
from tqdm import tqdm
import os
from torch.utils.data.dataloader import DataLoader
import numpy as np
import sys
import copy

def binary_classification_train(model:nn.Module, dataloaders:dict, criterion, optimizer, scheduler, num_epochs:int,
                save_path:str, device:str, class_name:tuple=('LAD', 'LCX', 'RCA'),  monitor="val_loss", fold:int=-1, patience:int=0, debug:bool=False)->dict:
    
    if monitor not in ['val_loss', 'val_acc']:sys.exit("MonitorValueError: \""+ monitor +"\" was not defined.") 

    history = {
        'class_name':class_name,
        'loss':[],
        'val_loss':[],
        'learning_rate':[]
    }
    
    dataset_sizes = {
        'train':len(dataloaders['train'].dataset),
        'val':len(dataloaders['val'].dataset)
    }
    
    
    for name in class_name:
        history[name+'_acc']=[]
        history['val_'+name + '_acc'] = []

    
    class_num = len(class_name)
    model=model.to(device)
    m = nn.Sigmoid().to(device)
    criterion=criterion.to(device)


    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_sen = [0.0]*class_num
    best_spe = [0.0]*class_num
    best_acc = [0.0]*class_num
    patience_count=0
    early_stop = False


    for epoch in range(num_epochs):

        if monitor == 'val_loss':
            print('Epoch {}/{}, Best val loss: {:.4f}'.format(epoch, num_epochs - 1, best_loss))
        elif monitor == 'val_acc':
            print('Epoch {}/{}, Best val acc: {:.4f}'.format(epoch, num_epochs - 1, sum(best_acc)/len(best_acc) ))
        
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            conf_matrix = torch.zeros(class_num, 2,2)
            
            c = 0
            for inputs, labels in tqdm(dataloaders[phase], leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

                c +=1
                if c>10 and debug:
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    if type(criterion) == nn.modules.loss.MSELoss:
                        loss = criterion(m(outputs), labels)
                    else:
                        loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                if type(criterion) != nn.modules.loss.MSELoss:
                    outputs = m(outputs)

                for part_class in range(class_num):
                    for t, p in zip(labels[:, part_class], [1 if x > 0.5 else 0 for x in outputs[:, part_class]]):
                        conf_matrix[part_class, int(t), int(p)] += 1

            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                history['loss'].append(epoch_loss)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_loss'].append(epoch_loss)

            epoch_sen=[0]*class_num
            epoch_spe=[0]*class_num
            epoch_acc=[0]*class_num

            for index, name in enumerate(class_name):
                TP = conf_matrix[index, 1, 1].item()
                TN = conf_matrix[index, 0, 0].item()
                FP = conf_matrix[index, 0, 1].item()
                FN = conf_matrix[index, 1, 0].item()
                epoch_sen[index] = TP/(TP+FN+1e-7)
                epoch_spe[index] = TN/(TN+FP+1e-7)
                epoch_acc[index] = (TP+TN)/(TP+TN+FP+FN)

                if phase == 'train':
                    history[name+'_acc'].append(epoch_acc[index])
                else:
                    history['val_'+name + '_acc'].append(epoch_acc[index])

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            for index, name in enumerate(class_name):
                print(name+'  ', end='')
                print("sen: {:.4f} spe: {:.4f} acc: {:.4f}".format(epoch_sen[index], epoch_spe[index], epoch_acc[index]))


            if phase == 'val':
                if monitor == "val_loss" and epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    best_sen = epoch_sen.copy()
                    best_spe = epoch_spe.copy()
                    best_acc = epoch_acc.copy()
                    # best_model_wts = copy.deepcopy(model.state_dict())  # keep the best validation accuracy model
                    state = {
                        'state_dict' : copy.deepcopy(model.state_dict()),
                        'best_sen' : best_sen,
                        'best_spe' : best_spe,
                        'best_acc' : best_acc,
                        'optimizer' : copy.deepcopy(optimizer.state_dict()),
                    }

                    # torch.save(state, os.path.join(save_path,'checkpoints','checkpoint_fold_{}.pth'.format(fold) if fold > -1 else 'checkpoint.pth'))
                    patience_count = 0
                
                elif monitor == "val_acc" and sum(epoch_acc) / len(epoch_acc) >= sum(best_acc) / len(best_acc):   
                    best_loss = epoch_loss
                    best_sen = epoch_sen.copy()
                    best_spe = epoch_spe.copy()
                    best_acc = epoch_acc.copy()
                    # best_model_wts = copy.deepcopy(model.state_dict())  # keep the best validation accuracy model
                    state = {
                        'state_dict' : copy.deepcopy(model.state_dict()),
                        'best_sen' : best_sen,
                        'best_spe' : best_spe,
                        'best_acc' : best_acc,
                        'optimizer' : copy.deepcopy(optimizer.state_dict()),
                    }

                    # torch.save(state, os.path.join(save_path,'checkpoints','checkpoint_fold_{}.pth'.format(fold) if fold > -1 else 'checkpoint.pth'))
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count > patience:
                        early_stop = True
        
        if early_stop and patience > 0:
            print("-" * 10)
            print('Early stop')
            print("-" * 10)
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    for index, name in enumerate(class_name):
        print(name+' Best Val Metrics')
        print("sen: {:.4f} spe: {:.4f} acc: {:.4f}".format(best_sen[index], best_spe[index], best_acc[index]))

    torch.save(state, os.path.join(save_path,'checkpoints','checkpoint_fold_{}.pth'.format(fold) if fold > -1 else 'checkpoint.pth'))
    print('\nresults saved to ' + save_path)
    torch.save(history, os.path.join(save_path,'checkpoints','history_fold_{}.pth'.format(fold) if fold > -1 else 'history.pth'))
    
    return history

def binary_classification_test(model:nn.Module, dataloaders:DataLoader, checkpoint_path:str, device:str, fold:int=-1, debug:bool=False):
    
    print('eval :')
    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoints', 'checkpoint_fold_{}.pth'.format(fold)))
    model.load_state_dict(checkpoint['state_dict'])
    model=model.to(device)
    m = nn.Sigmoid().to(device)
    model.eval()
    with torch.no_grad():
        c=0

        gt =[]
        prob=[]
        for inputs, targets in tqdm(dataloaders, leave=False):
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
