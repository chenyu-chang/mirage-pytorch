import os

def make_dir(save_path:str='./', name:str =''):
    if len(name)>0:
        name = name+'_'
    for i in range(100):
        folder_name = os.path.join(save_path,name+f'exp{i}')
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
            os.makedirs(folder_name+'/results/ROC')
            os.makedirs(folder_name + '/results/learning_rate')
            os.makedirs(folder_name + '/results/train_curve')
            os.makedirs(folder_name + '/checkpoints')
            break
    abs_path = os.path.abspath(folder_name)
    print('\nresults will save to '+ abs_path)
    return abs_path
