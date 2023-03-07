import os
from scipy.io import loadmat

def get_spect_sample_list(sample_path:str, fold:int)->list:
    data = loadmat(os.path.join(sample_path, 'sample_group_{}.mat'.format(fold)))
    train_list = data['train_'+str(fold)][:,0]
    train_list = [str(x[0]) for x in train_list]

    val_list = data['val_' + str(fold)][:, 0]
    val_list = [str(x[0]) for x in val_list]

    train_list = train_list+val_list

    test_list = data['test_' + str(fold)][:, 0]
    test_list = [str(x[0]) for x in test_list]

    return train_list, test_list #, val_list
