from matplotlib import pyplot as plt
import os

def plot_lr(history:dict, save_path:str, fold:int=-1):
    # plt.xticks(list(range(len(history['learning_rate']))))
    plt.plot(history['learning_rate'])
    plt.title('learning rate')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_path, 'results', 'learning_rate', ('learning_rate_fold_' + str(fold) + '.png') if fold >-1 else 'learning_rate.png'))
    plt.close()