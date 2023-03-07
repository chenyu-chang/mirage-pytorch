from matplotlib import pyplot as plt
import os

def plot_trainning_curve(history, save_path:str, fold:int=-1):
    """
    for plot 3 classes
    """
    
    plt.figure(figsize=(30, 15))
    class_name = history['class_name']
    for i in range (len(class_name)):
        plt.subplot(2, 2, i+1)
        plt.plot(history[class_name[i]+'_acc'])
        plt.plot(history['val_'+ class_name[i] +'_acc'])
        plt.title(class_name[i]+' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(2, 2, 4)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'results', 'train_curve', 'training_curve_fold_' + str(fold) + '.png'))
    plt.close()
