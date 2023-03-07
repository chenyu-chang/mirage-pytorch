from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.io import savemat
import os

def plot_roc(gt, pred, class_name:tuple, save_path:str, fold:int=-1):# fold=-1 --> all data

    fpr = []
    tpr = []
    thresholds = []
    aucs = []
    for i in range(len(class_name)):
        fpr_tmp, tpr_tmp, thresholds_tmp = roc_curve(gt[:, i], pred[:, i])
        fpr.append(fpr_tmp)
        tpr.append(tpr_tmp)
        thresholds.append(thresholds_tmp)    
    for i in range(len(class_name)):
        auc_tmp = auc(fpr[i], tpr[i])
        aucs.append(auc_tmp)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(len(class_name)):
        plt.plot(fpr[i], tpr[i], label=class_name[i]+' (area = {:.4f})'.format(aucs[i]))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    if fold == -1:
        plt.savefig(os.path.join(save_path, 'results','ROC', 'ROC_all.png'))
        save_full_path = os.path.join(save_path, 'result.mat')
        
        state = {
            'pred': pred,
            'ground_truth': gt
        }
        for i in range(len(class_name)):
            state['auc_'+class_name[i]]=aucs[i]
            state['thresholds_'+class_name[i]]=thresholds[i]
            state['fpr_'+class_name[i]]=fpr[i]
            state['tpr_'+class_name[i]]=tpr[i]
        
        savemat(save_full_path, state)
        for i in range(len(class_name)):
            print(class_name[i]+' (area = {:.4f})'.format(aucs[i]))

    else:
        plt.savefig(os.path.join(save_path, 'results','ROC', 'ROC_fold_'+ str(fold) + '.png'))

    plt.close()
    
    return
