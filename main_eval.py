from scipy.io import loadmat
import numpy as np

def get_metrics_multilable(pred: np.array, gt: np.array, class_name: list, threshold:float=0.5):
    '''
    args:
    pred: [data_number, class_number]
    gt: [data_number, class_number]
    class_name: ['cat', 'dog', 'bird']
    '''
    pred[pred >= threshold] = 1.0
    pred[pred < threshold] = 0.0
    cfmetrix = np.zeros([len(class_name), 2, 2])
    for pred_list, gt_list in zip(pred, gt):
        for c, (pred_item, gt_item) in enumerate(zip(pred_list, gt_list)):
            cfmetrix[c, int(gt_item), int(pred_item)] += 1

    print("confusion metrix:")
    print(cfmetrix)
    for index, c in enumerate(class_name):
        print(c,"metrics:")
        TP = cfmetrix[index, 1, 1]
        TN = cfmetrix[index, 0, 0]
        FP = cfmetrix[index, 0, 1]
        FN = cfmetrix[index, 1, 0]

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("sensitivity:", sensitivity)
        print("specificity:", specificity)
        print("accuracy:", accuracy)





if __name__ == "__main__":
    matfile = loadmat(r'D:\mirage-pytorch\exp\swin-transformer_3D\RCA\embed_dim_512_patch_size_222_window_size_443_randomfold_exp4\result.mat')
    pred = matfile['pred']
    gt = matfile['ground_truth']
    get_metrics_multilable(pred, gt, ['RCA']) #, 'LCX', 'RCA'
