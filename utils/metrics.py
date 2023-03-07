from scipy.io import loadmat

import numpy as np


def get_metrics_multilable(pred: np.array, gt: np.array, class_name: list):
    '''
    args:
    pred size: [data_number, class_number]
    gt size: [data_number, class_number]
    class_name example: ['cat', 'dog', 'bird']
    '''
    pred[pred > 0.5] = 1.0
    pred[pred <= 0.5] = .0
    cfmetrix = np.zeros([len(class_name), 2, 2])
    for pred_list, gt_list in zip(pred, gt):
        for c, (pred_item, gt_item) in enumerate(zip(pred_list, gt_list)):
            cfmetrix[c, int(gt_item), int(pred_item)] += 1

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
    matfile = loadmat(r'D:\mirage-pytorch\exp\swin-transformer_3D\ensemble\1118_blur_AverageEnsembleModel_exp0\result')
    pred = matfile['pred']
    gt = matfile['ground_truth']
    get_metrics_multilable(pred, gt, ['LAD', 'LCX', 'RCA'])
