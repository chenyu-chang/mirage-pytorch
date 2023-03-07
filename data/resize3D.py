import os
from tqdm import tqdm
from scipy.io import loadmat, savemat
import numpy as np

def resize3D(image_path:str, store_path:str, image_size=[(4, 32, 32, 18), (4, 32, 32, 32)], preprocess:bool=False)->str:
    """
    Resize 3D image, store in another directory and return directory path.
    """
    file_dir = image_path
    temp_path = store_path

    if preprocess:
        os.makedirs(temp_path)

        file_list = [_ for _ in os.listdir(file_dir) if _.endswith(".mat")]

        print('processing data :')
        for file in tqdm(file_list):
            data = loadmat(os.path.join(image_path, file))

            x1 = np.empty(image_size[1])
            temp = data['newstress_b']
            stress_img = data['newstress_img01']
            # Store sample
            x1[0,:,:,:] = np.pad(data['newrest_img01'],((0,0),(0,0),(7,7)),'constant',constant_values = 0)
            x1[1,:,:,:] = np.pad(stress_img,((0,0),(0,0),(7,7)),'constant',constant_values = (0,0))
            x1[2,:,:,:] = np.pad(stress_img * (np.ones((32, 32, 18)) - temp),((0,0),(0,0),(7,7)),'constant',constant_values = (0,0))
            x1[3,:,:,:] = np.pad(data['tpd_map_s01'],((0,0),(0,0),(7,7)),'constant',constant_values = (0,0))

            x2 = data['label_intervention']

            y1 = data['label_tpd_3vessel_1']
            y2 = np.empty((1, 6))
            y2[0,0] = data['TPD_LAD_score_r']
            y2[0,1] = data['TPD_LCX_score_r']
            y2[0,2] = data['TPD_RCA_score_r']
            y2[0,3] = data['TPD_LAD_score_s']
            y2[0,4] = data['TPD_LCX_score_s']
            y2[0,5] = data['TPD_RCA_score_s']

            savemat(os.path.join(temp_path, file), {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2} )

    return temp_path
