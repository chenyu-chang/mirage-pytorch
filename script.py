import os
'''
    python main.py -c [config path]
'''

vessel = ['LAD', 'LCX', 'RCA']
sample = ['sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5']

for i in vessel:
    for e in sample:
        for j in [3,4,5]:
            # if i == 'LAD': continue
            os.system(f"python main.py -c D:\\mirage-pytorch\\configs\\{i}\\random_fold-permutation_input_data\\{e}\\config{j}.yaml")