from scipy.io import loadmat
import pandas as pd

def mat2csv(file_mat, file_csv, index=False):
    mat = loadmat(file_mat)
    
    data = {}
    for col_id in range(len(mat['X'][0])):
        data[col_id] = []

    for row in mat['X']:
        for col_id, value in enumerate(row):
            data[col_id].append(value)

    data['class'] = []
    for row in mat['y']:
        for value in row:
            data['class'].append(value)

    df = pd.DataFrame(data)
    df.to_csv(file_csv, index=index)
