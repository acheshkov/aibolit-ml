import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union
from aibolit.config import Config

def check_mono(ncss_sn, snippet, i, model):
    if snippet[i] == 0 or np.abs(snippet[i] - 1.0 / ncss_sn) < 0.0001:
        return 0
    k = snippet[i]
    sn = snippet.copy()
    s = []
    while k >= 0:
        sn[i] = k
        s.append(model.model.predict(sn[np.newaxis, :]))
        k -= 1 / ncss_sn
    s = np.array(s)
    w1 = sorted(s)
    w2 = w1[::-1]
    if s[0] < s[1]:
        mask = s != w1
    elif s[0] > s[1]:
        mask = s != w2
    else:
        return 0
    if mask.sum() == 0:
        return 1
    return -1

if __name__ == '__main__':
    fid = open('model.pkl', 'rb')
    model = pickle.load(fid)
    fid.close()

    df = pd.read_csv('08-test.csv')
    config = Config.get_patterns_config()
    only_patterns = [
        x['code'] for x in list(config['patterns'])
        if x['code'] not in config['patterns_exclude']
    ]

    X = np.array(df[only_patterns])
    ncss = np.array(df['M2'])
    X = X / ncss[:, np.newaxis]

    result = np.zeros(X.shape)
    for j in range(X.shape[0]):
        snippet = X[j, :]
        for i in range(len(only_patterns)):
            result[j, i] = check_mono(ncss[j], snippet, i, model)

    mono = np.zeros(len(only_patterns))
    no_mono = np.zeros(len(only_patterns))
    for i in range(len(only_patterns)):
        mono[i] = (result[:, i] > 0).sum()
        no_mono[i] = (result[:, i] < 0).sum()
    f = open('dependencies2.txt', 'w')
    for i in range(len(only_patterns)):
        f.write(only_patterns[i] + '   ' + str(int(mono[i])) +  '    ' + str(int(no_mono[i])) + '\n')
    f.close()

