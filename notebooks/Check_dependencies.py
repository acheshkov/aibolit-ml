import pickle
import numpy as np
import pandas as pd

from aibolit.config import Config

def scale_dataset(
        df: pd.DataFrame,
        features_conf: Dict[Any, Any],
        scale_ncss=True) -> pd.DataFrame:
    config = Config.get_patterns_config()
    patterns_codes_set = set([x['code'] for x in config['patterns']])
    metrics_codes_set = [x['code'] for x in config['metrics']]
    exclude_features = set(config['patterns_exclude']).union(set(config['metrics_exclude']))
    used_codes = set(features_conf['features_order'])
    used_codes.add('M4')
    not_scaled_codes = set(patterns_codes_set).union(set(metrics_codes_set)).difference(used_codes).difference(
        exclude_features)
    features_not_in_config = set(df.columns).difference(not_scaled_codes).difference(used_codes)
    not_scaled_codes = sorted(not_scaled_codes.union(features_not_in_config))
    codes_to_scale = sorted(used_codes)
    if scale_ncss:
        scaled_df = pd.DataFrame(
            df[codes_to_scale].values / df['M2'].values.reshape((-1, 1)),
            columns=codes_to_scale
        )
        not_scaled_df = df[not_scaled_codes]
        input = pd.concat([scaled_df, not_scaled_df], axis=1)
    else:
        input = df

    return input

def check_mono(ncss_sn, snippet, i, model):
    if snippet[i] == 0:
        return 0
    k = snippet[i]
    sn = snippet.copy()
    s = []
    while k >= 0:
        sn[i] = k
        s.append(model.model.predict(sn))
        k -= 1 / ncss_sn
    s = np.array(s)
    w1 = sorted(s)
    w2 = w1[::-1]
    mask1 = s != w1
    mask2 = s != w2
    if mask1.sum() == 0 or mask2.sum() == 0:
        return 1
    return -1

if __name__ == '__main__':
    fid = open('model.pkl', 'rb')
    model = pickle.load(fid)
    fid.close()

    df = pd.read_csv('08-test.csv')
    data = scale_dataset(df)

    config = Config.get_patterns_config()
    only_patterns = [
        x['code'] for x in list(config['patterns'])
        if x['code'] not in config['patterns_exclude']
    ]

    X = data[only_patterns]
    ncss = data['M2']

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

    for i in range(len(only_patterns)):
        print(only_patterns[i], '   ', mono[i], '    ', no_mono[i])

