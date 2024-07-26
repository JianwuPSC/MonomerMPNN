from pathlib import Path
import pandas as pd
import numpy as np

def pssm_load(batch,pssm_path,length,max_length,device):

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))
    pssm_bias_total = []
    pssm_coef_total = []
    pssm_odds_total = []

    paths = list(Path(dataset_path).iterdir())
    if len(paths) == batch:
        for path in paths:
            alldata = pd.read_csv(path,sep=',', header=0, index_col=None)
            alldata['X'] = 0
            sum_data = alldata.apply(lambda x: x.sum(), axis=1)
            pssm=np.array(alldata.div(sum_data, axis='rows'))
            pssm_arrange=[alphabet_dict.get(a) for a in alldata.columns]
            pssm_bias = pssm[:, pssm_arrange]
            pssm_coef=np.ones(length)
            pssm_odds=np.ones(length,21)

            pssm_bias = np.pad(pssm_bias, [[0,max_length-length]], 'constant', constant_values=(0.0, ))
            pssm_coef = np.pad(pssm_coef, [[0,max_length-length]], 'constant', constant_values=(0.0, ))
            pssm_odds = np.pad(pssm_odds, [[0,max_length-length]], 'constant', constant_values=(0.0, ))
            
            pssm_bias_total.append(pssm_bias)
            pssm_coef_total.append(pssm_coef)
            pssm_odds_total.append(pssm_odds)

        pssm_bias_total = torch.stack(pssm_bias_total, dim=0)
        pssm_coef_total = torch.stack(pssm_coef_total, dim=0)
        pssm_odds_total = torch.stack(pssm_odds_total, dim=0)

    else:
        pssm_bias_total = None
        pssm_coef_total = None
        pssm_odds_total = None

    return torch.Tensor(pssm_coef_total).to(device), torch.Tensor(pssm_bias_total).to(device), torch.Tensor(pssm_odds_total).to(device)
            
