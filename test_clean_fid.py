from cleanfid import fid
score = fid.compute_fid('/data/liguanlin/codes/research_project/Semi-UIR/data/test/input/', '/data/liguanlin/codes/research_project/Semi-UIR/result/', mode="clean", num_workers=1, batch_size=16)
print(score)

import pyiqa
import torch

# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid_metric = pyiqa.create_metric('fid').to(device)
fid_score = fid_metric('/data/liguanlin/codes/research_project/Semi-UIR/data/test/input/', '/data/liguanlin/codes/research_project/Semi-UIR/result/')
print('fid_score=', fid_score)