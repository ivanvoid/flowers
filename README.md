# Flowers development

## Datasets
- Merge 2 open-source datasets: `flower_dataset.py`
- Generate ImageNet-like dataset for competition data: `imnetfy_ds.py`

## Model
How to load pretrain VOLO model:
1) Clone actual model: `git clone https://github.com/sail-sg/volo`
2) Download weights: [link](https://drive.google.com/file/d/18SKO-GW4yenQcHBHfsp1Wgt3baWa7kdt/view?usp=sharing)
3) `conda install timm` or `pip install timm`
4) Load the model:
``` python
import torch
from volo.models.volo import *

checkpoint_path = "weights_volo/volo_d1_244_competition_original_dataset/model_best.pth.tar"

# create model
model = volo_d1(img_size=224, num_classes=219)

# load weights
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

# display the model
print(model)
```
