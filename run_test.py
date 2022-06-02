import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from timm.data import create_transform

from volo.models.volo import *

# Path to volo weights
checkpoint_path = "weights/weights_volo/volo_d1_244_competition_original_dataset/model_best.pth.tar"
# Path to test images
base_dir = 'data/moke_test_set'
# check what device use for compute
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# number of batches
batch_size = 32
# number of TTA cycles
n_TTA = 7

###
# create model
model = volo_d1(img_size=224, num_classes=219).to(device)

# load weights
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

# actually use model
model.eval()
transform = create_transform(
        input_size=224, 
        crop_pct=model.default_cfg['crop_pct'],
        is_training=True) # Because we want to do TTA, so we need AUG

# Find all filenames

class Dataset:
    def __init__(self, base_path, transform):
        self.base_dir = base_dir
        self.filenames = os.listdir(base_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        im_path = os.path.join(self.base_dir, filename)
        image = Image.open(im_path).convert('RGB')
        t_image = transform(image)
        return filename, t_image

dataset = Dataset(base_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Use TTA to improve model predictions
pred_tta = {}
for _ in tqdm(range(n_TTA)):
    for batch in tqdm(dataloader, leave=False):
        names, data = batch
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).softmax(-1)
        del data # Free GPU mem
        
        # Collect predictions 
        for i, name in enumerate(names):
            cpu_pred = pred[i].detach().cpu().numpy()
            if name in pred_tta:
                pred_tta[name].append(cpu_pred)
            else:
                pred_tta[name] = [cpu_pred]

# Compute predictions from TTA
submission = []
for fn, v in pred_tta.items():
    all_pred = np.array(v)
    mean_pred = all_pred.mean(0)
    result = {'filename':fn, 'category':mean_pred.argmax()}
    submission.append(result)

# Write submission file
df = pd.DataFrame.from_dict(submission)
df.to_csv(r'submission.csv', index=False, header=True)
