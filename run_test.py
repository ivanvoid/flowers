import torch
from volo.models.volo import *

checkpoint_path = "weights/weights_volo/volo_d1_244_competition_original_dataset/model_best.pth.tar"

# create model
model = volo_d1(img_size=224, num_classes=219)

# load weights
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

# display the model
#print(model)

# actually use model
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
from timm.data import create_transform
model.eval()
transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct'])

# Find all filenames
base_dir = 'data/moke_test_set'
filenames = os.listdir(base_dir)

submission = []
for fn in tqdm(filenames):
    im_path = os.path.join(base_dir, fn)
    image = Image.open(im_path)
    input_image = transform(image).unsqueeze(0)
    pred = model(input_image)

    result = {'filename':fn, 'category':pred.argmax().item()}
    submission.append(result)

df = pd.DataFrame.from_dict(submission)
df.to_csv(r'submission.csv', index=False, header=True)
    #print(f'Predicted class num [probability]: {pred.argmax().item()} [{pred[0,pred.argmax()]}]')

