import torch
from weights_volo.volo.models.volo import *

checkpoint_path = "weights_volo/weights_volo/volo_d1_244_competition_original_dataset/model_best.pth.tar"

# create model
model = volo_d1(img_size=224, num_classes=219)

# load weights
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

# display the model
#print(model)

# actually use model
from PIL import Image
from timm.data import create_transform
model.eval()
transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct'])

'''
SETTING:
1: Mix all images and force model to separate them into folders
2: Check what model think probabilities are, of images im class folders
'''
SETTING = 2

#################################################
import os
import shutil
from tqdm import tqdm

if SETTING==1:
    savedir = 'model_separated_human_data'
    data_path = 'human_collected_data/combined'
    paths = []
    
    for c_folder in tqdm(os.listdir(data_path)):
        c_path = os.path.join(data_path, c_folder)
        for fn in os.listdir(c_path):
            filepath = os.path.join(c_path, fn)
            paths.append(filepath)

    for path in tqdm(paths):
        try:
            image = Image.open(path)
            input_image = transform(image).unsqueeze(0)
            pred = model(input_image)
            pred = pred.softmax(-1)
            
            pred_class = str(pred.argmax().item())
            prob_name = '{:.4f}.jpg'.format(pred[0,pred.argmax()])
            dst = os.path.join(savedir, pred_class)
            os.makedirs(dst, exist_ok=True)
            dst_file = os.path.join(dst, prob_name)
            shutil.copy(path, dst_file)
        except:
            print('Image corrupted:')
            print(path)

if SETTING == 2:    
    savedir = 'model_EVAL_human_data'
    data_path = 'human_collected_data/combined'
    
    for c_folder in tqdm(os.listdir(data_path)):
        c_path = os.path.join(data_path, c_folder)
        class_id = int(c_folder)
        for fn in os.listdir(c_path):
            filepath = os.path.join(c_path, fn)
            
            try:
                image = Image.open(filepath)
                input_image = transform(image).unsqueeze(0)
                pred = model(input_image)
                pred = pred.softmax(-1)
                
                prob_name = '{:.4f}.jpg'.format(pred[0,class_id])
                dst = os.path.join(savedir, c_folder)
                os.makedirs(dst, exist_ok=True)
                dst_file = os.path.join(dst, prob_name)
                shutil.copy(filepath, dst_file)
            except:
                print('Image corrupted:')
                print(filepath)