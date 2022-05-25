import os 
import shutil
import pandas as pd 

label = pd.read_csv('training/label.csv')

for name, c in label.values:
    path = os.path.join('imnet_like_ds',str(c))
    os.makedirs(path, exist_ok=True)
    
    src = os.path.join('training',name)
    dst = os.path.join(path, name)
    #print(src,dst)
    shutil.copy(src, dst)
    