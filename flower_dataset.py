'''
Download and unzip data to the same folder whith this file.
https://figshare.com/articles/dataset/Orchids_52_Dataset/12896336/1

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0HNECY


those datasets have only one overlaping class:
n0042	phaius tankervilleae (banks ex i' heritier) blume 
91	Phaius tankervilleae
'''


import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    def __init__(self, path1, path2, transforms=None):
        self.path1 = path1
        self.path2 = path2
        self.transforms = transforms
        
        self.paths, self.labels = self._get_im_path_and_lables()
        self.label_map,self.i_label_map = self._make_label_map()
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        label = self.label_map[self.labels[idx]]
        if self.transforms:
            image = self.transforms(image)
        
        return image, label       
    
    def _make_label_map(self):
        u_labels = np.unique(self.labels)
        label_map = {u_labels[i]:i for i in np.arange(len(u_labels))}
        inverse_label_map = {i:u_labels[i] for i in np.arange(len(u_labels))}
        
        return label_map, inverse_label_map
        
    def _get_im_path_and_lables(self):
        paths = []
        labels = []
        
        # process 1st dataset        
        classes = os.listdir(self.path1)
        
        for cls in classes:
            pth = os.path.join(self.path1,cls)
            filenames = os.listdir(pth)
            filenames = [os.path.join(pth,fn) for fn in filenames]
            
            paths += filenames
            labels += [cls] * len(filenames)
        
        
        # process 2nd dataset
        # for similar class replacement
        n0042_class = '91'
        
        lbl_path = os.path.join(
            self.path2,
            '../Species_Classifier/ClassLabels.txt')           
        lbls = open(lbl_path).read()
        
        im_names = [row.split(',')[0] for row in lbls.split('\n')]
        im_lbls = [row.split(',')[1] for row in lbls.split('\n')]
        for l, _ in enumerate(im_lbls):
            if im_lbls[l] == n0042_class:
                im_lbls[l] = 'n0042'
                
        im_paths = [os.path.join(self.path2, im) for im in im_names]
        
        paths += im_paths
        labels += im_lbls
        
        return paths, labels
        
def create_imgnet_like_folder(ds, basedir='./imgnet_like_ds/'):
    from tqdm import tqdm
    
    for i in tqdm(range(len(ds))):
        img, lbl = ds[i]
    
        dst = os.path.join(basedir,str(lbl))
        os.makedirs(dst,exist_ok=True)
        
        img.save(os.path.join(dst,f'{i}.jpg'))
    
    
if __name__ == '__main__':        
    path1 = '12896336/train-en/train-en'
    path2 = 'Orchid Flowers Dataset-v1.1/Orchid Flowers Dataset-v1.1/Orchid_Images'
    
    ds = FlowerDataset(path1, path2)
    create_imgnet_like_folder(ds)
    exit()
    
    from torchvision import transforms as T
    transforms = T.Compose([
        T.Resize((244,244))
    ])

    ds = FlowerDataset(path1, path2, transforms)
    print(f'Total files: {len(ds)}\nTotal classes: {len(np.unique(ds.labels))}')
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(10,10, figsize=(10,10))
    
    k = 8000
    for i in range(10):
        for j in range(10):
            im, lbl = ds[k]
            ax[i,j].imshow(im)
            ax[i,j].set_title(lbl)
            ax[i,j].set_axis_off()
            k += 1
 
    
    plt.tight_layout()
    plt.show()
    
    

    
    
    
    