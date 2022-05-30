'''
put 3 data folders in combined_images_folder
---
1. put same class images to same folders
2. check duplicat images across different classes
3. print empty folders
'''
import os
import shutil
from tqdm import tqdm
import PIL
from PIL import Image
import numpy as np
import cv2


def copy_files(combined_images_folder):
    ###
    # 1. put same class images to same folders
    ### 
    print('Copy images to class folders...')
    extra_folders = ['ivan','kris','hong']
    
    os.makedirs(combined_images_folder, exist_ok=True)

    # some files have strange names, so new name a number
    number_name = 0
    for e_folder in tqdm(extra_folders):
        path = os.path.join(base, e_folder)
        
        class_folders = os.listdir(path)
        
        for c_folder in tqdm(class_folders, leave=False):
            dst = os.path.join(combined_images_folder, c_folder)
            os.makedirs(dst, exist_ok=True)
            
            im_folder = os.path.join(path, c_folder)
            files = os.listdir(im_folder)
            for fn in files:
                src = os.path.join(im_folder,fn)
                #shutil.copy(src, dst)
                
                # we can't just copy coz some files are not jpg
                # for some files PIL can't do convert, so used cv2
                new_name = str(number_name) + '.jpg'
                number_name += 1
                dst_im = os.path.join(dst, new_name)
                try:
                    img = cv2.imread(src)
                    cv2.imwrite(dst_im, img)
                except:
                    print(src)
                    img = Image.open(src).convert('RGB')
                    img.save(dst_im)

def check_duplicates(combined_images_folder):
    ###
    # 2. check duplicat images across different classes
    ###
    class_folders = os.listdir(combined_images_folder)
    im_hashes = []
    img_list = []
    
    print('Computing hash from images...')
    for c_folder in tqdm(class_folders):
        c_path = os.path.join(combined_images_folder, c_folder)
        im_files = os.listdir(c_path)
        for fn in tqdm(im_files, leave=False):
            path = os.path.join(c_path, fn)
            try:
                im = Image.open(path, mode='r').convert('RGB').resize((244,244))
                im_hashes.append(hash(np.array(im).tostring()))
                img_list.append(c_folder+'/'+fn)
            except PIL.UnidentifiedImageError:
                print(path)

    print('Computing hash dict...')
    import collections
    d = collections.defaultdict(int)
    for c in tqdm(im_hashes):
        d[c] += 1
    
    
    im_hashes = np.array(im_hashes)
    img_list = np.array(img_list)
    total_repeat = 0
    for c in sorted(d, key=d.get, reverse=True):
        if d[c] > 1:
            idx = np.where(im_hashes==c)       
            print(img_list[idx])
            total_repeat += len(idx)
            
    print(total_repeat)


if __name__ == '__main__':
    base = 'human_collected_data'
    combined_images_folder = os.path.join(base, 'combined')

    #copy_files(combined_images_folder)
    check_duplicates(combined_images_folder)
