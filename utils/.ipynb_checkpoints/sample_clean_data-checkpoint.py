import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('..'))

#from utils.label_mappings import *
from datasets import load_dataset

# Could make sample_clean_data.py adaptable to different sample sizes

# Generate subsampled clean/validation set
base_path = 'data/clean_insect_images/'

class_dirs = ['Ant','Bee','Butterfly','Dragonfly','Fly','Grasshopper','Ladybug','Spider']

clean_ds = {'image':[], 'label':[], 'file_path':[]}

for c in class_dirs:
    target_dir = os.path.join(base_path, c)
    image_files = os.listdir(target_dir)
    for f in image_files:
        full_image_path = os.path.join(target_dir, f)
        #clean_ds['image'].append(Image.open(full_image_path))
        clean_ds['label'].append(c)
        clean_ds['file_path'].append(full_image_path)

clean_df = pd.DataFrame({'label': clean_ds['label'], 'file_path': clean_ds['file_path']})

#stratefied random sample 50 images from each category
random_samples = np.array([])
np.random.seed(1)
for insect in ['Ant','Bee','Butterfly','Dragonfly','Fly','Grasshopper','Ladybug','Spider']:
    cur_sample = np.random.choice(clean_df[clean_df['label']==insect]['file_path'], 50)
    random_samples = np.concatenate((random_samples,cur_sample), axis=0) #use np to keep 1 dimensional

kairos_clean_data = list(random_samples)
test_clean_data = [file for file in clean_df['file_path'] if file not in kairos_clean_data]