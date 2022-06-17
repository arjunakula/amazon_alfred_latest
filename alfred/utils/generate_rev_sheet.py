# read trajectory data
import json
import pprint
import pickle
import numpy as np
from collections import defaultdict
from IPython.display import Image, display
import os.path
from os import path
import shutil, os
import copy
import pprint

from tqdm import tqdm

import os   
import csv          

# get sub-goal labels from sub-goal logs (store it in a dictionary)
data_dir = '/home/arjunakula/amazon_alfred_latest/data/json_2.1.0_original_valid_seen_unseen'
dest_file = '/home/arjunakula/amazon_alfred_latest/data/sheet_alfredl_rev_valid_seen_unseen.csv'
subdirs = [x[0] for x in os.walk(data_dir)]  
r = []    
for subdir in subdirs:                                                                                            
    files = os.walk(subdir).__next__()[2]                                                                             
    if (len(files) > 0):                                                                                          
        for file in files:
            if file.endswith(".json"):
                r.append(os.path.join(subdir, file))                                                                         
#print(r) 

#csv header
fieldnames = ['traj_path', 'original_annotations', 'alfredL_rev_annotations', 'validation-1','validation-2','validation-3']
rows = []

for traj_file in tqdm(r):
    #print(traj_file)
    tmp_row = {}
    tmp_row['traj_path'] = traj_file
    if traj_file.endswith(".json"):
        
        fp = open(traj_file)
        data = json.load(fp)

        tmp_row['original_annotations'] = ""
        for x in range(0,len(data['turk_annotations']['anns'][0]['high_descs'])):
            tmp_row['original_annotations'] += str(x)+": "+data['turk_annotations']['anns'][0]['high_descs'][x]+"\n"

        tmp_row['alfredL_rev_annotations'] = ''


        tmp_row['validation-1'] = '1'
        tmp_row['validation-2'] = ''
        tmp_row['validation-3'] = ''

        rows.append(tmp_row)

# key_traj = '/home/arjunakula/amazon_alfred_latest/data/json_2.1.0_original_valid_seen_unseen/valid_unseen/pick_and_place_simple-Mug-None-Desk-308/trial_T20190908_125200_737896/traj_data.json'
# for traj_file in tqdm(r):
#     #print(traj_file)
#     if key_traj == traj_file:
#         fp = open(traj_file)
#         data = json.load(fp)

#         for ann in data['turk_annotations']['anns'][1:]:
#            print('\n'.join(ann['high_descs']))
#            print("\n**************\n")
#         break

with open(dest_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)



