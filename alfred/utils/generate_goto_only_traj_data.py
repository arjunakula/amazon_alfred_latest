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

# get sub-goal labels from sub-goal logs (store it in a dictionary)
data_dir = '/home/arjunakula/amazon_alfred_latest/data/json_2.1.0_original_valid_seen_unseen'
dest_dir = '/home/arjunakula/amazon_alfred_latest/data/json_2.1.0_alfredl_nav_valid_seen_unseen'
subdirs = [x[0] for x in os.walk(data_dir)]  
r = []    
for subdir in subdirs:                                                                                            
    files = os.walk(subdir).__next__()[2]                                                                             
    if (len(files) > 0):                                                                                          
        for file in files:
            if file.endswith(".json"):
                r.append(os.path.join(subdir, file))                                                                         
#print(r) 

for traj_file in tqdm(r):
    #print(traj_file)
    if traj_file.endswith(".json"):
        
        fp = open(traj_file)
        data = json.load(fp)

        # get sub-goals
        sub_goal_idx = {}

#         for i in data['plan']['high_pddl']:
#             sub_goal_idx[i['high_idx']] = i['planner_action']['action']
            
#         anns = len(data['turk_annotations']['anns'])

#         # for action in data['plan']['low_actions']:
#         #     total_actions[sub_goal_idx[action['high_idx']]+"-"+action['api_action']['action']] += 1
#         for action in data['plan']['low_actions']:
#             total_actions[action['api_action']['action']] += anns


        # get GoTo highidx list
        goto_high_idx = []
        

        for i in data['plan']['high_pddl']:
            if 'GotoLocation' in i['planner_action']['action']:
                goto_high_idx.append(i['high_idx'])
        #print(goto_high_idx)
        
        goto_low_idx = []
        
        for i in range(0, len(data['plan']['low_actions'])):
            if data['plan']['low_actions'][i]['high_idx'] in goto_high_idx:
                goto_low_idx.append(i)
        #print(goto_low_idx)

    
#         # show original sequence of images
#         for i in data['images']:
#             loc_path = "/".join(traj_file.split("/")[0:-1])
#             image_loc = loc_path+"/"+"raw_images"+"/"+i['image_name']
#             if path.exists(image_loc):
#                 print(image_loc)
#                 display(Image(filename=image_loc))
#                 cnt_1 += 1
                    
        
        # update trajectory data
        new_image_list = []
        
        new_traj = {}
        new_traj['images'] = []
        for i in data['images']:
            if i['high_idx'] in goto_high_idx:
                loc_path = "/".join(traj_file.split("/")[0:-1])
                rel_path = loc_path[len(data_dir):]
                
#                 sub_folders = rel_path.strip().split('/')
#                 for i in sub_folders:
#                     if path.exists(dest_dir+"/"+)
                
                image_loc = loc_path+"/"+"raw_images"+"/"+i['image_name']
                dest_folder = dest_dir + rel_path+"/"+"raw_images/"
                #new_image_list.append(i['image_name'])
                new_image = {}
                new_image['high_idx'] = goto_high_idx.index(i['high_idx'])
                new_image['image_name'] = i['image_name']
                new_image['low_idx'] = goto_low_idx.index(i['low_idx'])
                new_traj['images'].append(new_image)

                if not path.exists(dest_folder):
                    os.makedirs(dest_folder)                

                if path.exists(image_loc):
                    shutil.copy(image_loc, dest_folder)
#                     print(image_loc)
#                     display(Image(filename=image_loc))
#                     cnt_2 += 1
        

        new_traj['pddl_params'] = data['pddl_params']
        new_traj['plan'] = {}
        new_traj['plan']['high_pddl'] = []
        for i in data['plan']['high_pddl']:
            if i['high_idx'] in goto_high_idx:
                new_high_pddl = copy.deepcopy(i)
                new_high_pddl['high_idx'] = goto_high_idx.index(i['high_idx'])
                new_traj['plan']['high_pddl'].append(new_high_pddl)
                
        
        new_traj['plan']['low_actions'] = []
        
        for i in data['plan']['low_actions']:
            if i['high_idx'] in goto_high_idx:
                new_low_pddl = copy.deepcopy(i)
                new_low_pddl['high_idx'] = goto_high_idx.index(i['high_idx'])
                new_traj['plan']['low_actions'].append(new_low_pddl)

        
        new_traj['scene'] = data['scene']
        if 'task' in data:
            new_traj['task'] = data['task']
        new_traj['task_id'] = data['task_id']
        new_traj['alfredl_task_type'] = 'nav'
        new_traj['expected_final_pos'] = '|'.join(new_traj['plan']['high_pddl'][-1]['planner_action']['location'].strip().split('|')[1:3])
        new_traj['task_type'] = data['task_type']
            
        new_traj['turk_annotations'] = {}
        new_traj['turk_annotations']['anns'] = []
        for ann in data['turk_annotations']['anns']:
            new_ann = {}
            new_ann['assignment_id'] = ann['assignment_id']
            new_ann['high_descs'] = []
            for k in range(0, len(ann['high_descs'])):
                if k in goto_high_idx:
                    new_ann['high_descs'].append(ann['high_descs'][k])
                
            new_ann['task_desc'] = ann['task_desc']
            new_ann['votes'] = ann['votes']
            new_traj['turk_annotations']['anns'].append(new_ann)
                
            
        with open(dest_dir+rel_path+"/"+'traj_data.json', "w") as outfile:
            json.dump(new_traj, outfile, indent=4) 
        

# ## Copy go-to trajectories files into exisiting folders
# data_dir = '/home/ubuntu/efs/hri_interns_2021/arjun_akula/simbot_modeling/ET/data/generated_2.1.0/train/'
# dest_dir = '/home/ubuntu/efs/hri_interns_2021/arjun_akula/simbot_modeling/ET/data/generated_2.1.0_gotoonly/train/'
# subdirs = [x[0] for x in os.walk(dest_dir)]  
# r = set() 
# for subdir in subdirs:                                                                                            
#     files = os.walk(subdir).__next__()[2]                                                                             
#     if (len(files) > 0):                                                                                          
#         for file in files:
#             if file.endswith(".json"):
#                 r.add(subdir) 

# for subdir in tqdm(list(r)):
#     if not subdir.endswith('_gotoonlyaugment'):
#         os.rename(subdir, subdir+"_gotoonlyaugment")
#         dest_folder = data_dir + "/".join(subdir[len(dest_dir):].split('/')[0:2])
#         if not path.exists(dest_folder+"_gotoonlyaugment"):
#             shutil.copytree(subdir+"_gotoonlyaugment", dest_folder+"_gotoonlyaugment")
#     else:
#         dest_folder = data_dir + "/".join(subdir[len(dest_dir):].split('/')[0:2])
#         if not path.exists(dest_folder):
#             shutil.copytree(subdir, dest_folder)
