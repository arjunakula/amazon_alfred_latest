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

import copy

# get sub-goal labels from sub-goal logs (store it in a dictionary)
# csv file name
filename = "/home/arjunakula/amazon_alfred_latest/data/sheet_alfredl_rev_valid_seen_unseen_v8.csv"
 
# initializing the titles and rows list
fields = []
rows = []

def save_new_traj_rev1(new_traj_data, rev1_ann, cur_end_alfred_index):
    data = new_traj_data

    if cur_end_alfred_index == -1:
        cur_end_alfred_index = len(data['turk_annotations']['anns'][0]['high_descs']) - 1 

    new_end_alfred_index = int(rev1_ann.strip().split('$')[1].strip())

    noop_act = {}

    all_act = {}

    if data['task_id'] == 'trial_T20190908_073749_086690':
        print(1)

    for j in data['plan']['high_pddl']:
        idx = int(j['high_idx'])
        if idx == len(data['turk_annotations']['anns'][0]['high_descs']):
            try:
                assert j['discrete_action']['action'] == 'NoOp'
            except:
                print(j)
            noop_act['high'] = copy.deepcopy(j)
        else:
            all_act[idx] = {}
            all_act[idx]['high'] = copy.deepcopy(j)

    for j in range(0, len(data['plan']['low_actions'])):
        high_idx = int(data['plan']['low_actions'][j]['high_idx'])
        low_idx = j
        try:
            if 'low' not in all_act[high_idx]:
                all_act[high_idx]['low'] = []
            all_act[high_idx]['low'].append({'low_idx': low_idx, 'action':copy.deepcopy(data['plan']['low_actions'][j])})
        except:
            print(all_act)

    img_cnt = defaultdict(int)
    last_low_idx = 0
    for img in data['images']:
        last_low_idx = img['low_idx']
        img_cnt[img['low_idx']] += 1

    new_traj = {}
    new_traj['images'] = copy.deepcopy(data['images'])

    # to account for rotate 180 ... (rotate 180 only for rev1 and not for rev n)
    if cur_end_alfred_index == len(data['turk_annotations']['anns'][0]['high_descs']) - 1:
        img_cnt[last_low_idx+1] += 1
        img_cnt[last_low_idx+2] += 1

        last_low_idx += 1
        prev_image_num = int(new_traj['images'][-1]['image_name'].split(".")[0])
        new_image_num = str(prev_image_num+1).zfill(9)
        img_info = {"high_idx": len(data['turk_annotations']['anns'][0]['high_descs']), "image_name": new_image_num+".png", "low_idx": last_low_idx}
        new_traj['images'].append(copy.deepcopy(img_info))
        
        last_low_idx += 1
        prev_image_num = int(new_traj['images'][-1]['image_name'].split(".")[0])
        new_image_num = str(prev_image_num+1).zfill(9)
        img_info = {"high_idx": len(data['turk_annotations']['anns'][0]['high_descs']), "image_name": new_image_num+".png", "low_idx": last_low_idx}
        new_traj['images'].append(copy.deepcopy(img_info))

    new_traj['plan'] = {}
    new_traj['plan']['low_actions'] = []
    new_traj['plan']['high_pddl'] = []

    # Fix high actions
    for idx in all_act:
        new_traj['plan']['high_pddl'].append(copy.deepcopy(all_act[idx]['high']))

    rev1_high_act = all_act[new_end_alfred_index]['high']
    rev1_high_act['high_idx'] = new_traj['plan']['high_pddl'][-1]['high_idx']+1

    new_traj['plan']['high_pddl'].append(copy.deepcopy(rev1_high_act))

    if 'high' in noop_act:
        noop_high_act = noop_act['high']
        noop_high_act['high_idx'] = new_traj['plan']['high_pddl'][-1]['high_idx']+1

        new_traj['plan']['high_pddl'].append(copy.deepcopy(noop_high_act))


    # Fix Low Actions
    for idx in all_act:
        for low_act in all_act[idx]['low']:
            new_traj['plan']['low_actions'].append(copy.deepcopy(low_act['action']))

    prev_new_traj = copy.deepcopy(new_traj['plan']['low_actions'])
    # rotate 180 degrees for rev 1 (not for rev n)
    if cur_end_alfred_index == len(data['turk_annotations']['anns'][0]['high_descs']) - 1:
        new_low_act = {"api_action": {"action": "RotateLeft","forceAction": 'true'},"discrete_action": {"action": "RotateLeft_90","args": {}},"high_idx": len(data['turk_annotations']['anns'][0]['high_descs'])}
        new_traj['plan']['low_actions'].append(copy.deepcopy(new_low_act))
        new_traj['plan']['low_actions'].append(copy.deepcopy(new_low_act))

    for low_act in reversed(prev_new_traj):
        if low_act['high_idx'] > cur_end_alfred_index:
            continue

        if low_act['high_idx'] < new_end_alfred_index:
            break

        if 'move' in low_act['api_action']['action'].lower():
            new_low_act = copy.deepcopy(low_act)
            new_low_act['high_idx'] = len(data['turk_annotations']['anns'][0]['high_descs'])
            new_traj['plan']['low_actions'].append(copy.deepcopy(new_low_act))

        elif 'rotate' in low_act['api_action']['action'].lower():
            if 'left' in low_act['api_action']['action'].lower():
                new_low_act= {"api_action": {"action": "RotateRight","forceAction": 'true'},"discrete_action": {"action": "RotateRight_90","args": {}},"high_idx": len(data['turk_annotations']['anns'][0]['high_descs'])}
            else:
                new_low_act= {"api_action": {"action": "RotateLeft","forceAction": 'true'},"discrete_action": {"action": "RotateLeft_90","args": {}},"high_idx": len(data['turk_annotations']['anns'][0]['high_descs'])}
            new_traj['plan']['low_actions'].append(copy.deepcopy(new_low_act))

        elif 'look' in low_act['api_action']['action'].lower():
            if 'down' in low_act['api_action']['action'].lower():
                new_low_act= {"api_action": {"action": "LookUp","forceAction": 'true'},"discrete_action": {"action": "LookUp_15","args": {}},"high_idx": len(data['turk_annotations']['anns'][0]['high_descs'])}
            else:
                new_low_act= {"api_action": {"action": "LookDown","forceAction": 'true'},"discrete_action": {"action": "LookDown_15","args": {}},"high_idx": len(data['turk_annotations']['anns'][0]['high_descs'])}
            new_traj['plan']['low_actions'].append(copy.deepcopy(new_low_act))

    # Fix Images
    for idx in list(range(cur_end_alfred_index, new_end_alfred_index-1, -1)):
        for low_act in reversed(all_act[idx]['low']):
            if 'look' in low_act['action']['api_action']['action'].lower() or 'move' in low_act['action']['api_action']['action'].lower() or 'rotate' in low_act['action']['api_action']['action'].lower():
                last_low_idx += 1
                for cnt in range(0, img_cnt[low_act['low_idx']]):
                    prev_image_num = int(new_traj['images'][-1]['image_name'].split(".")[0])
                    new_image_num = str(prev_image_num+1).zfill(9)
                    img_info = {"high_idx": len(data['turk_annotations']['anns'][0]['high_descs']), "image_name": new_image_num+".png", "low_idx": last_low_idx}
                    new_traj['images'].append(copy.deepcopy(img_info))
 
    # Copy all other details 
    new_traj['pddl_params'] = data['pddl_params']
    new_traj['scene'] = data['scene']
    new_traj['task_id'] = data['task_id']
    new_traj['task_type'] = data['task_type']
    new_traj['alfredl_task_type'] = 'revn'

    try:
        if 'high' in noop_act:
            new_traj['expected_final_pos'] = '|'.join(new_traj['plan']['high_pddl'][-2]['planner_action']['location'].strip().split('|')[1:3])
        else:
            new_traj['expected_final_pos'] = '|'.join(new_traj['plan']['high_pddl'][-1]['planner_action']['location'].strip().split('|')[1:3])
    except:
        print(data)

    new_traj['turk_annotations'] = {}
    new_traj['turk_annotations']['anns'] = []
    for ann in data['turk_annotations']['anns']:
        new_ann = {}
        new_ann['assignment_id'] = ann['assignment_id']
        new_ann['high_descs'] = []
        for k in range(0, len(ann['high_descs'])):
            new_ann['high_descs'].append(ann['high_descs'][k])
            
        # add new rev1 annotation
        new_ann['high_descs'].append(rev1_ann.strip().split('$')[0].strip())
        new_ann['task_desc'] = ann['task_desc']
        new_ann['votes'] = ann['votes']
        new_traj['turk_annotations']['anns'].append(new_ann)
     
    # Add type and final pos info

    return new_traj
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))
 
# # printing the field names
# print('Field names are:' + ', '.join(field for field in fields))
 
# #  printing first 5 rows
# print('\nFirst 5 rows are:\n')

# for row in rows:
#     # parsing each column of a row
#     for col in row:
#         print("%10s"%col,end=" <SEP> "),
#     print('\n')

save_path = "/home/arjunakula/amazon_alfred_latest/data/json_2.1.0_alfredl_revn_valid_seen_unseen/"
for row in rows:
    traj_file = row[0]
    traj_file_suffix = "/".join(traj_file.strip().split("/")[-4:])
    rev1_ann = row[2].strip().split("\n")[0]

    fp = open(traj_file)
    traj_data = json.load(fp)

    new_traj_data = save_new_traj_rev1(traj_data, rev1_ann, -1)

    if len(row[2].strip().split("\n")) > 1:
        rev2_ann = row[2].strip().split("\n")[1].strip()
        new_traj_data = save_new_traj_rev1(copy.deepcopy(new_traj_data), rev2_ann, int(rev1_ann.strip().split("$")[1].strip())-1)

        dest_dir = save_path+traj_file_suffix
        if not path.exists('/'.join(dest_dir.split('/')[:-1])):
            os.makedirs('/'.join(dest_dir.split('/')[:-1]))
        with open(dest_dir, "w") as outfile:
            json.dump(new_traj_data, outfile, indent=4) 

