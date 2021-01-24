import os
import json
import argparse
import numpy as np
from glob import glob
import re
from natsort import natsorted

parser = argparse.ArgumentParser(description='raw_nyu_v2')
parser.add_argument('--dataset', type=str, default='scannet')
parser.add_argument('--test_loc', type=str, default='end')
parser.add_argument('--frame_interval', type=int, default=2)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--overlap', type=int, default=0)
parser.add_argument('--root_dir', type=str, default='/userhome/35/xxlong/dataset/scannet_test/')

args = parser.parse_args()



def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def video_split(frame_len, frame_train, interval, overlap):
    sample_interval = frame_train - overlap
    indices = []
    for start in range(interval):
        index_list = list(range(start, frame_len - frame_train * interval + 1, sample_interval))
        [indices.append(list(range(num, num + frame_train * interval, interval))) for num in index_list]
        indices.append(list(range(frame_len - frame_train * interval - start, frame_len - start, interval)))

    return indices


def create_dict(dataset, root_dir, frame_interval, seq_len, start=0, end=10):
    scenes = natsorted([x for x in os.listdir(root_dir) if "scene" in x])
    test_dict = []
    for scene in scenes[start:end]:
        scene_dir = os.path.join(root_dir, scene)
        print(scene_dir)
        rgb_list = glob(scene_dir + '/rgb/*.jpg')
        depth_list = glob(scene_dir + '/depth/*.png')
        rgb_list = natsorted(rgb_list)
        depth_list = natsorted(depth_list)

        assert len(rgb_list) == len(depth_list)

        for index, (rgb_file, depth_file) in enumerate(zip(rgb_list, depth_list)):
            rgb_index = int(re.findall(r'\d+', os.path.basename(rgb_file))[0])
            depth_index = int(re.findall(r'\d+', os.path.basename(depth_file))[0])

            assert rgb_index == depth_index

            if rgb_index % 10 == 0 and rgb_index > 0:
                test_info = {
                    'rgb_index': rgb_list[(index + 1) - seq_len: index + 1],
                    'depth_index': depth_list[(index + 1) - seq_len: index + 1],
                    'scene_name': scene,
                    'test_index': seq_len - 1
                }
                test_dict.append(test_info)

    test_info_save = './{}_frameinterval{}_seqlen{}_test_start{}_end{}.json'.format(dataset, frame_interval, seq_len, start, end)

    with open(test_info_save, 'w') as dst_file:
        json.dump(test_dict, dst_file)


if __name__ == '__main__':
    create_dict(args.dataset, args.root_dir, args.frame_interval, args.seq_len)
