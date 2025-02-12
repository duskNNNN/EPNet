import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import random
import provider
import json5
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--cfg',type=str,default='config/ShapeNetPart/test_save.json',help='config file')
    return parser.parse_args()


def main(args):
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfg = json5.load(f)
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu"]
    experiment_dir = 'log/part_seg/' + cfg["logdir"] 

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(json5.dumps(cfg, indent=4))

    provider.set_seed(int(cfg["seed"]))
    root = cfg["datapath"]
    if os.path.exists(root) is False:
        print(f"{root} not found")
        os._exit(-1)
    num_classes = int(cfg["classes"])
    num_part = int(cfg["parts"])
    seg_classes = cfg["content"]
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    npoints=int(cfg["npoint"])
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(cfg["model"])
    classifier = MODEL.get_model(npoints,num_classes,num_part, normal_channel=int(cfg["is_normal"])).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier =  torch.nn.DataParallel(classifier) 
    classifier.load_state_dict(checkpoint['model_state_dict'])

    files = os.listdir(root)
    save_folder_path = root +'_pred'
    if os.path.exists(save_folder_path) is False:
        os.mkdir(save_folder_path)
    label = np.array([0]).astype(np.int32)
    with torch.no_grad():
        classifier = classifier.eval()
        for (src) in tqdm(enumerate(files),total=len(files),smoothing=0.9):
            filepath = os.path.join(root,src[1])
            data = np.loadtxt(filepath).astype(np.float32)
            if int(cfg["is_normal"]) == 0:
                points = data[:, 0:3]
            else:
                points = data[:, 0:6]
            points[:, 0:3] = provider.pc_normalize_noscale(points[:, 0:3])
            choice = np.random.choice(len(points), npoints, replace=True)
            points = points[choice, :]
            points = points[np.newaxis, :, :]
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points)
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            points, label = points.float().cuda(), label.long().cuda()
            points = points.transpose(2, 1)
            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, axis=2).flatten()  
                
            if int(cfg["is_normal"]) == 0:
                points = points.transpose(2, 1).cpu().numpy().reshape(-1, 3)  
            else:
                points = points.transpose(2, 1).cpu().numpy().reshape(-1, 6)  
            combined_results = np.column_stack((points, cur_pred_val))
            save_path = os.path.join(save_folder_path,src[1])
            np.savetxt(save_path, combined_results, fmt='%.6f', delimiter=' ')
            

if __name__ == '__main__':
    args = parse_args()
    main(args)