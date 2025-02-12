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
    parser.add_argument('--cfg',type=str,default='config/ShapeNetPart/test_LVPC.json',help='config file')
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
    TEST_DATASET = PartNormalDataset(root=root, npoints=npoints, split='test', seg_classes=seg_classes,
                                     batch_size=int(cfg["batch_size"]),normal_channel=int(cfg["is_normal"]),
                                     seed=int(cfg["seed"]),sample_method=cfg["sample_method"])
    
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=int(cfg["batch_size"]),  pin_memory=True,
                                                 shuffle=False, num_workers=int(cfg["workers"]))
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    
    

    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(cfg["model"])
    classifier = MODEL.get_model(npoints,num_classes,num_part, normal_channel=int(cfg["is_normal"])).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier =  torch.nn.DataParallel(classifier) 
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        classifier = classifier.eval()
        for batch_id, (points, label, target,filepath) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),smoothing=0.9):
            path_list = filepath[0].split('/')
            save_root = root + "_pred"
            if os.path.exists(save_root) is False:
                os.mkdir(save_root)
            save_folder = os.path.join(save_root,path_list[-2])
            filenamelist = path_list[-1].split('.')
            save_filepath = os.path.join(save_folder,path_list[-1])
            save_filepath_origin = os.path.join(save_folder,filenamelist[0]+"_origin." +filenamelist[1])
            if os.path.exists(save_folder) is False:
                os.mkdir(save_folder)

            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            for _ in range(int(cfg["num_votes"]) ):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / int(cfg["num_votes"]) 
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, axis=2).flatten()  

                
            points = points.transpose(2, 1).cpu().numpy().reshape(-1, 6)  
            target = target.cpu().numpy().flatten()  

            combined_results = np.column_stack((points, cur_pred_val))
            origin_pointscloud = np.column_stack((points,target))

            np.savetxt(save_filepath, combined_results, fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f', delimiter=' ')

if __name__ == '__main__':
    args = parse_args()
    main(args)