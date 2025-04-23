"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from dataset import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_classes = {'Car':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=102400, help='point Number')
    parser.add_argument('--log_dir', type=str, required=False,default='exp_v3', help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    root = '/config/workspace/wc/ACT/data/LVPC'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    # num_classes = 16
    num_classes = 1
    # num_part = 50
    num_part = 18

    '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module('pt')
    classifier = MODEL.get_model(num_part).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target,filepath) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            # print(f"当前处理的文件是: {filepath}")
            path_list = filepath[0].split('/')
            # print(f"文件名: {path_list[-1]}")
            # print(f"所属文件夹: {path_list[-2]}")
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
            # 将经过投票池之后的结果保存
            for _ in range(args.num_votes):
                seg_pred = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, axis=2).flatten()  # 获取预测标签

                
            # 转换为 CPU 格式并进行数值计算
            points = points.transpose(2, 1).cpu().numpy().reshape(-1, 3)  # 转回原始形状并转换为 NumPy (x, y, z)
            target = target.cpu().numpy().flatten()  # 获取实际标签
            print(f"points shape is {points.shape}")
            print(f"cur_pred_val shape is {cur_pred_val.shape}")
            # 将点云和预测标签拼接在一起，保存格式为 x, y, z,  predicted_label
            combined_results = np.column_stack((points, cur_pred_val))
            origin_pointscloud = np.column_stack((points,target))

            # 保存结果到文件，不包含 header
            np.savetxt(save_filepath, combined_results, fmt='%.6f %.6f %.6f %.6f', delimiter=' ')
            # 原始的也保留
            # np.savetxt(save_filepath_origin, origin_pointscloud, fmt='%.6f %.6f %.6f %.6f', delimiter=' ')
            # print(f"Saved results to {save_filepath}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
