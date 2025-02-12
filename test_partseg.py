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
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(int(cfg["num_votes"]) ):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / int(cfg["num_votes"]) 
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            # np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('instance avg mIOU is: %.5f' % test_metrics['instance_avg_iou'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    return len(TEST_DATASET)

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    len_test = main(args)
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"spend {cost_time}s,average {cost_time / len_test}s")