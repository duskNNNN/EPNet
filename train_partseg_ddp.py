import argparse
from ast import arg
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import time
import torch.optim
import provider
import numpy as np
import json5
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--cfg',type=str,default='config/ShapeNetPart/train_LVPC.json',help='config file')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

def setup_distributed():
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()

    torch.cuda.set_device(local_rank)
    return local_rank

def main(args):
    args = parse_args()
    local_rank = setup_distributed()
    with open(args.cfg, 'r') as f:
        cfg = json5.load(f)
    def log_string(str):
        logger.info(str)
        print(str)

    is_main_process = local_rank == 0
    exp_dir = Path('./log/')
    if is_main_process:
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        exp_dir.mkdir(exist_ok=True)
        exp_dir = exp_dir.joinpath('part_seg')
        exp_dir.mkdir(exist_ok=True)
        exp_dir = exp_dir.joinpath(cfg["logdir"])
        exp_dir.mkdir(exist_ok=True)
        if cfg["logdir"] is None:
            exp_dir = exp_dir.joinpath(timestr)
        exp_dir.mkdir(exist_ok=True)
        checkpoints_dir = exp_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = exp_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)
        tensorboard_log = exp_dir.joinpath('runs/')
        writer = SummaryWriter(tensorboard_log)
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, cfg["model"]))
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
    npoints=int(cfg["npoint"])
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=npoints, split='trainval', seg_classes=seg_classes,
                                      batch_size=int(cfg["batch_size"]),normal_channel=int(cfg["is_normal"]),
                                      seed=int(cfg["seed"]),sample_method=cfg["sample_method"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET,sampler=train_sampler, batch_size=int(cfg["batch_size"]),pin_memory=True,
                                                   num_workers=int(cfg["workers"]) , drop_last=True)
    
    TEST_DATASET = PartNormalDataset(root=root, npoints=npoints, split='test', seg_classes=seg_classes,
                                     batch_size=int(cfg["batch_size"]),normal_channel=int(cfg["is_normal"]),
                                     seed=int(cfg["seed"]),sample_method=cfg["sample_method"])
    
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=int(cfg["batch_size"]), pin_memory=True,
                                                 shuffle=False, num_workers=int(cfg["workers"]) )
    
    if is_main_process:
        log_string("The number of training data is: %d" % len(TRAIN_DATASET))
        log_string("The number of test data is: %d" % len(TEST_DATASET))

    MODEL = importlib.import_module(cfg["model"])
    if is_main_process:
        shutil.copy('models/%s.py' % cfg["model"], str(exp_dir))
        shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(npoints,num_classes,num_part, normal_channel=int(cfg["is_normal"])).cuda()
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    criterion = MODEL.get_loss(cfg["loss"]).cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    checkpoint_path = str(exp_dir) + '/checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if is_main_process:

            log_string('Use pretrain model')
    else:
        if is_main_process:

            log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    optimizer = provider.set_optimizer(cfg,classifier)
    if cfg["schedule"] != 'origin':
        lr_schedule = provider.set_schedule(cfg,optimizer)
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_instance_avg_iou = 0
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = int(cfg["step_size"])
    for epoch in range(start_epoch, int(cfg["epoch"])):
        train_sampler.set_epoch(epoch)
        mean_correct = []
        if is_main_process:
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, int(cfg["epoch"])))
        if cfg["schedule"] == 'origin':
            lr = max(float(cfg["learning_rate"]) * (float(cfg["lr_decay"]) ** (epoch // int(cfg["step_size"]))), LEARNING_RATE_CLIP)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
            if momentum < 0.01:
                momentum = 0.01
            log_string('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        else:
            lr=optimizer.param_groups[-1]['lr']
        if is_main_process:
            log_string('Learning rate:%f' % lr)
        classifier = classifier.train()
        if is_main_process:
            writer.add_scalar('Learning Rate', lr, global_epoch)  
        if cfg["schedule"] != 'origin':
            lr_schedule.step()
        total_train_loss = 0
        '''learning one epoch'''
        for i, (points, label, target,filepath) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3],0.9,1.1)
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3],0.05)
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (int(cfg["batch_size"]) * npoints))
            loss = criterion(seg_pred, target)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(trainDataLoader)
        train_instance_acc = np.mean(mean_correct)
        if is_main_process:
            log_string('Train Loss: %.5f Train accuracy: %.5f' % (avg_train_loss, train_instance_acc))
            writer.add_scalar('Loss/train', avg_train_loss, global_epoch)
            writer.add_scalar('Accuracy/train', train_instance_acc, global_epoch)
        if is_main_process:
            total_val_loss = 0

            with torch.no_grad():
                test_metrics = {}
                total_correct = 0
                total_seen = 0
                total_seen_class = [0 for _ in range(num_part)]
                total_correct_class = [0 for _ in range(num_part)]
                shape_ious = {cat: [] for cat in seg_classes.keys()}

                classifier = classifier.eval()

                for batch_id, (points, label, target,filepath) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    seg_pred_val = seg_pred.contiguous().view(-1, num_part)
                    target_val = target.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred_val, target_val)
                    total_val_loss += loss.item()
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
                    np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
                for cat in sorted(shape_ious.keys()):
                    log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
                test_metrics['class_avg_iou'] = mean_shape_ious
                test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)
            avg_val_loss = total_val_loss / len(testDataLoader)
            log_string('Validation Loss: %.5f' % avg_val_loss)
            writer.add_scalar('Loss/validation', avg_val_loss, global_epoch)
            log_string('Epoch %d test Accuracy: %f  Instance avg mIOU: %f   Class avg mIOU: %f' % (
                epoch + 1, test_metrics['accuracy'], test_metrics['instance_avg_iou'], test_metrics['class_avg_iou']))
            writer.add_scalar('Accuracy/test', test_metrics['accuracy'], global_epoch)
            writer.add_scalar('mIOU/instance', test_metrics['instance_avg_iou'], global_epoch)
            writer.add_scalar('mIOU/class', test_metrics['class_avg_iou'], global_epoch)
            if (test_metrics['instance_avg_iou'] >= best_instance_avg_iou):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'class_avg_iou': test_metrics['class_avg_iou'],
                    'instance_avg_iou': test_metrics['instance_avg_iou'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                best_class_avg_iou = test_metrics['class_avg_iou']
            if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
                best_instance_avg_iou = test_metrics['instance_avg_iou']
            log_string('Best accuracy is: %.5f' % best_acc)
            log_string('Best instance avg mIOU is: %.5f' % best_instance_avg_iou)
            log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
            global_epoch += 1
        if is_main_process:
            writer.close()
    return cfg["logdir"],best_acc,best_instance_avg_iou


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    logdir,acc,iou = main(args)
    end_time = time.time()
    cost_time = end_time - start_time
    hours = cost_time // 3600
    minutes = (cost_time - 3600 * hours) // 60
    seconds = np.floor(cost_time - 3600 * hours - minutes * 60) 
    print(f"spend {hours}h {minutes}m {seconds}s")
