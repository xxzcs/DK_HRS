import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset.busdataset_dk import get_bus
from utils import AverageMeter, accuracy

import torchvision.models as models

logger = logging.getLogger(__name__)
best_acc = 0
mean=(0.371, 0.371, 0.372)
std=(0.167, 0.167, 0.167)

def save_checkpoint(state, epoch, is_best, checkpoint, filename='checkpoint.pth.tar'):
    '''
    filepath = os.path.join(checkpoint, filename)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
    '''
    new_filename = filename[0:-8] + str(epoch) + '.pth.tar'
    torch.save(state, os.path.join(checkpoint, new_filename))
    if is_best:
        best_name = 'model_best' + str(epoch) + '.pth.tar'
        shutil.copyfile(os.path.join(checkpoint, new_filename), os.path.join(checkpoint, best_name))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='bus', type=str,
                        choices=['cifar10', 'bus', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=878,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['wideresnet', 'resnet18', 'resnext'],
                        help='net name')
    parser.add_argument('--total-steps', default=1400, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=28, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.01875, type=float,
                        help='initial learning rate') #0.3
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=5, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='pseudo label threshold') #0.7
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--labeledpath', default='', type=str, metavar='LPATH',
                    help='path of the labeled data')
    parser.add_argument('--unlabeledpath', default='', type=str, metavar='ULPATH',
                    help='path of the unlabeled data')
    parser.add_argument('--csvname', type=str, help='csv file name')

    args = parser.parse_args()
    global best_acc
   
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    # print args.world_size, which indicates the utilized count of GPUs 
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)
    '''
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    '''
    # declare the dataset and model used for our dataset
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # labeled_dataset, unlabeled_dataset, val_dataset = get_bus(
        # args, '../uda_data')

    if args.local_rank == 0:
        torch.distributed.barrier()
    valdir = '../uda_data/test'
    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]))
    test_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()


    model = models.__dict__[args.arch]()
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 2)
    # model.fc = torch.nn.Linear(num_ftrs, 128)
    # add mlp projection head
    newfc = model.fc.in_features
    # model.fc = torch.nn.Sequential(torch.nn.Linear(newfc,newfc), torch.nn.ReLU(), torch.nn.Linear(newfc, 2))
    model.fc = torch.nn.Sequential(torch.nn.Linear(newfc,newfc), torch.nn.ReLU(), torch.nn.Linear(newfc, 2))

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    # if args.use_ema:
    #     from models.ema import ModelEMA
    #     ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint directory '{}'..".format(args.resume))
        # assert os.path.isfile(
            # args.resume), "Error: no checkpoint directory found!"
        model_list = ['checkpoint47.pth.tar','checkpoint48.pth.tar','checkpoint49.pth.tar']
        # model_list = ['checkpoint97.pth.tar','checkpoint98.pth.tar','checkpoint99.pth.tar']
        acc_list = []
        all_list = os.listdir(args.resume)
        d_list = [name for name in all_list if name not in model_list]
        for name in d_list:
            os.remove(os.path.join(args.resume, name))

        for i in range(len(model_list)):
            modelname = os.path.join(args.resume, model_list[i])
            # checkpoint = torch.load(modelname)
            print("loading checkpoint: " + modelname)
            args.out = os.path.dirname(modelname)
            checkpoint = torch.load(modelname)
            best_acc = checkpoint['best_acc']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # if args.use_ema:
            #     ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # if args.use_ema:
            #     test_model = ema_model.ema
            # else:
            test_model = model
            test_accs = []
            if args.local_rank in [-1, 0]:
                epoch = args.start_epoch
                test_loss, test_acc = test(args, test_loader, test_model, epoch)
                # test_loss, test_acc = test_csv(args, test_loader, val_dataset, test_model, epoch)
                # args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
                # args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
                acc_list.append(test_acc)
                # is_best = test_acc > best_acc
                # best_acc = max(test_acc, best_acc)
        best_model_index = 0
        best_acc1 = acc_list[0]
        for i in range(1, len(acc_list)):
            if acc_list[i] >= best_acc1:
                best_model_index = i
                best_acc1 = acc_list[i]
        best_model_name = model_list[best_model_index]
        print("loading best model name: " + best_model_name)
        best_model_name = os.path.join(args.resume, best_model_name)
        checkpoint = torch.load(best_model_name)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # if args.use_ema:
        #     ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # if args.use_ema:
        #     test_model = ema_model.ema
        # else:
        test_model = model
        test_accs = []
        if args.local_rank in [-1, 0]:
            epoch = args.start_epoch
            # test_loss, test_acc = test(args, test_loader, test_model, epoch)
            test_loss, test_acc = test_csv(args, test_loader, val_dataset, test_model, epoch)
            # args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            # args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            acc_list.append(test_acc)
            '''
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, epoch, is_best, args.out)
            '''
            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
    '''
    if args.local_rank in [-1, 0]:
        args.writer.close()
    '''
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    classes = ('benign','malignant')
    N_CLASSES = 2

    class_correct = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            # compute the predict accuracy for each category
            class_result = outputs.argmax(dim=1)
            c = (class_result == targets).squeeze()
            for i in range(len(targets)):
                _label = targets[i]
                class_correct[_label] += c[i].item()
                class_total[_label] += 1

            prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top2.update(prec2.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top2:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                ))
        if not args.no_progress:
            test_loader.close()
        for i in range(N_CLASSES):
            print("###############################")
            print('Accuracy of %5s (%3d of %3d): %2d %%' % (
                            classes[i], class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))
    
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top2.avg))
    return losses.avg, top1.avg
def test_csv(args, test_loader, val_dataset, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    classes = ('benign','malignant')
    N_CLASSES = 2

    class_correct = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))

    with torch.no_grad():
        with open(args.csvname, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['imagename', 'gt_label', 'gtname', 'p1','p2'])

            for batch_idx, (inputs, targets) in enumerate(test_loader):
                data_time.update(time.time() - end)
                model.eval()

                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

                # compute the predict accuracy for each category
                class_result = outputs.argmax(dim=1)
                c = (class_result == targets).squeeze()
                for i in range(len(targets)):
                    _label = targets[i]
                    class_correct[_label] += c[i].item()
                    class_total[_label] += 1

                prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
                losses.update(loss.item(), inputs.shape[0])
                top1.update(prec1.item(), inputs.shape[0])
                top2.update(prec2.item(), inputs.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
                # write the probability of each category into csv 
                # print(output)
                output_softmax = F.softmax(outputs, dim=1)
                pp_list = output_softmax.tolist()
            
                batch_number = len(inputs)
                               
                for j in range(batch_number):
                    
                    img_name = os.path.basename(val_dataset.imgs[j + batch_idx * 32][0])
                    gt_label = val_dataset.imgs[j + batch_idx * 32][1]
                    benign_probability = pp_list[j][0]
                    malignant_probability = pp_list[j][1]
                    cls_name = 'benign' if gt_label ==0 else 'malignant'
                    writer.writerow([img_name, gt_label, cls_name, round(benign_probability,9), round(malignant_probability,9)])

                if not args.no_progress:
                    test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top2:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top2=top2.avg,
                    ))
        if not args.no_progress:
            test_loader.close()
        for i in range(N_CLASSES):
            print('Accuracy of %5s (%3d of %3d): %2d %%' % (
                            classes[i], class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))
    
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top2.avg))
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()
