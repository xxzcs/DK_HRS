import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils.context import disable_tracking_bn_stats
from utils.ramps import exp_rampup
from utils.loss import mse_with_softmax
from utils.loss import kl_div_with_logit

from tqdm import tqdm

from dataset.busdataset_dk import get_bus
from utils import AverageMeter, accuracy

import torchvision.models as models

logger = logging.getLogger(__name__)
best_acc = 0


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
                                    lr_drop_iter,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            no_gress =  float(current_step) / float(max(1, num_warmup_steps))
        
        elif current_step <= lr_drop_iter[0]:
            no_gress = 1
        elif current_step > lr_drop_iter[0] and current_step <= lr_drop_iter[1]:
            no_gress =  0.1
        elif current_step > lr_drop_iter[1] and current_step <= lr_drop_iter[2]:
            no_gress = 0.01
        elif current_step > lr_drop_iter[2]:
            no_gress = 0.001
        # else:
            # no_gress = 1
        return no_gress

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def __l2_normalize(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0),-1)), 1, keepdim=True)[0].view(
            d.size(0),1,1,1)
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d,2.0), tuple(range(1, len(d.size()))), keepdim=True))
    return d

def gen_r_vadv(args, model, x, vlogits, niter):
    # perpare random unit tensor
    d = torch.rand(x.shape).sub(0.5).to(args.device)
    d = __l2_normalize(d)

    # calc adversarial perturbation
    for _ in range(niter):
        d.requires_grad_()
        rlogits = model(x + args.xi * d)
        adv_dist = kl_div_with_logit(rlogits, vlogits)
        adv_dist.backward()
        d = __l2_normalize(d.grad)
        model.zero_grad()
    return 1 * d
def mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))

def linear_rampup(args, current):
    rampup_length = args.epochs 
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
def sigmoid_rampup(args, current):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    rampup_length = args.epochs 
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def cosine_rampdown(args, current):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    rampdown_length = args.epochs 
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
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
                        help='number of total steps to run') #16(2750)
    parser.add_argument('--eval-step', default=110, type=int, #16(110)
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr-drop-iter', nargs="+", 
                        default=[40000//3, 40000*2//3, 40000*8//9])
    parser.add_argument('--batch-size', default=32, type=int,
                        help='train batchsize') #16
    parser.add_argument('--lr', '--learning-rate', default=0.01875, type=float,
                        help='initial learning rate') #imagenet0.3, 16(0.09375)
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
    parser.add_argument('--T', default=0.5, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='pseudo label threshold') #0.7
    parser.add_argument('--dkthreshold', default=0.7, type=float,
                        help='pseudo label threshold') #0.7
    parser.add_argument('--dkweight', default=1.0, type=float,
                        help='the weight of dkloss') 
    parser.add_argument('--dkdecline', default="linear", type=str,
                        help='the declining way of dkloss') 
    parser.add_argument('--xi', default=30.0, type=float,
                        help='parameter to compute the virtual adversarial direction') 
    parser.add_argument('--lrampup', default=30.0, type=float,
                        help='rampup length of exprampup') 
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

    args = parser.parse_args()
    args.lr_drop_iter = [int(val) for val in args.lr_drop_iter]
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

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    # declare the dataset and model used for our dataset
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, val_dataset = get_bus(
        args, 'uda_data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)


    test_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # model = create_model(args)
    # define the resnet18 model
    '''
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    '''
    model = models.__dict__[args.arch]()
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 2)
    
    newfc = model.fc.in_features
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
        optimizer, args.warmup, args.total_steps, args.lr_drop_iter)
    # scheduler = MultiStepLR(optimizer, milestones=[420], gamma=0.1)
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

def info_nce_loss(args, features):

    labels = torch.cat([torch.arange(args.mu * args.batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / 0.07 #args.temperature
    return logits, labels

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        # new losses from vat
        losses_supvat = AverageMeter()
        losses_unsupvat = AverageMeter()
        # dk losses
        losses_dk = AverageMeter()   #

        mask_probs = AverageMeter()
        # mask_dk_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                inputs_u_w, inputs_u_s, inputs_dk = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                inputs_u_w, inputs_u_s, inputs_dk = unlabeled_iter.next()

            # print("############################")
            # print(inputs_x.shape)
            # print(inputs_u_w.shape)
            # print(inputs_u_s.shape)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_dk)), 3*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 3*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s, logits_dk = logits[batch_size:].chunk(3)
            logits_udk = torch.cat([logits_u_s, logits_dk], dim=0)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            # consistency regularization (mse) of weakaugimg and the dkimg
            # mask_dk = max_probs.ge(args.dkthreshold).float()
            # Ldkcon = (F.cross_entropy(logits_dk, targets_u,
            #                       reduction='none') * mask_dk).mean()
            # contrastive loss between strongaug img and dkimg
            if args.dkdecline == "linear":
                weight_dkcon = args.dkweight * (1 - linear_rampup(args, epoch))
            elif args.dkdecline == "lrampup":
                weight_dkcon = args.dkweight * linear_rampup(args, epoch)
            elif args.dkdecline == "srampup":
                weight_dkcon = args.dkweight * sigmoid_rampup(args, epoch)
            elif args.dkdecline == "sigmoid":
                weight_dkcon = args.dkweight * (1 - sigmoid_rampup(args, epoch))
            elif args.dkdecline == "cosine":
                weight_dkcon = args.dkweight * cosine_rampdown(args, epoch)
            elif args.dkdecline == "constant":
                weight_dkcon = args.dkweight 
            logits_sdk, labels_sdk = info_nce_loss(args, logits_udk)
            Ldkcon = F.cross_entropy(logits_sdk, labels_sdk, reduction='mean')
            # weight_dkcon = 1 - linear_rampup(args, epoch)

            # loss = Lx + args.lambda_u * Lu
            # VAT losses
            ## local distributional smoothness (LDS)
            
            with torch.no_grad():
                sup_logits = logits_x.clone().detach()
                unsup_logits = logits_u_w.clone().detach()  #weak loss item
            # with disable_tracking_bn_stats(model):
            sup_data = inputs_x.to(args.device)
            weak_data = inputs_u_w.to(args.device)
            sup_vadv  = gen_r_vadv(args, model, sup_data, sup_logits, 1) 
            rlogits = model(sup_data + sup_vadv)
            sup_lds  = kl_div_with_logit(rlogits, sup_logits)
            rampup = exp_rampup(args.lrampup)
            sup_lds *= rampup(epoch) * 1.0
            
            unsup_vadv  = gen_r_vadv(args, model, weak_data, unsup_logits, 1) 
            vlogits = model(weak_data + unsup_vadv)
            unsup_lds  = kl_div_with_logit(vlogits, unsup_logits)
            unsup_lds *= rampup(epoch) * 1.0
            
            
            loss = Lx + args.lambda_u * Lu + weight_dkcon * Ldkcon + sup_lds + unsup_lds 

            # loss = Lx + args.lambda_u * Lu + Ldkcon + sup_lds + unsup_lds
            
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_dk.update(Ldkcon.item())  #
            losses_supvat.update(sup_lds.item()) #
            losses_unsupvat.update(unsup_lds.item()) #

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            # mask_dk_probs.update(mask_dk.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.7f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_dk: {loss_dk:.4f}. loss_supvat: {loss_supvat:.4f}. loss_unsupvat: {loss_unsupvat:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_dk=losses_dk.avg,
                    loss_supvat=losses_supvat.avg,
                    loss_unsupvat=losses_unsupvat.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        
        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

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

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


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

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

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

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top2.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
