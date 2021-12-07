import argparse
import os
import sys
import math
import time
import shutil
import tarfile
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from opts import parser
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
import ops.logging as logging
import ops.utils as utils
from ops.SoftwarePipeLine import SoftwarePipeLine
from ops.CosineAnnealingLR import WarmupCosineLR
from ops.LabelSmoothing import LabelSmoothingLoss
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_

from sklearn.metrics import confusion_matrix



best_prec1 = 0

logger = logging.get_logger(__name__)


def main():
    
    
    global args, best_prec1
    args = parser.parse_args()
    
    # Setup logging format.
    logging.setup_logging(args.log_dir)
    
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))

    args_dict = args.__dict__
    logger.info("------------------------------------")
    logger.info(args.arch+" Configurations:")
    for key in args_dict.keys():
        logger.info("- {}: {}".format(key, args_dict[key]))
    logger.info("------------------------------------")
    logger.info (args.mode)
    if args.dataset == 'ucf101':
        num_class = 101
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        rgb_read_format = "{:05d}.jpg"        
    elif args.dataset == 'kinetics':
        num_class = 400
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'something':
        num_class = 174
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'diving48':
        num_class = 48
        rgb_read_format = "{:05d}.jpg"        
    elif args.dataset == 'finegym99':
        num_class = 99
        rgb_read_format = "{:05d}.jpg" 
    elif args.dataset == 'finegym288':
        num_class = 288
        rgb_read_format = "{:05d}.jpg"         
    elif args.dataset == 'minikinetics':
        num_class = 150
        rgb_read_format = "{:05d}.jpg"        
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    
    ############ Intervolution Configs ############
#     transform_config = {
#                     'position': [[2],[1,3],[1,3,5],[1]],
#                     'transform': 'Intervolution', # ['conv', 'LSA', 'Intervolution']
#                     'kernel_size': [5,7,7],
#                     'nh':8,
#                     'dk':0,
#                     'dv':0,
#                     'dd':0,
#                     'kernel_type':'VplusR', # ['V', 'R', 'VplusR']
#                     'feat_type':'VplusR', # ['V', 'R', 'VplusR']
#                 }
    transform_config = {
                    'transform': args.transform, # ['conv', 'LSA', 'Intervolution']
                    'position': eval(args.position),
                    'kernel_size': eval(args.kernel_size),
                    'nh': args.nh,
                    'dk': args.dk,
                    'dv': args.dv,
                    'dd': args.dd,
                    'kernel_type': args.kernel_type, # ['V', 'R', 'VplusR']
                    'feat_type': args.feat_type, # ['V', 'R', 'VplusR']
                }
    ###############################################
    
    model = TSN(num_class, args.num_segments, args.pretrained_parts, args.modality, args.dataset,
                base_model=args.arch, transform = transform_config,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn, stochastic_depth = args.stochastic_depth)
    
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    
    policies = model.get_optim_policies()

    train_augmentation = model.get_augmentation()
    
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    model_dict = model.state_dict()
    
    logger.info("Model:\n{}".format(model))
    
    logger.info("pretrained_parts: {}".format(args.pretrained_parts))

    if args.arch == "ResNet":
        pretrained_dict={}
        new_state_dict = {} #model_dict
        for k, v in model_dict.items():
            if ('fc' not in k):
                new_state_dict.update({k:v})
        div = True
        roll = False   
    else:
        raise ValueError('Unknown base model: {}'.format(args.arch))
    
    
    un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
    logger.info("un_init_dict_keys: {}".format(un_init_dict_keys))
    logger.info("\n------------------------------------")

    for k in un_init_dict_keys:
        new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
        if 'weight' in k:
            if 'bn' in k:
                logger.info("{} init as: 1".format(k))
                constant_(new_state_dict[k], 1)
            else:
                logger.info("{} init as: xavier".format(k))
                xavier_uniform_(new_state_dict[k])
        elif 'bias' in k:
            logger.info("{} init as: 0".format(k))
            constant_(new_state_dict[k], 0)
    
    
    logger.info("------------------------------------")
    utils.get_FLOPs_params(model, args.arch, torch.randn(args.num_segments,3,crop_size,crop_size).cuda())
    logger.info("------------------------------------")
    
    model.load_state_dict(new_state_dict)
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 1
    
    if args.dataset in ['kinetics', 'minikinetics']:
        TrainSource = Kinetics
        ValSource = Kinetics
        TestSource = Kinetics
    elif args.dataset == 'something':
        TrainSource = TSNDataSet
        ValSource = TSNDataSet 
        TestSource = TSNDataSet 
        
    train_source = TrainSource("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   mode = args.mode,
                   image_tmpl=args.rgb_prefix+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       GroupScale((240,320)),                              
                       train_augmentation,
                       Stack(roll=roll),
                       ToTorchFormatTensor(div=div),
                       normalize,
                   ]))
    
    val_source = ValSource("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   mode =args.mode,
                   image_tmpl=args.rgb_prefix+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   random_shift=False,
                   test_mode=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale((240,320)),                         
                       GroupCenterCrop(crop_size),
                       Stack(roll=roll),
                       ToTorchFormatTensor(div=div),
                       normalize,
                   ]))
    
    
    train_loader = SoftwarePipeLine(torch.utils.data.DataLoader(
        train_source,
        batch_size=args.batch_size, shuffle=True,  
        num_workers=args.workers, pin_memory=True))
    
    val_loader = SoftwarePipeLine(torch.utils.data.DataLoader(
        val_source,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True))
    
    test_loader = SoftwarePipeLine(torch.utils.data.DataLoader(
        val_source,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True))
    
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'smooth_nll':
        criterion = LabelSmoothingLoss(args.label_smoothness).cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        logger.info(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
    
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,nesterov=args.nesterov)
    
    # If cosine learning rate decay
    if args.cosine_lr:
        args.lr_steps = [args.epochs]
        lr_scheduler_clr = WarmupCosineLR(optimizer=optimizer, milestones=[args.warmup, args.epochs], warmup_iters=args.warmup, min_ratio=1e-3, dataset=args.dataset)
        lr_scheduler_clr.last_epoch = args.start_epoch
    
    # To save result score list
    output_list = []
    if args.evaluate:
        prec1, score_tensor = validate(test_loader,model,criterion,0)
        output_list.append(score_tensor)
        fn='score_240c1.pt'
        save_validation_score(output_list, filename=fn)
        logger.info("validation score saved in {}".format('/'.join((args.val_output_folder, fn))))
        return
    
        
    #############################################################
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps, args.num_long_cycles, args.last_cycle_tune)   ##### original code   
        optimizer.step()
        if args.cosine_lr:
            lr_scheduler_clr.step()     
        total_iter = train(train_loader, model, criterion, optimizer, epoch)      
    
        # train for one epoch
    ##############multi-grid implementation######################
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch >= args.epochs - 10:
            prec1, score_tensor = validate(val_loader, model, criterion, total_iter)
            
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    

    

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_time = AverageMeter()
    model_speed = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(True)

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    
    batch_size = args.batch_size
    end = time.time()
    
    
    for i, (input, target, video_idx) in enumerate(train_loader):
        # discard final batch
        if i == len(train_loader)-1:
            break 
        
        # measure data loading time
        load_time = time.time()
        data_time.update(load_time - end)
        
        # target size: [batch_size]
        target = target.cuda()
        
        #############################################################
        input_var = input
        target_var = target
        
        # We do not use mixup data augmentation.
        input_var,target_var_a,target_var_b,lam = mixup_data(input_var,target_var,args.mixup_alpha)
    
        output = model(input_var)
        
        loss = mixup_criterion(criterion,output,target_var_a,target_var_b,lam)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5)) #target
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        loss.backward()
        
        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    logger.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            else:
                total_norm = 0

            optimizer.step()
            optimizer.zero_grad()           
        
        
        # total iterations
        total_iter = len(train_loader) * epoch + i
        
        
        # measure elapsed time
        val_time.update(time.time() - load_time)
        model_speed.update(batch_size / val_time.val)
        batch_time.update(time.time() - end)
        
        if i % args.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                   'Speed {model_speed.val:.3f} V/s ({model_speed.avg:.4f}) V/s'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1=top1, top5=top5, model_speed=model_speed, lr=optimizer.param_groups[-2]['lr'])))
        end = time.time()
    
    return total_iter

def validate(val_loader, model, criterion, total_iter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_time = AverageMeter()
    model_speed = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)
    
    # switch to evaluate mode
    model.eval()
    
    # To save output softmax score results
    output_list = []
    batch_size = args.batch_size
    pred_arr = []
    target_arr = []
    end = time.time()
    for i, (input, target, video_idx) in enumerate(val_loader):
        # discard final batch
#         if i == len(val_loader)-1:
#             break

        # measure data loading time
        load_time = time.time()
        data_time.update(load_time - end)
        
        input_var = input
        target = target.cuda()
        target_var = target
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)   
        
        # class acc
        pred = torch.argmax(output.data, dim=1)
        pred_arr.extend(pred.cpu().numpy())
        target_arr.extend(target.cpu().numpy())
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        # measure elapsed time
        val_time.update(time.time() - load_time)
        model_speed.update(batch_size / val_time.val)
        batch_time.update(time.time() - end)
        
        # put all results into output_list
        output_list.append(output)
        
#         logger.info(output.size())
        if i % args.print_freq == 0:
            logger.info(('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))
            logger.info('Runtime: [{0}/{1}]\t'
                  'Speed: {model_speed.val:.3f} V/s ({model_speed.avg:.4f} V/s)\t'
                  'val_time: {val_time.val:.3f} ({val_time.avg:.4f})\t'
                  'data_time: {data_time.val:.3f} ({data_time.avg:.4f})\t'
                  'batch_time: {batch_time.val:.3f} ({batch_time.avg:.4f})'.format(
                  i, len(val_loader), model_speed=model_speed, val_time=val_time, data_time=data_time, batch_time=batch_time))
        end = time.time()
    
    output_tensor = torch.cat(output_list, dim=0)
    
        
    logger.info(('Validation Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f} Time {batch_time.avg:.4f} Speed {model_speed.avg:.4f} V/s'
          .format(top1=top1, top5=top5, loss=losses, batch_time=batch_time, model_speed=model_speed)))
        
    target_arr = np.array(target_arr)
    pred_arr = np.array(pred_arr)
    cf = confusion_matrix(target_arr, pred_arr).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/(cls_cnt+0.0001)
    logger.info ('Class Accuracy {:.02f}%'.format(np.mean(cls_acc)*100))    
    return top1.avg, output_tensor



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), "epoch", str(state['epoch']), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

def save_validation_score(score, filename='score.pt'):
    filename = '/'.join((args.val_output_folder, filename))
    torch.save(score, filename)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    ##############multi-grid implementation#################
    # We do not use multi-grid training.
def _parse_grid(lr_steps, num_cycles):
    grids = []
    lr_steps_ = [0]+lr_steps[:-1]
    for i, (prev,post) in enumerate(zip(lr_steps_,lr_steps)):
        endpoint = True if post == lr_steps[-1] else False
        num = num_cycles+1 if post == lr_steps[-1] else num_cycles
        grids = np.append(grids,np.linspace(prev,post,num,endpoint=endpoint))
    grids = grids.astype(int)
    
    return grids

def adjust_learning_rate(optimizer, epoch, lr_steps, num_cycles, finetune=False):
    """Sets the learning rate along the multigrid cycle"""
    lr_factors = 1
    if num_cycles > 0 and epoch < lr_steps[-1]: 
        grids = _parse_grid(lr_steps, num_cycles)
        turn = sum(epoch>=grids)-1
        lr_factors = 2 ** (num_cycles-1-(turn % num_cycles))
    #######################################################
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay * lr_factors #multi-grid implementation
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']        

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# We do not use mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

        
if __name__ == '__main__':
    main()
