"""
(Training, Generating edge maps)
Pixel Difference Networks for Efficient Edge Detection (accepted as an ICCV 2021 oral)
See paper in https://arxiv.org/abs/2108.07009

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import os
import cv2
import random
import torch
import numpy
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from edge_dataloader import BSDS_VOCLoader, BSDS_Loader, Multicue_Loader, \
    NYUD_Loader, Custom_Loader, BIPED_Loader
from utils import get_model_parm_nums, load_checkpoint, save_checkpoint, \
    AverageMeter, adjust_learning_rate, cross_entropy_loss_RCF, Logger
from models.convert_pidinet import convert_pidinet,convert_pidinet_v2
parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='output',
                    help='path to save result and checkpoint')

parser.add_argument('--ablation', action='store_true',
                    help='not use bsds val set for training')
parser.add_argument('--dataset', type=str, default='BSDS',
                    choices=['BSDS', 'BSDS-PASCAL', 'NYUD-image', 'NYUD-hha', 'Multicue-boundary-1',
                             'Multicue-boundary-2', 'Multicue-boundary-3', 'Multicue-edge-1', 'Multicue-edge-2',
                             'Multicue-edge-3', 'Custom', 'BIPED', "UDED", "BRIND"],
                    help='data settings for BSDS, Multicue and NYUD datasets')

parser.add_argument('--model', type=str, default='baseline',
                    # choices=['pidinet_tiny', 'pidinet_small', 'pidinet', 'pidinet_converted',
                    #          'pidinet_tiny_v2', 'pidinet_small_v2', 'pidinet_v2',
                    #          'pidinet_tiny_v20', 'pidinet_small_v20', 'pidinet_v20',
                    #          'pidinet_tiny_v200', 'pidinet_small_v200', 'pidinet_v200', ],
                    help='model to train the dataset')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--loss_weight', default="same", choices=["same", "optim"])
parser.add_argument('--sa', action='store_true',
                    help='use CSAM in pidinet')
parser.add_argument('--dil', action='store_true',
                    help='use CDCM in pidinet')
parser.add_argument('--config', type=str, default='carv4',
                    help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=1334,
                    help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default=None,
                    help='gpus available')
parser.add_argument('--checkinfo', action='store_true',
                    help='only check the informations about the model: model size, flops')

parser.add_argument('--epochs', type=int, default=20,
                    help='number of total epochs to run')
parser.add_argument('--iter-size', type=int, default=24,
                    help='number of samples in each iteration')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate for all weights')
parser.add_argument('--lr-type', type=str, default='multistep',
                    help='learning rate strategy [cosine, multistep]')
parser.add_argument('--lr-steps', type=str, default="10-14",
                    help='steps for multistep learning rate')
parser.add_argument('--opt', type=str, default='adam',
                    help='optimizer')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay for all weights')
# parser.add_argument('-j', '--workers', type=int, default=4,
#                     help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3,
                    help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--lmbda', type=float, default=1.1,
                    help='weight on negative pixels (the beta parameter in the paper)')

parser.add_argument('--resume', default=None, help='load checkpoint if not None')
parser.add_argument('--print-freq', type=int, default=100,
                    help='print frequency')
parser.add_argument('--save-freq', type=int, default=1,
                    help='save frequency')
parser.add_argument('--evaluate', type=str, default=None,
                    help='full path to checkpoint to be evaluated')
parser.add_argument('--evaluate-converted', action='store_true',
                    help='convert the checkpoint to vanilla cnn, then evaluate')

parser.add_argument('--note', type=str, required=True)
# parser.add_argument('--pretrain', type=str, default="pretrain_fileV2/checkpoint.pth.tar")
parser.add_argument('--pretrain', type=str, default=" ")
parser.add_argument('--act', type=str, default="RReLU",
                    choices=["ReLU", "PReLU", "GELU", "SiLU", "Hardswish", "RReLU", "Abss", "Square"],
                    help="we use RReLU in PiDiNetv2")
# train time rate: 0.19    0.44     0.28    0.77       0.52     1.78
# test time rate:  0.17    0.42     0.32    0.86       0.53     0.73

args = parser.parse_args()

# device = torch.device("cuda:{}".format(args.gpu) if args.gpu is not None else "cpu")
#
# model.to(device)
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main(running_file):
    Cmd = 'python ' + ' '.join(sys.argv)
    running_file.write('\n%s\n' % Cmd)
    running_file.flush()

    global args

    # Refine args
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
    # 可复现自检
    # torch.use_deterministic_algorithms(True)
    # RuntimeError: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation,
    # but you set 'torch.use_deterministic_algorithms(True)'

    if args.lr_steps is not None and not isinstance(args.lr_steps, list):
        args.lr_steps = list(map(int, args.lr_steps.split('-')))

    print(args)
    ### Create model
    model = getattr(models, args.model)(args)

    print("##" * 20)
    total = get_model_parm_nums(model)
    print("Num of learnable paramter is: {}".format(total))
    # flops, params = get_flops_params(model, 224)

    model_dict = model.state_dict()
    if os.path.isfile(args.pretrain):
        ckpt = torch.load(args.pretrain)
        pretrained_dict = ckpt["state_dict"]

        for k, v in pretrained_dict.items():
            name = k.replace('module.', '')  # 去除 "module." 的 prefix
            if name in model_dict:
                model_dict[name] = v
                print("Sucessfully loaded pretrained model: {}".format(name))
        model.load_state_dict(model_dict)
    else:
        print("No avaliable pretrained model")
    print("##" * 20)

    ### Output its model size, flops and bops
    if args.checkinfo:
        count_paramsM = get_model_parm_nums(model)
        print('Model size: %f MB' % count_paramsM)
        print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
        return

    ### Define optimizer
    conv_weights, bn_weights, relu_weights = model.get_weights()
    param_groups = [{
        'params': conv_weights,
        'weight_decay': args.wd,
        'lr': args.lr}, {
        'params': bn_weights,
        'weight_decay': 0.1 * args.wd,
        'lr': args.lr}, {
        'params': relu_weights,
        'weight_decay': 0.0,
        'lr': args.lr
    }]
    info = ('conv weights: lr %.6f, wd %.6f' +
            '\tbn weights: lr %.6f, wd %.6f' +
            '\trelu weights: lr %.6f, wd %.6f') % \
           (args.lr, args.wd, args.lr, args.wd * 0.1, args.lr, 0.0)

    print(info)
    running_file.write('\n%s\n' % info)
    running_file.flush()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.99))
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise TypeError("Please use a correct optimizer in [adam, sgd]")

    ### Transfer to cuda devices
    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')
    # model = model.to(device)
    # cudnn.benchmark = True

    ### Load Data
    if args.dataset == 'BSDS':
        train_dataset = BSDS_Loader(split="train", threshold=args.eta, ablation=args.ablation)
        test_dataset = BSDS_Loader(split="test", threshold=args.eta)
    elif args.dataset == 'BSDS-PASCAL':
        train_dataset = BSDS_VOCLoader(split="train", threshold=args.eta, ablation=args.ablation)
        test_dataset = BSDS_VOCLoader(split="test", threshold=args.eta)
    elif 'Multicue' in args.dataset:
        train_dataset = Multicue_Loader(root=args.datadir, split="train", threshold=args.eta, setting=args.dataset[1:])
        test_dataset = Multicue_Loader(root=args.datadir, split="test", threshold=args.eta, setting=args.dataset[1:])
    elif 'NYUD' in args.dataset:
        train_dataset = NYUD_Loader(root=args.datadir, split="train", setting=args.dataset[1:])
        test_dataset = NYUD_Loader(root=args.datadir, split="test", setting=args.dataset[1:])

    elif 'BIPED' in args.dataset:
        train_dataset = BIPED_Loader(split="train")
        test_dataset = BIPED_Loader(split="test")

    elif 'BRIND' in args.dataset:
        train_dataset = BIPED_Loader(root="/workspace/00Dataset/BRIND", split="train")
        test_dataset = BIPED_Loader(root="/workspace/00Dataset/BRIND", split="test")
    elif 'UDED' in args.dataset:
        # train_dataset = BIPED_Loader(root='/workspace/00Dataset/UDED',split="train")
        test_dataset = BIPED_Loader(root='/workspace/00Dataset/UDED', split="test")

    elif 'Custom' in args.dataset:
        train_dataset = Custom_Loader(root=args.datadir)
        test_dataset = Custom_Loader(root=args.datadir)
    else:
        raise ValueError("unrecognized dataset setting")

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    ### Create log file
    log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)

    args.start_epoch = 0

    ### Evaluate directly if required
    if args.evaluate is not None:
        checkpoint = load_checkpoint(args.evaluate)

        args.start_epoch = checkpoint['epoch']
        if args.evaluate_converted and "v2" not in args.model:
            model.load_state_dict(convert_pidinet(checkpoint['state_dict'], args.config))
        elif args.evaluate_converted:
            model.load_state_dict(convert_pidinet_v2(checkpoint['state_dict'], args.config))
        else:
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                from collections import OrderedDict
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                model.load_state_dict(new_state_dict)
        testdir = os.path.join(args.savedir, 'eval_results_convert', "SS", str(args.start_epoch))
        if not os.path.isdir(testdir):
            test(test_loader, model, running_file, args, testdir)
            # multiscale_test(test_loader, model, args.start_epoch, running_file, args)
        else:
            print("{} is exist".format(testdir))
        print('##########Time########## %s' % (time.strftime('%Y-%m-%d %H:%M:%S')))
        return

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True)
    ### Optionally resume from a checkpoint
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                state_dict = {}
                for k in checkpoint['state_dict'].keys():
                    state_dict[k[7:]] = checkpoint['state_dict'][k]
                model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # args.epochs += args.start_epoch

            # args.lr_steps = [lrs + args.start_epoch for lrs in args.lr_steps]
            # print("lr step:{}".format(args.lr_steps))

    ### Train
    saveID = None

    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate
        lr_str = adjust_learning_rate(optimizer, epoch, args)

        # train
        tr_avg_loss = train(
            train_loader, model, optimizer, epoch, running_file, args, lr_str)

        log = "Epoch %03d/%03d: train-loss %s | lr %s | Time %s\n" % \
              (epoch, args.epochs, tr_avg_loss, lr_str, time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(log_file, 'a') as f:
            f.write(log)

        saveID = save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, args.savedir, saveID, keep_freq=args.save_freq)

        ## test cur epoch
        testdir = os.path.join(args.savedir, 'eval_results', "SS", str(epoch))
        test(test_loader, model, running_file, args, testdir)
        # multiscale_test(test_loader, model, epoch, running_file, args)

    return


def train(train_loader, model, optimizer, epoch, running_file, args, running_lr):
    show_dir = os.path.join(args.savedir, "train_shot", str(epoch))
    os.makedirs(show_dir, exist_ok=True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    weights = [1, 1, 1, 1, 1] if args.loss_weight == "same" else [0.2, 0.2, 0.2, 0.2, 1]


    ## Switch to train mode
    model.train()

    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    wD = len(str(len(train_loader) // args.iter_size))
    wE = len(str(args.epochs))

    end = time.time()
    iter_step = 0
    counter = 0
    loss_value = 0
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_loader):

        ## Measure data loading time
        data_time.update(time.time() - end)

        if args.gpu:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        ## Compute output
        outputs = model(image)
        if not isinstance(outputs, list):
            loss = cross_entropy_loss_RCF(outputs, label, args.lmbda)
        else:
            loss = 0

            for w, o in zip(weights, outputs):
                loss += w * cross_entropy_loss_RCF(o, label, args.lmbda)

        counter += 1
        loss_value += loss.item()
        loss = loss / args.iter_size
        loss.backward()
        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
            iter_step += 1

            # record loss
            losses.update(loss_value, args.iter_size)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_value = 0

            # display and logging
            if iter_step % args.print_freq == 1:
                outputs.append((label == 1).float())
                outputs = torch.cat(outputs, 0)
                torchvision.utils.save_image(1 - outputs, os.path.join(show_dir, "iter-%d.jpg" % i))
                runinfo = str(('Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}]\t' \
                               % (wE, wE, wD, wD) + \
                               'Time {batch_time.val:.3f}\t' + \
                               'Data {data_time.val:.3f}\t' + \
                               'Loss {loss.val:.4f} (avg:{loss.avg:.4f})\t' + \
                               'lr {lr}\t').format(
                    epoch, args.epochs, iter_step, len(train_loader) // args.iter_size,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, lr=running_lr))
                print(runinfo)
                running_file.write('%s\n' % runinfo)
                running_file.flush()

    str_loss = '%.4f' % (losses.avg)
    return str_loss


import time


def test(test_loader, model, running_file, args, testdir):
    from PIL import Image
    import scipy.io as sio
    model.eval()

    jpg_dir = os.path.join(testdir, 'jpg')
    png_dir = os.path.join(testdir, 'png')
    mat_dir = os.path.join(testdir, 'mat')

    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    eval_info = '\nBegin to eval...\nImg generated in %s\n' % png_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))

    times = 0
    for idx, (image, img_name) in enumerate(tqdm(test_loader)):

        img_name = img_name[0]
        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            start_time = time.perf_counter()
            results = model(image)
            end_time = time.perf_counter()
            times += (end_time - start_time)
            result = torch.squeeze(results[-1]).cpu().numpy()

        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]

        torchvision.utils.save_image(1 - results_all,
                                     os.path.join(jpg_dir, "%s.jpg" % img_name))
        sio.savemat(os.path.join(mat_dir, '%s.mat' % img_name), {'img': result})
        result = Image.fromarray((result * 255).astype(numpy.uint8))
        result.save(os.path.join(png_dir, "%s.png" % img_name))
        # runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        # # print(runinfo)
        # running_file.write('%s\n' % runinfo)

    running_file.write('\ntotal model time:{} total num:{} FPS:{}\n'
                       .format(times, len(test_loader), len(test_loader) / times))
    running_file.write('\nDone\n')


def multiscale_test(test_loader, model, epoch, running_file, args):
    from PIL import Image
    import scipy.io as sio
    model.eval()

    png_dir = os.path.join(args.savedir, 'eval_results', "MS", str(epoch), 'png')
    mat_dir = os.path.join(args.savedir, 'eval_results', "MS", str(epoch), 'mat')

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    eval_info = '\nBegin to eval...\nImg generated in %s\n' % png_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))

    for idx, (image, img_name) in enumerate(tqdm(test_loader)):
        img_name = img_name[0]

        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        scale = [0.5, 1, 1.5]
        _, H, W = image.shape
        multi_fuse = numpy.zeros((H, W), numpy.float32)

        with torch.no_grad():
            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))
                if args.use_cuda:
                    results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
                    result = torch.squeeze(results[-1].detach()).cpu().numpy()
                else:
                    results = model(torch.unsqueeze(torch.from_numpy(im_), 0))
                    result = torch.squeeze(results[-1].detach()).numpy()
                fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse
            multi_fuse = multi_fuse / len(scale)

        sio.savemat(os.path.join(mat_dir, '%s.mat' % img_name), {'img': multi_fuse})
        result = Image.fromarray((multi_fuse * 255).astype(numpy.uint8))
        result.save(os.path.join(png_dir, "%s.png" % img_name))
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        # print(runinfo)
        running_file.write('%s\n' % runinfo)
    running_file.write('\nDone\n')


if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' \
                                % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))
    with open(running_file, 'w') as f:
        main(f)
    print('done')
