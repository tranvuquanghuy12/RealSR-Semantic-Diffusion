# TODO:working out logging message and frequency
# TODO:switching tasks per iteration or per epoch? reflect? easier to log?
# TODO:logging val results
# TODO:augmentationg for semantic segmentation and validation on val split
import os, os.path as osp
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

from time import time
import numpy as np
from utils.util import AverageMeter, get_confusion_matrix
import lpips
from piq import ssim

fn_alex = lpips.LPIPS(net="alex")

from utils.mylogger import wandb_init, parser_add_wandb_args, wandb, logger as mylogger


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser_add_wandb_args(parser)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    wandb_init(args, opt)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    # torch.backends.cudnn.benckmark = True
    # # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    train_loader = {}
    val_loader = {}
    # nepochs = 0
    for task in ['SR', 'SS']:
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'train':
                # print('\n\n\n\n\n\n\n\n', dataset_opt)
                train_set = create_dataset(dataset_opt, task)
                task_opt = dataset_opt[task] if task in dataset_opt else dataset_opt
                train_size = int(math.ceil(len(train_set) / task_opt['batch_size']))
                total_iters = int(opt['train']['niter'])
                total_epochs = int(math.ceil(total_iters / train_size))
                if task == "SR":
                    nepochs = int(math.ceil(total_iters / train_size))
                if opt['dist']:
                    train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                    total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                else:
                    train_sampler = None
                train_loader[task] = create_dataloader(train_set, dataset_opt,
                                                       opt, train_sampler, task)
                if rank <= 0:
                    mylogger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                        len(train_set), train_size))
                    mylogger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                        total_epochs, total_iters))

                # nepochs = total_epochs if total_epochs > nepochs else nepochs
            elif phase == 'val':
                # if task == "SR":
                #     continue
                val_set = create_dataset(dataset_opt, task)
                val_loader[task] = create_dataloader(val_set, dataset_opt, opt,
                                                     None, task)
                if rank <= 0:
                    mylogger.info('Number of val images in [{:s}]: {:d}'.format(
                        dataset_opt['name'], len(val_set)))
            else:
                raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    # assert train_loader is not None
    assert len(train_loader) > 0

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        mylogger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    mylogger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    avg_RS = 0
    avg_SS = 0
    best_mIoU = 0.0
    for epoch in range(start_epoch, nepochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        # Training Super-Resolution
        val_SR = val_loader['SR'] if 'SR' in val_loader else None
        print ("-"*10)
        avg_RS, logs_step = train_SR(model, train_loader['SR'], val_SR,
                                     total_iters, epoch, avg_RS,
                                     logger, rank, opt)
        # model.update_ema(0.95)
        # validation
        if epoch % 5 == 0 and rank <= 0 and val_SR is not None:
            validate_SR(val_SR, model, opt, epoch)

        # Training Semantic Segmentation
        print ("-"*10)
        val_SS = val_loader['SS'] if 'SS' in val_loader else None
        avg_SS, logs_step = train_SS(model, train_loader['SS'],
                                     total_iters, epoch, avg_SS, logger,
                                     rank, opt)
        # model.update_ema(0.9998)
        # validation
        if epoch % 5 == 0 and rank <= 0 and val_SS is not None:
            _, mean_IoU = validate_SS(model, val_loader['SS'], epoch, logger, rank, opt)
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                logger.info(f'Saving BEST model with mIoU {best_mIoU:.4f}')
                model.save('best', mylogger)
                model.save_training_state(epoch, epoch * len(train_loader['SR']), save_name='best')

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')

    # clean up media logs on wandb server
    print ("Cleaning up WANDB server. Removing media files: ")
    api = wandb.Api()
    run = api.run(wandb.run.path)
    cnter = 0
    for f in run.files():
        # media/images/Train | Super-Resolution_0_352e...png
        if ("Train" in f.name or "Super-Resolution" in f.name) and ".png" in f.name:
            f.delete()
            cnter += 1
    print (f"Cleaning up DONE. REMOVED {cnter} files")

def log_semseg(data, model, epoch, title, wandb_msk=None, do_log=True):
    from data.SSSR_dataset import SSSRDataset
    cl = SSSRDataset.label_name()
    # table = wandb.Table(columns=["ID", "input", "SS"])
    wandb_msk = [] if wandb_msk is None else wandb_msk
    for i in range(data['img'].size(0)):
        # input = data['img'][i].permute((1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
        input = util.tensor2img(data['img'][i].float().cpu())
        pred = model.fake_H['SS'] if isinstance(model.fake_H, dict) else model.msk_P
        wandb_msk.append(wandb.Image(
            input,
            masks = {
                "prediction": {
                    "mask_data": pred[i].argmax(0).detach().cpu().numpy(),
                    "class_labels": cl},
                "ground truth": {
                    "mask_data": data['msk'][i].detach().cpu().numpy().astype(np.uint8),
                    "class_labels": cl},
            }
        ))
        # table.add_data(i, wandb.Image(input), wandb_msk)

    if do_log:
        wandb.log({title: wandb_msk}, step=epoch)
    else:
        return wandb_msk

def log_supres(train_data, model, epoch, title, wandb_img=None, do_log=True):

    # should be declared once and had rows added every time
    # table = wandb.Table(columns=["ID", "input", "pred", "target"])
    wandb_img = [] if wandb_img is None else wandb_img
    for i in range(min(3, train_data['LQ'].size(0))):

        # for visualization purpose, use nearest
        inp = F.interpolate(model.var_L.detach(), scale_factor=4)
        inp = util.tensor2img(inp[i].float().cpu())
        prd = util.tensor2img(model.fake_H.detach()[i].float().cpu())
        tgt = util.tensor2img(model.var_H.detach()[i].float().cpu())

        wandb_img.append(wandb.Image(
            np.vstack((inp, prd, tgt)),
            caption="input-prediction-target"
        ))
        # table.add_data(i, wandb.Image(inp), wandb.Image(prd), wandb.Image(tgt))

    if do_log:
        wandb.log({title: wandb_img}, step=epoch)
    else:
        return wandb_img

def train_SS(model, train_loader, total_iters, epoch,
             avg, logger, rank, opt):

    begin = time()
    logs_step = []
    for iter, train_data in enumerate(train_loader):
        current_step = epoch * len(train_loader) + iter
        # if current_step % opt['logger']['print_freq'] == 0:
        #     logger.info(f'<epoch:{epoch:3d}, iter:{current_step:8,d}> ')
        # if epoch < 87:
        #     continue

        if current_step > total_iters:
            break

        #### training
        model.feed_data(train_data, task="SS")
        model.optimize_parameters(current_step, task="SS")

        avg += time() - begin
        begin = time()

        #### log
        if current_step % opt['logger']['print_freq'] == 0:
            logs = model.get_current_log()
            logs_step.append(logs)
            eta = avg / (current_step+1) * (total_iters-current_step) / 3600.
            message = f'<y>SS</y> ep:{epoch:3,d} it:{current_step:8,d} eta:{eta:>5.1f}h '
            for k, v in logs.items():
                message += '{:s}: {:.2e} '.format(k, v)
            if rank <= 0:
                mylogger.opt(colors=True).info(message)

        #### save models and training states
        if current_step % opt['logger']['save_checkpoint_freq'] == 0 and epoch > 5:
            if rank <= 0:
                logger.info('Saving models and training states (LATEST).')
                model.save('latest', mylogger)
                model.save_training_state(epoch, current_step, save_name='latest')

        #### update learning rate
        model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

    # return 0, None
    if epoch % 10 == 0 and opt.get('use_multitask', True):
        log_semseg(train_data, model, epoch, title= "Train | Semantic Segmentation")

    if len(logs_step) > 0:
        avg_log = {k: np.average([i[k] for i in logs_step]) for k in logs}
        wandb.log(avg_log, step=epoch)

    return avg, logs_step

def validate_SS(model, val_loader, epoch, logger, rank, config):
    """
    for distributed implementation with rank, see ref:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1/lib/core/function.py
    """
    mylogger.info("Validating Semantic Segmentation...")

    model.reset_confusion_matrix()
    idx = 0
    nlogs = 5
    wandb_msk = []
    with torch.no_grad():
        for _, val_data in enumerate(val_loader):
            idx += 1
            model.feed_data(val_data, task="SS")
            model.test()
            model.update_confusion_matrix()

            if idx < nlogs:
                wandb_msk = log_semseg(val_data, model, epoch,
                                       "Val | Semantic Segmentation",
                                       wandb_msk, do_log=False)

    log_semseg(val_data, model, epoch, "Val | Semantic Segmentation",
               wandb_msk, do_log=True)

    IoU_array, mean_IoU = model.summary_confusion_matrix()
    print ('valid_mIoU', mean_IoU)
    wandb.log({"mIOU/Val": mean_IoU}, step=epoch)

    return IoU_array, mean_IoU

def train_SR(model, train_loader, val_loader,
             total_iters, epoch, avg, logger, rank, opt):
    begin = time()
    logs_step = []
    for iter, train_data in enumerate(train_loader):
        # current_step += 1
        current_step = epoch * len(train_loader) + iter

        # #### log
        # if current_step % opt['logger']['print_freq'] == 0:
        #     logger.info(f'<epoch:{epoch:3d}, iter:{current_step:8,d}> ')
        # if epoch < 87:
        #     continue

        if current_step > total_iters:
            # print ("BREAK: ", current_step, epoch, len(train_loader), total_iters)
            break

        #### update learning rate
        model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

        #### training
        model.feed_data(train_data)
        model.optimize_parameters(current_step)

        # if epoch > 1:
        #     model.update_ema(.95)

        avg += time() - begin
        begin = time()

        #### log
        if current_step % opt['logger']['print_freq'] == 0:
            logs = model.get_current_log()
            logs_step.append(logs)
            eta = avg / (current_step+1) * (total_iters-current_step) / 3600.
            message = f'<g>SR</g> ep:{epoch:3,d} it:{current_step:8,d} eta:{eta:>5.1f}h '
            for k, v in logs.items():
                message += '{:s}: {:.2e} '.format(k, v)
            if rank <= 0:
                mylogger.opt(colors=True).info(message)

        #### save models and training states
        if current_step % opt['logger']['save_checkpoint_freq'] == 0 and epoch > 5:
            if rank <= 0:
                logger.info('Saving models and training states (LATEST).')
                model.save('latest', mylogger)
                model.save_training_state(epoch, current_step, save_name='latest')

            # print ("Log supres")
    else:
        # if epoch < 87:
        #     return 0, None

        if epoch % 10 == 0:
            if isinstance(model.fake_H, dict):
                model.fake_H = model.fake_H['SR']
            log_supres(train_data, model, epoch, title="Train | Super-Resolution")

        if len(logs_step) > 0:
            avg_log = {k: np.average([i[k] for i in logs_step if k in i]) for k in logs}
            wandb.log(avg_log, step=epoch)

    return avg, logs_step

def validate_SR(val_loader, model, opt, epoch, save_frq=10):
    mylogger.info("Validating Super-Resolution...")

    avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0
    avg_ssim = avg_lpips = 0.0
    idx = 0
    nlogs = 5
    wandb_img = []
    for val_data in val_loader:
        idx += 1
        img_name = osp.splitext(osp.basename(val_data['LQ_path'][0]))[0]
        img_dir = osp.join(opt['path']['val_images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        gt_img = util.tensor2img(visuals['GT'])  # uint8

        # Save SR images for reference, not all images
        if idx % save_frq == 0:
            save_img_path = osp.join(img_dir, f'{img_name:s}_{epoch:03d}.png')
            util.save_img(sr_img, save_img_path)

        avg_ssim += ssim(visuals['SR'].unsqueeze(0),
                         visuals['GT'].unsqueeze(0)).item()
        sr = (visuals['SR'].clamp_(0, 1) - 0.5) * 2
        gt = (visuals['GT'].clamp_(0, 1) - 0.5) * 2
        avg_lpips += fn_alex(sr, gt).item()

        # calculate PSNR
        crop_size = opt['scale']
        gt_img = gt_img / 255.
        sr_img = sr_img / 255.
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

        if idx < nlogs:
            model.fake_H = model.fake_H['SR'] if isinstance(model.fake_H, dict) else model.fake_H
            wandb_img = log_supres(val_data, model, epoch, "Val | Super-Resolution",
                       wandb_img, do_log=False)

    model.fake_H = model.fake_H['SR'] if isinstance(model.fake_H, dict) else model.fake_H
    log_supres(val_data, model, epoch, "Val | Super-Resolution",
                       wandb_img, do_log=True)

    mylogger.info(f"Generated imgs: \n {osp.dirname(save_img_path)}")

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_lpips = avg_lpips / idx
    val_pix_err_f /= idx
    val_pix_err_nf /= idx
    val_mean_color_err /= idx

    # log
    mylogger.info('# Validation PSNR:^{:.4e} | SSIM:^{:.4e} | LPIPS:v{:.4e}'.format(
        avg_psnr, avg_ssim, avg_lpips))
    wandb.log({"PSNR/Val": avg_psnr, "SSIM/Val": avg_ssim, "LPIPS/Val":
               avg_lpips}, step=epoch)

if __name__ == '__main__':
    main()
