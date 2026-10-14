import os.path as osp
import sys
import os
import wandb
from loguru import logger
fmt = "<green>{time:MM-DD HH:mm:ss}</green> | "\
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "\
        "<level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=fmt)

def __init_wandb(args):
    if wandb.run.dir is not None and osp.exists(wandb.run.dir):
        args.save_folder = wandb.run.dir
    wandb.config.update(args)

def parser_add_wandb_args(parser=None):

    from datetime import datetime as dt
    expid = dt.now().strftime("%m%d.%H%M")

    if parser is None:
        import argparse
        parser = argparse.ArgumentParser(description='Pytorch Training')
    parser.add_argument('--group', default=expid, type=str)
    parser.add_argument('--job_type', default="default", type=str)
    parser.add_argument('--save_folder', default='weights/')
    parser.add_argument('--note', default='', type=str)

    return parser

def wandb_init(args, opt):

    wandb.init(job_type=(None if args.job_type == "" else args.job_type),
               group=(None if args.group=="" else args.group))

    root = opt['path']['experiments_root']
    # new_root = f"{old_root}-{wandb.run.name}-{wandb.run.id}"
    # new_root = osp.join(osp.abspath(f"{wandb.run.dir}/.."), osp.basename(root))
    new_root = osp.join(osp.abspath(f"{wandb.run.dir}/.."), "local_files")

    for key in opt['path']:
        if not isinstance(opt['path'][key], str):
            continue
        opt['path'][key] = opt['path'][key].replace(root, new_root)
        if key not in ['resume_state', 'pretrain_model_G', 'root', 'strict_load']:
            os.makedirs(opt['path'][key], exist_ok=True)

    wandb.config.update({"run_path": osp.abspath(f"{wandb.run.dir}/..")})
    wandb.config.update(opt)
