'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, task="SR"):
    phase = dataset_opt['phase']
    task_opt = dataset_opt[task] if task in dataset_opt else dataset_opt
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert task_opt['batch_size'] % world_size == 0
            batch_size = task_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = task_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt, task="SR"):

    logger = logging.getLogger('base')
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'SSSR':
        from data.SSSR_dataset import SSSRDataset as D
        # TODO: bad deviation
        dataset = D(dataset_opt, task)
        logger.info('Dataset [{:s} - {:s}] is created. For task {:s}'.format(
            dataset.__class__.__name__, dataset_opt['name'], task))
        return dataset
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
