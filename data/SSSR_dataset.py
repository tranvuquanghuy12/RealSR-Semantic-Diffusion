import random
import os.path as osp
import numpy as np
from scipy.io import loadmat
import cv2
import glob
from scipy.spatial.distance import cdist
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import data.util as util
from data.data_loader import noiseDataset
from data.LQGT_dataset import LQGTDataset
from PIL import Image
from preprocess.utils import imresize as preprocess_imresize


import time


from utils.mylogger import logger

# class SSSRDataset(LQGTDataset):
class SSSRDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, task="SR"):
        """
            task: ["SR", "SS"] for partially training task
        """
        super(SSSRDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        # assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]
        # print(opt['aug'])
        self.noises = None
        if self.opt['phase'] == 'train':
            if opt['aug'] and 'noise' in opt['aug']:
                if osp.exists(opt['SR']['noise_data']):
                    self.noises = noiseDataset(opt['SR']['noise_data'],
                                               opt['GT_size']/opt['scale'])
                    if len(self.noises) == 0:
                        logger.warning("EMPTY noise list: "+opt['SR']['noise_data'])
                elif "Oracle" in opt['SR']['noise_data']:
                    self.noises = lambda x: torch.from_numpy(
                        np.random.normal(0, 4, x).astype(np.float32)/255.)
                else:
                    logger.warning("Unknown noise or file NOT exists: "+opt['SR']['noise_data'])

        # semantic segmentation
        self.task = task
        if task == "SS":
            self.semseg_lst = open(self.opt[task]['semseg_lst']).read().splitlines()
            self.semseg_lst *= 1
            print (f"SS task: {len(self.semseg_lst)}")
            self.aux_lst = open(self.opt['SR']['supres_lst']).read().splitlines()
        if task == "SR":
            self.supres_lst = open(self.opt[task]['supres_lst']).read().splitlines()
            print (f"SR task: {len(self.supres_lst)}")
            self.aux_lst = open(self.opt['SS']['semseg_lst']).read().splitlines()

            self.paths_LQ, self.paths_GT = zip(*[l.split(' ') for l in self.supres_lst])

            print (self.opt['SR']['noise_data'])
            if self.opt['phase'] == "train" and self.opt['SR']['degrade'] is \
                not None and 'Bicubic' in self.opt['SR']['degrade'] :
                self.paths_GT = self.paths_LQ
                self.paths_LQ = None

    def __getitem__(self, index):

        # TODO: LR for SR and LR for SS should be of the same size

        if self.task == "SR":
            return self.getitem_SR(index)
            # print (out['LQ'].shape, out['GT'].shape)
        elif self.task == "SS":
            return self.getitem_SS(index)

    def getitem_SS(self, index):
        line = self.semseg_lst[index].split(" ")
        img = Image.open(line[0])
        seg = Image.open(line[1])
        aux = Image.open(self.aux_lst[index % len(self.aux_lst)].split(" ")[0])

        if self.opt['phase'] == 'train':
            if self.opt['use_rot']:
                angle = np.random.randint(0, 359)
                angle = angle // 5 * 5
                # use NEAREST resample to not introduce smoothing effect
                # expand = False (keep image size), fillcolor=(0, 0, 0) in both
                img = img.rotate(angle, resample=Image.NEAREST)
                seg = seg.rotate(angle, resample=Image.NEAREST)
                aux = aux.rotate(angle, resample=Image.NEAREST)

        img = np.array(img).astype(np.float32) / 255. # after utils.py/read_img
        seg = np.array(seg)
        aux = np.array(aux).astype(np.float32) / 255. # after utils.py/read_img

        if self.opt['phase'] == 'train':

            # # crop-and-paste augmentation
            # ind = np.random.randint(0, len(self.semseg_lst)-1)
            # ln = self.semseg_lst[ind].split(" ")
            # im_ = np.array(Image.open(ln[0])).astype(np.float32) / 255.
            # sg_ = np.array(Image.open(ln[1]))
            # h, w = int(im_.shape[0]/2), int(im_.shape[1]/2)
            # h = np.random.randint(int(h/2), h) # random cropped shape from h/4-h/2
            # w = np.random.randint(int(w/2), w) # random cropped shape from w/4-w/2
            # # choose random crop location
            # yc = np.random.randint(0, im_.shape[0] - h)
            # xc = np.random.randint(0, im_.shape[1] - w)
            # # choose random paste location
            # yp = np.random.randint(0, img.shape[0] - h)
            # xp = np.random.randint(0, img.shape[1] - w)
            # img[yp:yp+h, xp:xp+w, :] = im_[yc:yc+h, xc:xc+w, :]
            # seg[yp:yp+h, xp:xp+w, :] = sg_[yc:yc+h, xc:xc+w, :]
            # # to test
            # # Image.fromarray(np.hstack((img, seg))).save(f"temp/{index}.png")

            # # random crop
            # H, W, _ = img.shape
            # LQ_size = self.opt['GT_size'] // self.opt['scale']

            # rnd_h = random.randint(0, max(0, H - LQ_size))
            # rnd_w = random.randint(0, max(0, W - LQ_size))
            # img = img[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            # seg = seg[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # rnd_h = rnd_h % (aux.shape[0] - LQ_size)
            # rnd_w = rnd_w % (aux.shape[1] - LQ_size)
            # aux = aux[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # augmentation flip and rotation
            img, seg = util.augment([img, seg], self.opt['use_flip'], self.opt['use_rot'])

        # convert seg RGB to class labels
        seg = SSSRDataset.color_to_class(seg) # seg has only 2 dimensions/axes H,W

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        seg = torch.from_numpy(np.ascontiguousarray(seg)).long()
        aux = torch.from_numpy(np.ascontiguousarray(np.transpose(aux, (2, 0, 1)))).float()

        return {'img': img, 'msk': seg, 'img_path': line[0], 'seg_path':
                line[1]}#, 'aux': aux}

    def getitem_SR(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = None
        assert  osp.exists(GT_path), "Not exists: " + GT_path
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)
        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        AU_path = self.aux_lst[index % len(self.aux_lst)].split(" ")[0]
        img_AU = util.read_img(self.LQ_env, AU_path, None)
        # get LQ image
        if self.paths_LQ and self.paths_LQ[index] != self.paths_GT[index]:
            LQ_path = self.paths_LQ[index]
            assert  osp.exists(LQ_path), "Not exists: " + LQ_path
            resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # img_AU = img_AU[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition
            # img_AU = util.channel_convert(img_AU.shape[2], self.opt['color'],
            #                               [img_AU])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            # img_AU = img_AU[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        # img_AU = torch.from_numpy(np.ascontiguousarray(np.transpose(img_AU, (2, 0, 1)))).float()

        if self.opt['phase'] == 'train':
            # add noise to LR during train
            # if self.opt['aug'] and 'noise' in self.opt['aug']:
            if self.noises is not None:
                noise = self.noises(img_LQ.shape) if callable(self.noises) else\
                        self.noises[np.random.randint(0, len(self.noises))]
                img_LQ = torch.clamp(img_LQ + noise, 0, 1)
                # Image.fromarray((img_LQ.permute((1, 2,
                # 0)).numpy()*255).astype(np.uint8)).save(f"temp/{index}.png")

        if LQ_path is None:
            LQ_path = GT_path

        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path':
                GT_path}#, 'aux': img_AU}

    @staticmethod
    def label_palette():
        """
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        # ISPRS color palette
        palette = {
            0: (255, 255, 255),    # Impervious Surface
            1: (  0,   0, 255),    # Building
            2: (  0, 255, 255),    # Low Vegetation
            3: (  0, 255,   0),    # Tree
            4: (255, 255,   0),    # Car
            5: (255,   0,   0),    # Background
            6: (  0,   0,   0),    # Void
        }
        return palette

    @staticmethod
    def label_name():
        """
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        return {
            0: 'Surface',
            1: 'Building',
            2: 'LowVeg',
            3: 'Tree',
            4: 'Car',
            5: 'Background',
            6: 'Void',
        }

    @staticmethod
    def invert_palette():
        return {v: k for k, v in SSSRDataset.label_palette().items()}

    @staticmethod
    def class_to_color(arr_2d):
        """ Numeric labels to RGB-color encoding
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        palette=SSSRDataset.label_palette()
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d

    @staticmethod
    def color_to_class(arr_3d):
        """ RGB-color encoding to grayscale labels using Nearest Neighbor
        to handle JPEG compression artifacts in Kaggle datasets.
        """
        palette = SSSRDataset.invert_palette()
        colors = np.array(list(palette.keys())) # Shape: (7, 3)
        classes = np.array(list(palette.values())) # Shape: (7,)
        
        flat_img = arr_3d.reshape(-1, 3)
        dists = cdist(flat_img, colors)
        closest = np.argmin(dists, axis=1)
        
        arr_2d = classes[closest]
        arr_2d = arr_2d.reshape(arr_3d.shape[:2]).astype(np.uint8)

        return arr_2d

    def save_pred(self, preds, sv_path, name):
        import os.path as osp
        palette = SSSRDataset.label_palette()
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.putpalette(palette)
            save_img.save(osp.join(sv_path, name[i]+'.png'))

    def __len__(self):
        if self.task == "SS":
            return len(self.semseg_lst)
        else: # self.task == "SR"
            # return len(self.supres_lst)
            return len(self.paths_GT)

