# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import random
from glob import glob
import os.path as osp
from utils import frame_utils
from utils.augmentor_un import FlowAugmentor
import logging

def get_extention(data_on_cluster=False):
    if data_on_cluster:
        return '/path1/' #change data path.
    else:
        return '/path2/' #change data path.
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, show_extra_info=False, get_flow=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.show_extra_info = show_extra_info
        self.max_flow = None
        self.get_flow = get_flow

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_img_generic(self.image_list[index][0])
            img2 = frame_utils.read_img_generic(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        if len(self.flow_list)>0:
            if self.flow_list[index] is not None:
                flow, valid = frame_utils.read_flow_generic(self.flow_list[index])
            else:
                flow, valid = None, None
        else:
            flow, valid = None, None

        img1 = frame_utils.read_img_generic(self.image_list[index][0])
        img2 = frame_utils.read_img_generic(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if flow is not None:
            flow = np.array(flow).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            
            #assert np.all(valid), "ground-truth flow is sparse, but dense augmentor is used"
            img1, img2, img1_ph, img2_ph, full_imgs_dict  = self.augmentor(img1, img2)
            img1_ph = torch.from_numpy(img1_ph).permute(2, 0, 1).float()
            img2_ph = torch.from_numpy(img2_ph).permute(2, 0, 1).float()
        
        else:
            img1_ph, img2_ph, full_imgs_dict = None, None, None

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if flow is not None:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).to(dtype=torch.bool)
            

        if flow is not None:
            if not self.sparse:
                valid &= (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

            if self.max_flow is not None:
                valid &= (flow[0].abs() < self.max_flow) & (flow[1].abs() < self.max_flow)

            valid &= torch.isfinite(flow[0]) & torch.isfinite(flow[1])
            flow[:, ~(torch.isfinite(flow[0]) & torch.isfinite(flow[1]))] = 0.0

        if valid is not None:
            valid = valid.float()

        if self.get_flow:
            if self.show_extra_info:
                return img1, img2, flow, valid, self.extra_info[index]
            else:
                return img1, img2, flow, valid
        else:
            if self.show_extra_info:
                return img1, img2, self.extra_info[index], img1_ph, img2_ph, full_imgs_dict
            else:
                return img1, img2, img1_ph, img2_ph, full_imgs_dict



    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, data_on_cluster, aug_params=None, split='training', root='sintel', dstype='clean', show_extra_info=False, read_flow_gt=False):
        root = get_extention(data_on_cluster) + root
        super(MpiSintel, self).__init__(aug_params, show_extra_info=show_extra_info, get_flow=read_flow_gt)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)


        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id
            if read_flow_gt:
                if split != 'test':
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class FlyingChairs(FlowDataset):
    def __init__(self, data_on_cluster, aug_params=None, split='training', root='fc/data', read_flow_gt=False):
        root = get_extention(data_on_cluster) + root
        super(FlyingChairs, self).__init__(aug_params, get_flow=read_flow_gt)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))
       
        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.image_list += [ [images[2*i], images[2*i+1]] ]
                if read_flow_gt:
                    self.flow_list += [ flows[i] ]


class KITTI(FlowDataset):
    def __init__(self, data_on_cluster, aug_params=None, split='training', root='kitti15/dataset', read_flow_gt=False):
        root = get_extention(data_on_cluster) + root
        super(KITTI, self).__init__(aug_params, sparse=True, get_flow=read_flow_gt)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if read_flow_gt:
            if split == 'training':
                self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class KITTI_mv(FlowDataset):
    def __init__(self, data_on_cluster, aug_params=None, split='training', root='kitti15_mv'):
        root = get_extention(data_on_cluster) + root
        super(KITTI_mv, self).__init__(aug_params)

        image_root = osp.join(root, split, "image_2")
        for i in range(200):
            img_sequence = str(i).zfill(6)
            for j in range(20):
                if osp.isfile(osp.join(image_root, f'{img_sequence}_{str(j+1).zfill(2)}.png')):
                    self.image_list += [[osp.join(image_root, f'{img_sequence}_{str(j).zfill(2)}.png'), osp.join(image_root, f'{img_sequence}_{str(j+1).zfill(2)}.png')]]
                else:
                    continue

     


def fetch_dataloader(args, phase, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args["train"]["dataset"][phase] == 'chairs':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': 0.0, 'max_scale': 0.6, 'do_flip': True}
        train_dataset = FlyingChairs(args['data_on_cluster'], aug_params, split='training')
    

    elif args["train"]["dataset"][phase] == 'sintel_un':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(data_on_cluster=args['data_on_cluster'], aug_params=aug_params, split='test', dstype='clean')
        sintel_final = MpiSintel(data_on_cluster=args['data_on_cluster'], aug_params=aug_params, split='test', dstype='final')       

        train_dataset = sintel_clean + sintel_final
    

    elif args["train"]["dataset"][phase] == 'sintel_train_un':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(data_on_cluster=args['data_on_cluster'], aug_params=aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(data_on_cluster=args['data_on_cluster'], aug_params=aug_params, split='training', dstype='final')       

        train_dataset = sintel_clean + sintel_final

    elif args["train"]["dataset"][phase] == 'kitti_mv':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False, 'cut_top': args["train"]["cut_top"][phase]}
        train_dataset = KITTI_mv(args['data_on_cluster'], aug_params, split='testing')
    
    elif args["train"]["dataset"][phase] == 'kitti_mv_train':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI_mv(args['data_on_cluster'], aug_params, split='training')
    
   

    else:
        ds = args["train"]["dataset"][phase]
        raise ValueError(f"unsupported dataset type {ds}")

    train_loader = data.DataLoader(train_dataset, batch_size=args["train"]["batch_size"][phase], 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True, collate_fn=default_collate)

    logger = logging.getLogger("Muun_RAFT.train")
    logger.info('Training with %d image pairs' % len(train_dataset))
    return train_loader, len(train_dataset)


def default_collate(batch):
    """
    Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512

    """
    img1 = torch.stack([item[0] for item in batch])
    img2 = torch.stack([item[1] for item in batch])
    img1_ph = torch.stack([item[2] for item in batch])
    img2_ph = torch.stack([item[3] for item in batch])
    if batch[0][4] is None:
        full_img_dict = None
    else:
        full_img_dict = {}
        max_uncropped_img_h = max([item[4]["uncropped_size_hw"][0] for item in batch]) # for mixed set training, you get images with different aspect ratioes.
        max_uncropped_img_w = max([item[4]["uncropped_size_hw"][1] for item in batch])

        for key in ["offset_wh_uv", "uncropped_size_hw"]: 
            full_img_dict[key] = torch.as_tensor([item[4][key] for item in batch])
        
        for key in ["full_img1", "full_img2"]:
            full_img_dict[key] = torch.stack([F.pad(item[4][key], (0,max_uncropped_img_w - item[4]["uncropped_size_hw"][1], 0,max_uncropped_img_h - item[4]["uncropped_size_hw"][0])) for item in batch]) # Careful!!! Paddings are first reversed by pytorch and then applied: first w, h padding arguments but padds h,w accordingly. Pads the last n dims, if you give 2n arguments. 
        
        
    return img1, img2, img1_ph, img2_ph, full_img_dict