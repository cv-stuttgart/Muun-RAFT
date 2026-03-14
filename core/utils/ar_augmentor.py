import numpy as np

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as TT


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, ph_aug=False, center_crop=False, spatial_aug_prob=0.8, stretch_prob=0.8):
        
        # spatial augmentation params
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = stretch_prob
        self.max_stretch = 0.2
        self.photo_aug = ph_aug

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        if ph_aug:
            self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        else:
            self.photo_aug = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0)
        self.asymmetric_color_aug_prob = 0.4
        self.eraser_aug_prob = 0.5
        self.hor_flip = TT.RandomHorizontalFlip(1.0)
        self.ver_flip = TT.RandomVerticalFlip(1.0)

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        B, c, h, w = img1.shape
        img1, img2 = img1.type(torch.uint8), img2.type(torch.uint8)
        if np.random.rand() < self.asymmetric_color_aug_prob:
            
            img1 = self.photo_aug(img1).type(torch.float32)
            img2 = self.photo_aug(img2).type(torch.float32)

        # symmetric
        else:

            image_stack = torch.cat([img1, img2], dim=0)
        
            image_stack = self.photo_aug(image_stack).type(torch.float32)
            img1, img2 = torch.split(image_stack, B, dim=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[2:]
        if np.random.rand() < self.eraser_aug_prob:

            mean_color = img2.mean(dim=(2, 3))
           
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
        
                img2[:, :, y0:y0+dy, x0:x0+dx] = mean_color.unsqueeze(dim=2).unsqueeze(dim=2)

        return img1, img2

    def spatial_transform(self, img1, img2, flow, teacher_mask):
        # randomly sample scale
        ht, wd = img1.shape[2:]
        min_scale = np.maximum(
            (ht + 8) / float(ht), 
            (wd + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            int_scale_x, int_scale_y = int(scale_x*ht), int(scale_y*wd)

            img1 = TF.resize(img1, (int_scale_x, int_scale_y))
            img2 = TF.resize(img2, (int_scale_x, int_scale_y))
            flow = TF.resize(flow, (int_scale_x, int_scale_y))
            flow = torch.stack((flow[:,0,:,:]*int_scale_y/wd, flow[:,1,:,:]*int_scale_x/ht), dim=1) 
            
            if teacher_mask is not None:
                teacher_mask = TF.resize(teacher_mask, (int_scale_x, int_scale_y))

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
               
                img1 = self.hor_flip(img1)
                img2 = self.hor_flip(img2)
                flow = self.hor_flip(flow)
                flow = torch.stack((flow[:,0,:,:]*-1, flow[:,1,:,:]*1), dim=1)
                if teacher_mask is not None:
                    teacher_mask = self.hor_flip(teacher_mask)

            if np.random.rand() < self.v_flip_prob: # v-flip
               
                img1 = self.ver_flip(img1)
                img2 = self.ver_flip(img2)
                flow = self.ver_flip(flow)
                flow = torch.stack((flow[:,0,:,:]*1, flow[:,1,:,:]*-1), dim=1)
                if teacher_mask is not None:
                    teacher_mask = self.ver_flip(teacher_mask)

      
        if not self.center_crop:
            if (img1.shape[2] - self.crop_size[0]) == 0 and (img1.shape[3] - self.crop_size[1]) == 0:
                y0 = 0
                x0 = 0
            else:
                y0 = np.random.randint(0, img1.shape[2] - self.crop_size[0])
                x0 = np.random.randint(0, img1.shape[3] - self.crop_size[1])
        else:
            y0 = int((img1.shape[2] - self.crop_size[0])/2)
            x0 = int((img1.shape[3] - self.crop_size[1])/2)
        img1 = img1[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if teacher_mask is not None:
            teacher_mask = teacher_mask[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            return img1, img2, flow, teacher_mask
        else:
            return img1, img2, flow
       

    def __call__(self, img1, img2, flow, teacher_mask=None, step=0):
        
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        if teacher_mask is not None:
            img1, img2, flow, teacher_mask = self.spatial_transform(img1, img2, flow, teacher_mask=teacher_mask)
        else:
            img1, img2, flow = self.spatial_transform(img1, img2, flow, teacher_mask=teacher_mask)

        img1 = img1.contiguous()
        img2 = img2.contiguous()
        flow = flow.contiguous()
        if teacher_mask is not None:
            teacher_mask = teacher_mask.contiguous()
            return img1, img2, flow, teacher_mask
        else:
            return img1, img2, flow
    