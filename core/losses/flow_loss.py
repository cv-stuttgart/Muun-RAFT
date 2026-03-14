import torch.nn as nn
import torch
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import  get_guassian_consistency_mask

MAX_FLOW = 400


class unFlowLoss(nn.modules.Module):
    def __init__(self, config, phase):
        super(unFlowLoss, self).__init__()
        self.config = config
        self.gamma = self.config["gamma"][phase]
        self.ar_gamma = self.config["ar_gamma"][phase]
        self.sm_gamma = self.config["sm_gamma"][phase]
        self.sm_order = self.config["sm_order"][phase]
        self.sm_weight = self.config["sm_weight"][phase]
        self.ph_l1_weight = self.config["ph_l1_weight"][phase]
        self.ph_ssim_weight = self.config["ph_ssim_weight"][phase]
        self.ph_ternary_weight = self.config["ph_ternary_weight"][phase]
        self.edge_sensitivity = self.config["edge_sense"]
        self.sm_img_range = self.config["sm_img_range"]
        self.ph_img_range = self.config["ph_img_range"]
        self.flow_loss_type = self.config["flow_loss_type"][phase]
        self.ternary_patch_size = self.config["ternary_patch_size"]

     

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []
        l1_ssim_loss=0

        if self.ph_l1_weight > 0:
            loss += [self.ph_l1_weight * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.ph_ssim_weight > 0:
            loss += [self.ph_ssim_weight * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]
        l1_ssim_loss = sum([l.mean() for l in loss]) / (occu_mask1.mean() + 0.00001)

        if self.ph_ternary_weight > 0:
            loss_ph = self.ph_ternary_weight * TernaryLoss(im1_recons,
                                                      im1_scaled, occ=occu_mask1, patch_size=self.ternary_patch_size)
            return loss_ph + l1_ssim_loss
        else:
            return l1_ssim_loss
        

    def loss_smooth(self, flow, im1_scaled, edge_sensitivity):
        if self.sm_order==2:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, alpha=edge_sensitivity)] # before an alpha was given, but it was set to 10 in all settings.
        return sum([l.mean() for l in loss])

        
    def forward(self, flow12_list, flow21_list, occ12_list, occ21_list, img1, img2, flow_predictions_of_aug_imgs, flow_gt, teacher_student_masking=False, teacher_mask=None, flow_loss_current_weight=0):
        
        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(flow12_list)    
        batch_size = img1.shape[0]
        img1_sm, img1_ph, img2_sm, img2_ph = img1, img1, img2, img2 # it is now 0-255

        loss = 0
        L_ph = 0
        L_sm = 0
        L_ph_fw_list =[]

        img1_sm = img1_sm/255
        img2_sm = img2_sm/255

        img1_ph = img1_ph/255
        img2_ph = img2_ph/255
       
            
        if flow_loss_current_weight > 0.0:
            flow_loss = 0.0
            mag = torch.sum(flow_gt**2, dim=1).sqrt()
            valid = (mag < MAX_FLOW)

        #global counter
        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            ar_i_weight = self.ar_gamma**(n_predictions - i - 1)
            sm_i_weight = self.sm_gamma**(n_predictions - i - 1)
            
            im1_recons = flow_warp(img2_ph.detach(), flow12_list[i])
            im2_recons = flow_warp(img1_ph.detach(), flow21_list[i])

            L_ph_fw = self.loss_photomatric(img1_ph, im1_recons, occu_mask1=occ12_list[i])
            L_ph_fw_list.append(L_ph_fw)
            
            L_sm_fw = self.loss_smooth(flow12_list[i], img1_sm, edge_sensitivity=self.edge_sensitivity)
            L_ph += L_ph_fw * i_weight
            L_sm += L_sm_fw * sm_i_weight * self.sm_weight

            #bw # I just add them. I don't compute avg. Not (fw + bw)/2
            L_ph_bw = self.loss_photomatric(img2_ph, im2_recons, occu_mask1=occ21_list[i])
            L_sm_bw = self.loss_smooth(flow21_list[i], img2_sm, edge_sensitivity=self.edge_sensitivity)
            L_ph += L_ph_bw * i_weight
            L_sm += L_sm_bw * sm_i_weight * self.sm_weight

            if flow_loss_current_weight > 0.0:
                # compute validity mask of flow_predictions_of_aug_imgs:
                if teacher_student_masking:
                    student_fw_invalid_mask = 1 - get_guassian_consistency_mask(flow=flow_predictions_of_aug_imgs[i][0:batch_size], flow_bw=flow_predictions_of_aug_imgs[i][batch_size:, :, :, :], sigma=0.03)
                    student_bw_invalid_mask = 1 - get_guassian_consistency_mask(flow=flow_predictions_of_aug_imgs[i][batch_size:], flow_bw=flow_predictions_of_aug_imgs[i][0:batch_size, :, : ,:], sigma=0.03)
                    student_fw_bw_invalid_mask = torch.cat((student_fw_invalid_mask, student_bw_invalid_mask), dim=0)

                    teacher_student_mask = student_fw_bw_invalid_mask.detach() * teacher_mask.detach()
                    
                else:
                    teacher_student_mask = torch.ones((img1.shape[0]*2, 1, img1.shape[2], img1.shape[3])).cuda()
                
                #####flow error:
                i_loss = (flow_predictions_of_aug_imgs[i] - flow_gt.detach()).abs()

                num_valid_pixels = torch.sum(valid[:, None])
                num_all_pixels = torch.sum(torch.ones_like(valid, dtype=float))
                valid_ratio = num_valid_pixels / num_all_pixels
            
                if self.flow_loss_type == "L2":
                    l2_norm = torch.sum((valid[:, None] *i_loss)**2 + 0.000001, dim=1).sqrt() # valid epe, size: (b, h, w) # Considering validity improves the stability for training on KITTI!
                    l2_loss = (l2_norm * teacher_student_mask).mean() / valid_ratio 
            
                    flow_loss += ar_i_weight * l2_loss

                else:
                    raise ValueError(f'No loss with name "{self.flow_loss_type}" is implemented.')
                
                
        if flow_loss_current_weight > 0.0:        
            epe = torch.sum((flow_predictions_of_aug_imgs[-1] - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid.view(-1)].mean()
        else:
            flow_loss = torch.tensor(0).to(img1.device)
            epe = torch.tensor(0).to(img1.device)

        flow_loss = flow_loss * flow_loss_current_weight   
        loss += L_ph + L_sm + flow_loss
        return (loss*batch_size).unsqueeze(dim=0), (L_ph*batch_size).unsqueeze(dim=0), (L_sm*batch_size).unsqueeze(dim=0), (flow_loss*batch_size).unsqueeze(dim=0), (epe*batch_size).unsqueeze(dim=0) # to handle proper mean over all GPUS using 