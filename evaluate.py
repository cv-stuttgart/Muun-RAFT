import sys
sys.path.append('core')

import argparse
import os
import os.path as osp
import numpy as np
import torch
import datasets_un
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from Muun_RAFT import Muun_RAFT

from config.config_loader import cpy_eval_args_to_config
from custom_logger import init_logger
import logging
from tqdm import tqdm    

@torch.no_grad()
def create_kitti_submission(model, data_on_cluster=False, iters=[8, 8, 4], padding_mode="kitti", output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets_un.KITTI(data_on_cluster, split='testing', aug_params=None)
    coarsest_scale = 16
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in tqdm(range(len(test_dataset))):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode=padding_mode, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1=image1, image2=image2, iters=iters, test_mode=True, bw=False)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

@torch.no_grad()
def create_sintel_submission(model, data_on_cluster=False, iters=[8, 8, 4], warm_start=True, output_path='sintel_submission', split= "test"):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    coarsest_scale = 16
   
    for dstype in ['clean', 'final']:
        test_dataset = datasets_un.MpiSintel(data_on_cluster, split=split, aug_params=None, dstype=dstype, show_extra_info=True)

        flow_prev, sequence_prev = None, None
        for test_id in tqdm(range(len(test_dataset))):
            if split == "test":
                image1, image2, (sequence, frame), _,_,_ = test_dataset[test_id]
            elif split == "training":
                image1, image2,_,_, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True, bw=False) #warm warm
            flow = torch.squeeze(padder.unpad(flow_pr[0]), dim=0).permute(1, 2, 0).cpu().numpy()
            
            if warm_start:
                
                flow_prev = forward_interpolate(flow_low[0])[None].cuda() #warm warm
                
  
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame_%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

@torch.no_grad()
def validate_sintel_during_training(model, data_on_cluster, warm= True, iters=[8, 8, 4]):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    coarsest_scale = 16
    for dstype in ['clean', 'final']:
        val_dataset = datasets_un.MpiSintel(data_on_cluster,split='training', dstype=dstype, show_extra_info = True, read_flow_gt=True)
        epe_list = []
        efel_list= []

        flow_prev, sequence_prev = None, None

        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1, image2)
           
            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True, bw=False)
            flow = padder.unpad(flow_pr[0]).cpu()

            if warm:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            #****
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            efel = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            efel_list.append(efel.cpu().numpy())
            #******
            sequence_prev = sequence

        efel_list = np.concatenate(efel_list)
        FL = 100 * np.mean(efel_list)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation WARM: %s (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, FL_error: %f" % (str(warm), dstype, epe, px1, px3, px5, FL))
        if warm:
            results['warm' + dstype + f' {sum(iters)}'] = np.mean(epe_list)
            results['warm' + dstype + 'FL_error' + f' {sum(iters)}' ] = FL
        else:
            results[dstype + f' {sum(iters)}'] = np.mean(epe_list)
            results[dstype + 'FL_error'+ f' {sum(iters)}'] = FL

    return results



@torch.no_grad()
def validate_sintel(model, model_path, data_on_cluster=False, warm=False, iters=[8, 8, 4]):
    """ Evaluate trained model on Sintel(train) clean + final passes on train split """
    logger = logging.getLogger('eval.sintel')
    logger_for_eval_all = logging.getLogger('eval_all.sintel')
    model.eval()
    results = {}
    coarsest_scale = 16
    
    epe_sequence={}
    
    for dstype in ['clean', 'final']:

        val_dataset = datasets_un.MpiSintel(data_on_cluster, split='training', dstype=dstype, show_extra_info = True, read_flow_gt=True)
        epe_list = []
        efel_list = []
        flow_prev, sequence_prev = None, None
        sequences = []


        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
            frame += 1

            if sequence not in sequences:
                sequences.append(sequence)

            if f"{sequence}_epe" not in epe_sequence:
                epe_sequence[f"{sequence}_epe"] = 0
                epe_sequence[f"{sequence}_count"] = 0

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            if sequence != sequence_prev:
                flow_prev = None
            if not warm:
                flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True, bw=False)
                flow = padder.unpad(flow_pr[0]).cpu()
            elif warm:
                flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True, bw=False)
                flow = padder.unpad(flow_pr[0]).cpu()
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            #*****
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            efel = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            efel_list.append(efel.cpu().numpy())
            #******

            sequence_prev = sequence

            epe_sequence[f"{sequence}_epe"] += epe.mean()
            epe_sequence[f"{sequence}_count"] += 1

        for seq in sequences:
            epe_per_seq = epe_sequence[f"{seq}_epe"]/epe_sequence[f"{seq}_count"]
            logger.info("Sintel validation, iters: %d, type: %s warm: %r   seq:(%s)  EPE: %f" % (sum(iters), dstype, warm, seq, epe_per_seq)) 
                
        efel_list = np.concatenate(efel_list)
        FL = 100 * np.mean(efel_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        logger.info("Sintel validation iters:%d warm: %r (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, FL-Error: %f:" % (sum(iters), warm, dstype, epe, px1, px3, px5, FL))
        logger_for_eval_all.info("Sintel iters:%d validation of the model: %s ,warm: %r, : (%s) EPE: %f , FL-Error: %f:" % (sum(iters), model_path, warm, dstype, epe, FL))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_chairs(model, data_on_cluster=False, iters=[4,4,4]):
    """ Perform evaluation on the FlyingChairs (validation) split """
    model.eval()
    epe_list = []

    val_dataset = datasets_un.FlyingChairs(data_on_cluster, split='validation', read_flow_gt=True)
    
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True, bw=False)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs iters:%d EPE: %f" %(sum(iters), epe))
    return {'chairs': epe}



@torch.no_grad()
def validate_kitti(model, data_on_cluster=False, iters=[8, 8, 4], padding="kitti"):
    """ Peform validation using the KITTI-2015 (train) split """
    logger = logging.getLogger("eval.kitti")
    model.eval()
    val_dataset = datasets_un.KITTI(data_on_cluster, split='training', read_flow_gt=True)
    out_list, epe_list = [], []
    coarsest_scale = 16
   
    
    for data_id in tqdm(range(len(val_dataset))):

        image1, image2, flow_gt, valid_gt = val_dataset[data_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, mode=padding, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)
        
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True, bw=False)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    logger.warning("Kitti iters:%d validation:%f, %f" % (sum(iters), epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1", "True")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--warm', action = 'store_true', help="use warm start", default = True)
    parser.add_argument('--iters', type=int, nargs='+', default=[4,4,4])
    parser.add_argument('--mixed_precision', help='use mixed precision', type = str2bool, default = True)
    parser.add_argument('--data_on_cluster', action = 'store_true', help="adjust the address according to the cluster.", default = False)
    parser.add_argument('--cuda_corr', type = str2bool, default = True)
    parser.add_argument('--img1_path', default = "none")
    parser.add_argument('--img2_path', default = "none")
    parser.add_argument('--out_path', default = "none")
   
    parser.add_argument('--fnet_norm', default = "group")
    parser.add_argument('--cnet_norm', default = "group")
    parser.add_argument('--bw', action = 'store_true', default = False)

    args = parser.parse_args()
    config = cpy_eval_args_to_config(args)
    print(config)
    model = torch.nn.DataParallel(Muun_RAFT(config))
    model.load_state_dict(torch.load(config["model"]))
    

    model.cuda()
    model.eval()
    path = os.path.join(os.path.dirname((config["model"])), "eval.log")
    logger = init_logger("eval", path, stream_level= logging.WARNING)
    logger = init_logger("eval_all", "./eval_all.log")
    print("**********MIXED:", config['mixed_precision'])
    print('WARM: ', config["warm"])
    
    with torch.no_grad():
        if config["dataset"] == 'chairs':
            validate_chairs(model.module, data_on_cluster=config["data_on_cluster"], iters = [4,4,4])

        elif config["dataset"] == 'sintel':
            validate_sintel(model.module, config["model"], config["data_on_cluster"], warm=config["warm"], iters = [8,8,4])
     

        elif config["dataset"] == 'sintel_test':
            if config["out_path"] != "none":
                model_name = os.path.basename(os.path.dirname(config["model"]))
                out_path = osp.join(config["out_path"], model_name, f'sintel_test_iters{sum([8,8,4])}_mixed{config["mixed_precision"]}')
                print(out_path)
                create_sintel_submission(model, data_on_cluster= config["data_on_cluster"] , iters=[8,8,4], warm_start=True, output_path=out_path)
            else:
                print(osp.join(os.path.dirname(config["model"]), f'sintel_test_iters{sum([8,8,4])}_mixed{config["mixed_precision"]}'))
                out_path = osp.join(os.path.dirname(config["model"]), f'sintel_test_iters{sum([8,8,4])}_mixed{config["mixed_precision"]}')
                create_sintel_submission(model, data_on_cluster= config["data_on_cluster"] , iters= [8,8,4], warm_start=True, output_path=out_path)

        
        elif config["dataset"] == 'kitti':
            validate_kitti(model.module, data_on_cluster=config["data_on_cluster"], iters=[8,8,4])

        
        elif config["dataset"] == "kitti_test":
            create_kitti_submission(model, output_path=osp.join(os.path.dirname(config["model"]), 'kitti_test', 'flow'), data_on_cluster=config["data_on_cluster"], iters=[8,8,4])

    
