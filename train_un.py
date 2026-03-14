from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Muun_RAFT import Muun_RAFT
import evaluate
import datasets_un
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward, get_guassian_consistency_mask
from losses.flow_loss import unFlowLoss
import logging
from config.config_loader import load_json_config, cpy_args_to_config
import json
from datetime import datetime
from utils.ar_augmentor import FlowAugmentor
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    print("NOOOOO GRADSCALE!!EXITING.....")
    exit(0)

SUM_FREQ = 50
VAL_FREQ = 1000

           
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(config, phase, model, local_step):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr= config["train"]["lr"][phase], weight_decay= config["train"]["wdecay"][phase], eps= config["epsilon"])
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, config["train"]["lr"][phase], config["train"]["num_steps"][phase]+100,
    pct_start=config["lr_peak"], cycle_momentum=False, anneal_strategy='linear')

    for i in range(local_step + 1):
        scheduler.step()
    return optimizer, scheduler
    

class StatsLogger:
    def __init__(self, name, current_steps, phase):
        self.total_steps = current_steps
        self.phase = phase
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir= os.path.join("checkpoints", name))
        self.metrics_file = os.path.join("checkpoints", name, "lrs.csv")
        self.time = datetime.now()
        self.logger = logging.getLogger("Muun_RAFT.stats") # this is a logger of type "Muun_RAFT". It can be more strict.
        
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as file:
                    file.write("step,lr\n") 

    def set_phase(self, phase, dataset):
        self.phase = phase
        self.dataset_being_trained = dataset

    def _print_training_status(self, lr):
        
        now = datetime.now()
        time_diff = now - self.time
        self.time = now
        training_str = "[number of steps: {0:6d}, lr: {1:2.7f}, dataset: {2}, phase: {3}, duration: {4:4.2f}, time:{5}] ".format(self.total_steps+1, lr, self.dataset_being_trained, self.phase, time_diff.total_seconds(), now)
        metrics_str = ",".join(f"{key}:{(value/SUM_FREQ):8.4f} "for key, value in self.running_loss.items())
        self.logger.info("%s %s", metrics_str, training_str)
        
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            

    def push(self, metrics, lr):
        self.total_steps += 1 # assume local step starts from -1, as it actually does.
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += torch.tensor(metrics[key]).item()

        with open(self.metrics_file, "a") as file:
            file.write("{:6d},{:10.7f}\n".format(self.total_steps, lr)) 

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status(lr)
            self.running_loss = {}

    def write_dict(self, results):

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def save_model_and_checkpoint(model, config, steps, phase, saving_policy = "limited"):
    if saving_policy == "unlimited":
        checkpoint_values_path = 'checkpoints/%s/_%s_phase%d_%d.pth' % (config["name"], config["train"]["dataset"][phase], phase, steps)
        torch.save(model.state_dict(), checkpoint_values_path)
    elif saving_policy == "limited":
        checkpoint_values_path = 'checkpoints/%s/%s_phase%d_%d.pth' % (config["name"], config["train"]["dataset"][phase], phase, steps)
        torch.save(model.state_dict(), checkpoint_values_path)
        checkpoint_txt_path = 'checkpoints/%s/checkpoint.txt' % config["name"]
        create_checkpoint_file(checkpoint_txt_path, phase, steps, checkpoint_values_path, config) 
    else:
        assert ValueError("Wrong saving policy given.")

def create_checkpoint_file(txtfile_path, phase, current_steps, checkpoint_name, config):
    if not os.path.exists(txtfile_path):
            
        with open(txtfile_path, 'w') as file:
            dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": None}
            json.dump(dict, file)
          
    else:
        with open(txtfile_path) as file:
            checkpoint_config = json.load(file)

        with open(txtfile_path, "w") as file:

            if checkpoint_config["newer"] == None:
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": None}
                json.dump(dict, file)
            elif (checkpoint_config["newer"] != None) and (checkpoint_config["older"] == None):
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": checkpoint_config["newer"]}
                json.dump(dict, file)
            else:
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": checkpoint_config["newer"]}
                json.dump(dict, file)
                # remove the older file:
                name = config["name"]
                older_file_path = checkpoint_config["older"]
                file_path_to_be_removed = older_file_path
                if os.path.exists(file_path_to_be_removed):
                    os.remove(file_path_to_be_removed)
                else:
                    logger = logging.getLogger("Muun_RAFT.saving")
                    logger.error("Checkpoint file did not exist. old checkpoint.txt: %s, new checkpoint.txt: %s", str(checkpoint_config), str(dict))
        
def fetch_model(config, phase):
    model = nn.DataParallel(Muun_RAFT(config), device_ids=config["gpus"])
    print("Parameter Count: %d" % count_parameters(model))           
    model.cuda() 
    model.train()

    return model

def fetch_data(config, phase):
    data_loader, _ = datasets_un.fetch_dataloader(config, phase)
    while True:
        for data_blob in data_loader:
            yield [x for x in data_blob]

def passed_steps(config, phase):
    steps = 0 
    if phase != 0:
        steps = sum(config["train"]["num_steps"][:phase])
   
    return steps

def training_step(config, model, data_group, optimizer, phase, scaler, loss_module, cur_step,
                   ar_augmentor, flow_loss_current_weight):

    iterations = config["train"]["iters"]
    for data in data_group:
        
        image1, image2, img1_ph, img2_ph, full_img_dict = data
        image1, image2, img1_ph, img2_ph = image1.cuda(), image2.cuda(), img1_ph.cuda(), img2_ph.cuda()
      
        flow_predictions, flow_bw_predictions = model(image1=img1_ph, image2=img2_ph, iters=iterations, bw=config["bw"])

        if config["occ_method"][phase] == "brox":
            occlusions = get_occu_mask_bidirection(flow_predictions,flow_bw_predictions)
            occlusions_bw = get_occu_mask_bidirection(flow_bw_predictions,flow_predictions)
        elif config["occ_method"][phase] == "wan":
            occlusions = get_occu_mask_backward(flow_predictions,flow_bw_predictions)
            occlusions_bw = get_occu_mask_backward(flow_bw_predictions,flow_predictions)
        
        if config["train"]["loss"]["ar"]:
            with torch.no_grad():
                flow_predictions_teacher, flow_bw_predictions_teacher = model(image1=image1, image2=image2, iters=iterations, bw=config["bw"], test_mode=True)
            if config["teacher_student_masking"]:
                teacher_mask_fw = get_guassian_consistency_mask(flow=flow_predictions_teacher[-1], flow_bw=flow_bw_predictions_teacher[-1], sigma= 0.003) #consider only the last flow!
                teacher_mask_bw = get_guassian_consistency_mask(flow=flow_bw_predictions_teacher[-1], flow_bw=flow_predictions_teacher[-1], sigma= 0.003) #consider only the last flow!
                img1_img2_aug, img2_img1_aug, flow_fw_flow_bw_truth, aug_teacher_mask = ar_augmentor(torch.cat((img1_ph, img2_ph), dim=0), torch.cat((img2_ph, img1_ph), dim=0), torch.cat((flow_predictions_teacher[-1], flow_bw_predictions_teacher[-1]), dim=0), step=cur_step, teacher_mask = torch.cat((teacher_mask_fw, teacher_mask_bw), dim=0))
               
            else:
                
                img1_img2_aug, img2_img1_aug, flow_fw_flow_bw_truth = ar_augmentor(torch.cat((img1_ph, img2_ph), dim=0), torch.cat((img2_ph, img1_ph), dim=0), torch.cat((flow_predictions_teacher[-1], flow_bw_predictions_teacher[-1]), dim=0), step=cur_step, teacher_mask = None)
                aug_teacher_mask = None
            flow_predictions_of_aug_imgs = model(image1=img1_img2_aug, image2=img2_img1_aug, iters=iterations, bw=False)

        else:
            flow_predictions_of_aug_imgs, flow_fw_flow_bw_truth = None, None
        
        batch_size = image1.shape[0]
        metrics = loss_module(flow12_list = flow_predictions, flow21_list=flow_bw_predictions, occ12_list=occlusions, occ21_list=occlusions_bw, img1=image1, img2=image2, flow_predictions_of_aug_imgs=flow_predictions_of_aug_imgs, flow_gt=flow_fw_flow_bw_truth, teacher_student_masking = config["teacher_student_masking"], teacher_mask=aug_teacher_mask, flow_loss_current_weight=flow_loss_current_weight)
        
        metrics = { k: v.sum() / batch_size for k, v in zip(["total_loss", "L_ph", "L_sm", "flow_loss", "epe_to_semi_gt"], metrics)}
        metrics["flow_loss_weight"] = flow_loss_current_weight
  
        scaler.scale(metrics["total_loss"]).backward()
    # ##after processing the "complete batch" --> update parameters:
    scaler.unscale_(optimizer)               
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
    scaler.step(optimizer)
    scaler.update()
    
    return metrics#, image1[0,:,:,:]

def current_flow_loss_weight(iter, activation_start_iter, last_increasing_iter, ar_ultimate_wight):
    output_weight = 0
    if iter < activation_start_iter:
        return output_weight
    elif iter >= activation_start_iter and iter < last_increasing_iter:
        output_weight = (iter - activation_start_iter) * ar_ultimate_wight/(last_increasing_iter - activation_start_iter)
    else:
        output_weight = ar_ultimate_wight
    return output_weight
    

def train_single_phase(model, data_generator, optimizer, scheduler, init_local_step, num_steps, stats_logger, training_step_fn, save_fn, phase):
    
    eval_iters_list = config["train"]["eval_iters"][phase]
    if config["train"]["loss"]["ar"]:
        ar_aug_settings = config["train"]["loss"]["aug_settings"][phase]
        ar_augmentor = FlowAugmentor(**ar_aug_settings)
        activation_start_iter = (config["train"]["loss"]["ar_start"][phase]/100.)*num_steps 
        last_increasing_iter = activation_start_iter + (config["train"]["loss"]["ar_increasing"]/100.)*num_steps 
        print(f"Phase {phase} and ar loss is activated!!")
    else:
        ar_augmentor = None
  
    for local_step in range(init_local_step, num_steps - 1):
    
        data_group = []
        optimizer.zero_grad()
        data = next(data_generator)
        data_group.append(data)

        if config["train"]["loss"]["ar"]:
            flow_loss_current_weight = current_flow_loss_weight(iter = local_step+1, activation_start_iter=activation_start_iter, last_increasing_iter=last_increasing_iter, ar_ultimate_wight=config["train"]["loss"]["ar_weight"][phase])
        else:
            flow_loss_current_weight = 0
        
        metrics = training_step_fn(model, data_group, optimizer, phase, cur_step=local_step, ar_augmentor=ar_augmentor, flow_loss_current_weight=flow_loss_current_weight)
        
        scheduler.step()

        stats_logger.push(metrics, scheduler.get_last_lr()[0])

        if (local_step + 1) % VAL_FREQ == VAL_FREQ - 1: #save checkpoint now
            if config["train"]["loss"]["ar"]:
                if (local_step + 1) == (config["train"]["loss"]["ar_start"][phase]/100) * config["train"]["num_steps"][phase] - 1:
                    save_fn(model, local_step + 2, "unlimited")
                else:
                    save_fn(model, local_step + 2, "limited") 
            else:
                save_fn(model, local_step + 2, "limited")
            results = {}

            evals = config["train"]["validation"][phase]
            if (isinstance(evals, str)):
                evals = [evals,]
            if (isinstance(eval_iters_list, int)):
                eval_iters_list = [eval_iters_list,]
            
            assert len(evals) == len(eval_iters_list), 'List of eval datasets and eval iters should have the same length.'

            for index, ev in enumerate(evals):
                if ev == 'chairs':
                    results.update(evaluate.validate_chairs(model.module, config["data_on_cluster"], iters = eval_iters_list[index]))
                elif ev == 'sintel':
                    results.update(evaluate.validate_sintel_during_training(model.module, config["data_on_cluster"], iters = eval_iters_list[index], warm=True))
                elif ev == 'sintel_test_relative':
                    results.update(evaluate.validate_sintel_test_relative(model.module, config["data_on_cluster"], iters = eval_iters_list[index], warm=True))
                elif ev == 'kitti':
                    results.update(evaluate.validate_kitti(model.module, config["data_on_cluster"], iters = eval_iters_list[index]))
                else:
                    raise ValueError(f"invalid evaluation set '{ev}'")

            stats_logger.write_dict(results)
            model.train()

           


def train_phases(init_phase, num_phases, local_step_of_init_phase, stats_logger, state_dict, datasets, num_steps, model_fn, data_fn, optimizer_scheduler_fn, training_step_fn, save_fn):
    local_step = local_step_of_init_phase
    for phase in range(init_phase, num_phases):
        torch.manual_seed(1234)
        np.random.seed(1234)
        stats_logger.set_phase(phase, datasets[phase])
        model = model_fn(phase)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        
        data = data_fn(phase)
        optimizer, scheduler = optimizer_scheduler_fn(phase, model, local_step)
        train_single_phase(model, data, optimizer, scheduler, local_step, num_steps[phase], stats_logger, training_step_fn, lambda model, step, policy: save_fn(model, step, phase, policy), phase)

        save_fn(model, num_steps[phase], phase, "unlimited")

        local_step = -1
        state_dict = model.state_dict()


def train(config):
    
    logger = logging.getLogger("Muun_RAFT.train")
    
    if config["train"]["restore_ckpt"] is None:
        possible_checkpoint_file = os.path.join( "checkpoints", config["name"], "checkpoint.txt")
        if (os.path.exists(possible_checkpoint_file)):
            file = open(possible_checkpoint_file)
            checkpoint_configs = json.load(file)
            config["current_phase"] = checkpoint_configs["phase"]
            config["train"]["restore_ckpt"] = checkpoint_configs["newer"]
            config["current_steps"] = checkpoint_configs["current_steps"] - 1 # local step is the index of the steps.         
    
    logger.info(config)

    if config["train"]["restore_ckpt"] is not None:
        state_dict = torch.load(config["train"]["restore_ckpt"])
        print("Loading checkpoint from %s....." %config["train"]["restore_ckpt"])
    else:
        state_dict = None

    total_phase_len = len(config["train"]["num_steps"])
    init_phase = config["current_phase"]
    local_steps = config["current_steps"] # local step is the last step (current step) in the current phase.

    passed_train_steps = passed_steps(config, init_phase)   
    stats_logger = StatsLogger(config["name"], local_steps + passed_train_steps, init_phase)
    scaler = GradScaler(enabled=config["mixed_precision"])
    current_phase = config["current_phase"]
    loss_module = nn.DataParallel(unFlowLoss(config["train"]["loss"], phase=current_phase), device_ids=config["gpus"])
    train_phases(init_phase, total_phase_len, local_steps, stats_logger, state_dict, config["train"]["dataset"], config["train"]["num_steps"],
    lambda phase: fetch_model(config, phase),
    lambda phase: fetch_data(config, phase),
    lambda phase, model, local_step: fetch_optimizer(config, phase, model, local_step),
    lambda model, data, optimizer, phase, cur_step, ar_augmentor, flow_loss_current_weight: training_step(config, model, data, optimizer, phase, scaler, loss_module, cur_step, ar_augmentor, flow_loss_current_weight),
    lambda model, step, phase, policy: save_model_and_checkpoint(model, config, step, phase, policy))

    stats_logger.close()
    
    print("--------------Reached the end of training. Exiting....")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--dataset', help="which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')#

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--lr_peak', type=float, default=0.05)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--loss', default="L1")
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])#
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')#

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')#
    parser.add_argument('--add_noise', action='store_true')#
    
    parser.add_argument('--current_phase', type=int, default=0)
    parser.add_argument('--current_steps', type=int, default= -1)
    parser.add_argument('--config', help= 'path to the configuration file')

    parser.add_argument('--cuda_corr', action='store_true', default=False)

    args = parser.parse_args()
    if args.config is not None:
        config = load_json_config(args.config) 
    else:
        config = cpy_args_to_config(args)
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000" #uncomment in case of memory fragmentation
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["NCCL_P2P_DISABLE"] = "1"


    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir(f'checkpoints/{config["name"]}'):
        os.mkdir(f'checkpoints/{config["name"]}')

    filehandler = logging.FileHandler(f"checkpoints/{config['name']}/log.txt")
    filehandler.setLevel(logging.INFO)

    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)

    logger = logging.getLogger("Muun_RAFT")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info("starting to train")

    train(config)
