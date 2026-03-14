import json


mandatory = {}
class DefaultSetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __setitem__(self, key, value):
        if key not in self.dictionary:
            if value is mandatory:
                raise ValueError(f" Argument --> {key} was mandatory but is not there")
            else:
                self.dictionary[key] = value

    def __getitem__(self, key):
        if key not in self.dictionary:
            self.dictionary[key] = {}
        return DefaultSetter(self.dictionary[key])

def load_json_config(config_path):

    file = open(config_path) 
    config = json.load(file)

    default_setter = DefaultSetter(config)
    default_setter["name"] = mandatory
    default_setter["train"]["lr"] = mandatory
    default_setter["train"]["dataset"] = mandatory
    default_setter["train"]["num_steps"] = mandatory
    default_setter["train"]["batch_size"] = mandatory
    default_setter["train"]["image_size"] = mandatory
    default_setter["train"]["validation"] = mandatory
    default_setter["train"]["restore_ckpt"] = None
    default_setter["train"]["optim"] = "AdamW"
    default_setter["train"]["iters"] = [4, 4, 4]
    default_setter["train"]["eval_iters"] = mandatory 
    default_setter["train"]["loss"]["gamma"] = mandatory
    default_setter["train"]["loss"]["ar_gamma"] = [0.8, 0.8]
    default_setter["train"]["loss"]["sm_gamma"] = [config["train"]["loss"]["gamma"][0], config["train"]["loss"]["gamma"][1]]
    default_setter["train"]["loss"]["flow_loss_type"] = ["L2", "L2"]
    default_setter["train"]["loss"]["ar"] = False
    default_setter["train"]["loss"]["ar_start"] = [40, 0] #percent!! of the phase's iters.
    default_setter["train"]["loss"]["ar_increasing"] = 10 #percent!! of next training iters, increase it to X value.
    default_setter["train"]["loss"]["ar_weight"] = [1.2, 1.2]
    default_setter["train"]["loss"]["aug_settings"] = [{'crop_size':config["train"]["image_size"][0], 'min_scale': 0.0, 'max_scale': 1.0, 'do_flip': True}, {'crop_size':config["train"]["image_size"][1], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}]

    default_setter["train"]["loss"]["sm_weight"] = mandatory
    default_setter["train"]["loss"]["sm_order"] = mandatory
    default_setter["train"]["loss"]["ph_l1_weight"] = mandatory
    default_setter["train"]["loss"]["ph_ssim_weight"] = mandatory
    default_setter["train"]["loss"]["ph_ternary_weight"] = mandatory
    default_setter["train"]["loss"]["edge_sense"] = 150
    default_setter["train"]["loss"]["sm_img_range"] = "0_1"
    default_setter["train"]["loss"]["ph_img_range"] = "0_1"
    default_setter["train"]["loss"]["ternary_patch_size"]= 7

    default_setter["train"]["cut_top"]= [False, False, False]

    default_setter["train"]["wdecay"] = mandatory
    default_setter["activate_lookup"] = False
    default_setter["lr_peak"] = 0.05
    default_setter["mixed_precision"] = False
    default_setter["gpus"] = [0]
    default_setter["epsilon"] = 1e-8
    default_setter["add_noise"] = False
    default_setter["clip"] = 1.0
    default_setter["dropout"] = 0.0
    default_setter["data_on_cluster"] = False
    default_setter["current_phase"] = 0
    default_setter["current_steps"] = -1
    default_setter["fnet_norm"] = "group"
    default_setter["cnet_norm"] = "group"
    default_setter["grad_acc"] = [1, 1, 1, 1]
    default_setter["cuda_corr"] = True

    
    default_setter["bw"] = True
    default_setter["mask_out"] = True
    default_setter["occ_method"] = ["wan", "wan"]
    default_setter["detach"] = True 
    default_setter["orig_image_for_Lph_lsm"] = True
    default_setter["teacher_student_masking"] = True
    default_setter["only_teacher_masking"] = False

    return config


def cpy_args_to_config(args):
    config = {}
    config["name"] = args.name
    config["epsilon"] = args.epsilon
    config["clip"] = args.clip
    config["dropout"] = args.dropout
    config["small"] = args.small
    config["gpus"] = args.gpus
    config["add_noise"] = args.add_noise
    config["mixed_precision"] = args.mixed_precision
    config["lr_peak"] = args.lr_peak

    config["train"] = {}
    config["train"]["gamma"] = [args.gamma]
    config["train"]["wdecay"] = [args.wdecay]
    config["train"]["validation"] = [args.validation]
    config["train"]["num_steps"] = [args.num_steps]
    config["train"]["lr"] = [args.lr]
    config["train"]["loss"] = [args.loss]
    config["train"]["image_size"] = [args.image_size]
    config["train"]["batch_size"] = [args.batch_size]
    config["train"]["dataset"] = [args.dataset]
    config["train"]["restore_ckpt"] = args.restore_ckpt
    config["train"]["iters"] = args.iters
    config["freeze_encoder"] = args.freeze_encoder
    config["current_phase"] = args.current_phase
    config["current_steps"] = args.current_steps

    return config

def cpy_eval_args_to_config(args):
    config = {}
    config["model"] = args.model
    config["warm"] = args.warm
    config["iters"] = args.iters
    config["dataset"] = args.dataset
    config["mixed_precision"] = args.mixed_precision
    config["data_on_cluster"] = args.data_on_cluster
    config["cuda_corr"] = args.cuda_corr
    
    config["fnet_norm"] = args.fnet_norm
    config["cnet_norm"] = args.cnet_norm

    config["img1_path"] = args.img1_path
    config["img2_path"] = args.img2_path
    config["out_path"] = args.out_path
    config["bw"] = args.bw
    return config
