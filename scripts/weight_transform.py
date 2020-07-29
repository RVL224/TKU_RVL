import sys
import os
import argparse
import numpy as np
import pickle

import torch 

# lib
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer

@torch.no_grad()
def check_model(cfg, ckpt):
    # init device
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))
    model.eval()

    count = 0
    for key, value in model.state_dict().items():
        
        # print(key,value.shape)

        if(key.find("num_batches_tracked") != -1):
            pass
        else:
            # print(key)
            print(key,value.shape)
            count += 1
    
    print("total layer {}".format(count))

@torch.no_grad()
def save_layer_param(cfg, ckpt, save_path, depth_wise, layer_names):
    # init device
    # device = torch.device(cfg.MODEL.DEVICE)
    device = torch.device("cpu")

    model = build_detection_model(cfg)
    model = model.to(device)
    model.eval()

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    # read param
    # print("\n\n")
    # for parameter in model.named_parameters():
    #     # for ind, num in enumerate(parameter[1]):
        
    #     print(parameter[1].shape)
    #     weight = np.transpose(parameter[1].detach().numpy(),(2,3,1,0))
    #     print(weight)

    #     sys.exit()

    # write
    if(save_path is not None):
        weight_dict = {}
        

        with open(save_path,"wb") as outfile:
            count = 0
            for key, value in model.state_dict().items():
                if(key.find("num_batches_tracked") != -1 or 
                   key.find("conv.0.bias") != -1):
                    continue
                index = layer_names.index(key)

                if(len(value.shape) == 4):
                    if(index in set(depth_wise)):
                        weight = np.transpose(value.detach().numpy(),(2,3,0,1))
                        print(index,1,key,value.shape,weight.shape)
                    else:
                        weight = np.transpose(value.detach().numpy(),(2,3,1,0))
                        print(index,2,key,value.shape,weight.shape)
                else:
                    weight = value.detach().numpy()
                    print(index, 3, key, value.shape, weight.shape)

                weight_dict[key] = weight
                count += 1

            print("total layer : {}".format(count))
            pickle.dump(weight_dict, outfile)
            print("save weight : {}".format(save_path))
        
        # count = 1
        # for key, value in model.state_dict().items():
        #     if(key.find("num_batches_tracked") != -1):
        #         continue
        #     print(count, key)
        #     if(count == 1):
        #         # depthwise
        #         # print(np.transpose(value.detach().numpy(),(2,3,0,1)).shape)
        #         # print(key)
        #         # print(np.transpose(value.detach().numpy(),(2,3,0,1)))
        #         # conv2d
        #         # print(np.transpose(value.detach().numpy(),(2,3,1,0)).shape)
        #         print(key)
        #         print(np.transpose(value.detach().numpy(),(2,3,1,0)))
        #         # bias
        #         # print(value.detach().numpy().shape)
        #         # print(key)
        #         # print(value.detach().numpy())
        #         break
        #     count += 1 


    print("finish")

def check_save_weight(save_path):
    with open(save_path, "rb") as loadfile:
        weight_dict = pickle.load(loadfile)
        print(weight_dict.keys())
        print(len(weight_dict.keys()))

def read_layer_name_from_file(filepath):
    with open(filepath, "r") as loadfile:
        layer_names = []
        for ind, line in enumerate(loadfile.readlines()):
            line = line.strip()
            name = line.split(" ")[0]
            layer_names.append(name)

        return layer_names

def get_depth_wise(save_path):
    with open(save_path, "r") as loadfile:
        depth_wise = []
        for ind, line in enumerate(loadfile.readlines()):
            if(line.find("depthwise_weights") != -1):
                depth_wise.append(int(ind))
        return depth_wise

def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")

    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument("--check_torch", type=bool, default=False, help="check pytorch model")
    parser.add_argument("--layer_name_torch", type=str, default="", help="pytorch model custom layer names")
    parser.add_argument("--layer_name_tf", type=str, default="", help="tensorflow model custom layer names")
    parser.add_argument("--save_path", type=str, default=None, help="Trained weights.")
    
    args = parser.parse_args()
    print(args)

    # config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
    
    
    if(args.check_torch):
        # check torch model layer
        check_model(cfg,args.ckpt)
    else:
        # read layer name for tensorflow version (manual adjust)
        layer_names = read_layer_name_from_file(args.layer_name_torch)
        
        # read depthwise order from tensorflow model
        depth_wise = get_depth_wise(args.layer_name_tf)
        
        # save param
        save_layer_param(cfg, args.ckpt, args.save_path, depth_wise, layer_names)
        
    # check_save_weight(args.save_path)


if __name__ == "__main__":
    main()