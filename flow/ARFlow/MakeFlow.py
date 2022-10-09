import json
import pprint
import datetime
import argparse
from path import Path
from easydict import EasyDict

import torch
from utils.torch_utils import init_seed
from flow.ARFlow.utils.flow_utils import flow_to_image
from flow.ARFlow.datasets.get_dataset import get_dataset
from flow.ARFlow.models.get_model import get_model


#import basic_train
from logger import init_logger
import warnings
warnings.filterwarnings("ignore")
from flow.ARFlow.utils.torch_utils import bias_parameters, weight_parameters, load_checkpoint, save_checkpoint, AdamW



def initModel(cfg, pretrained = ""):
    model = get_model(cfg_flow.model)
    # print(model)
    model.init_weights()
    modelASD = torch.nn.DataParallel(model, device_ids=[0])
    return modelASD




if __name__ == '__main__':
    flow_cfg_path = "D:/BTH/EXJOBB/ColabServers/DTVNet/flow/ARFlow/configs/sky.json"
    cfg_flow = EasyDict(json.load(open(flow_cfg_path)))
    print("potato")
    output_path = "D:/BTH/EXJOBB/ColabServers/DTVNet/data/sky_timelapse/flow/"
    model = initModel(cfg_flow)


    train_set, valid_set = get_dataset(cfg_flow)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg_flow.train.batch_size,
        num_workers=cfg_flow.train.workers, pin_memory=True, shuffle=False)
    device = torch.device("cuda:0")
    print(len(train_loader))

    for i_step, data in enumerate(train_loader):
        #print(i_step, data.keys(), data["path"])
        print(i_step)
        img1, img2 = data['img1'], data['img2']
        img_pair = torch.cat([img1, img2], 1).to(device)
        print("   ", img_pair.size())
        flows = model(img_pair)['flows_fw']
        pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])
        print(len(pred_flows))
        print(flow_to_image(flows))


        if i_step > 1:
            break

    #for x in range(3):
        #data = train_set.__getitem__(x)
        #print(x, data.keys())
        #print("\t", print(data["path"]))
        #img1, img2 = data['img1'], data['img2']
        #img_pair = torch.cat([img1, img2], 1).to(device)
        #print(img_pair)
        #print(img1.size())
        #flow = model(img_pair)["flows_fw"]
        #print(flow)




