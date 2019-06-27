# Python 3.6.8
# numpy 1.16.2
# pytorch 1.1.0

# Author: Liu Tianchi (NUS-HLT)
# Date: March 2019
# Last Modify: June 2019

import torch
import torch.optim as optim
import numpy as np
import random
from config import parse_args
from Data_loader import data_packed
from Print_info import print_info
from trainer import Trainer
from Pre_check import pre_check
from scores_eer import scores
from model import lstm
from Compute_EER import compute_EER

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main():

    # args from config
    args = parse_args()

    # pre-check
    saving_path = pre_check(args)

    # load and pre-process data
    data_pack = data_packed()
    data_pack.data_process(args)

    # print running information
    print_info(saving_path, args, data_pack)


    # model
    model = lstm(data_pack, args)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
    # if args.GPU_avaiable:
    #     model.cuda()
    model.cuda()

    # train
    trainer = Trainer(data_pack, saving_path, args, model, optimizer)

    if not args.load_model:
        trainer.train()

    # test
    trainer.test()

    # scores
    scores(saving_path, args, data_pack)

    # eer
    compute_EER(saving_path, args)

if __name__ == "__main__":
    main()
