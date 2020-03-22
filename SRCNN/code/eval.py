import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio

from torch.utils.data import DataLoader

from dataset import TestDatasetFromFolder
from math import log10

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_21.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
device = torch.device('cuda' if cuda else 'cpu')
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

scales = [3]

criterion = torch.nn.MSELoss()

avg_psnr = 0

test_set = TestDatasetFromFolder()
testing_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

with torch.no_grad():
    for batch_num, (data, target) in enumerate(testing_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr

print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(testing_loader)))
