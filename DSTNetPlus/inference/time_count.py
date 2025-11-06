import os
import argparse
import numpy as np
import os.path as osp
import logging
import time
import torch

from tqdm import tqdm
from basicsr.archs.deblur_gating_v9_arch import Deblurv8
from basicsr.archs.deblur_newbase_arch import BaseDeblur
from basicsr.archs.dstnet_arch import Deblur
from basicsr.archs.shiftnet_arch import GShiftNet, GShiftNetPlus
from collections import OrderedDict
from basicsr.utils import get_root_logger, get_time_str


def main(args):
    # Set log file
    log_file = osp.join(args.save_path, f'{args.model_name}_runtime_.log')
    logger = get_root_logger(logger_name='Runtime', log_level=logging.INFO, log_file=log_file)

    logger.info(torch.__version__)               # pytorch version
    logger.info(torch.version.cuda)              # cuda version
    logger.info(torch.backends.cudnn.version())  # cudnn version
    logger.info('{:>16s} : {:s}'.format('Model Name', args.model_name))

    torch.cuda.set_device(0)      # set GPU ID
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define network and load model
    model = GShiftNetPlus() #GShiftNet() #BaseDeblur() #Deblurv8(num_feat=96, num_kernel_block=5, num_block=25) #Deblur(64, 15) #

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('{:>16s} : {:<.4f} [K]'.format('#Params', num_parameters/10**3))

    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    """
    TESTING
    """
    input_data = torch.randn((1, 10, 3, args.crop_size[0], args.crop_size[1])).to(device)

    # GPU warmp up
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_data)
            logger.info('output:{0}, Max Memery [M]: {1}'.format(model(input_data).shape, torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2 / 8))

    print("Start timing ...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in tqdm(range(args.repeat)):
            start.record()
            _ = model(input_data)
            end.record()

            torch.cuda.synchronize()

            test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        logger.info('------> Average runtime of ({}) is : {:.6f} ms'.format(args.model_name, ave_runtime / args.batch_size / 8))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specify submission
    parser.add_argument("--model_name", type=str, default="ShiftNet")
    # specify dirs
    parser.add_argument("--save_path", type=str, default='results')
    # specify test case
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, nargs="+", default=[720, 1280])
    args = parser.parse_args()

    main(args)
