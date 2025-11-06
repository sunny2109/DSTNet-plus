import argparse
import cv2
import glob
import os
import torch
import logging
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader
from basicsr.utils import get_root_logger
from basicsr.archs.dstnetplus_deblur_arch import DSTNetPlus_Final
from basicsr.utils import utils_image as util
from basicsr.data.dataset_video_test import VideoRecurrentTestDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--testname', type=str, default='Set8', help='name of the testset')
    parser.add_argument('--folder_lq', type=str, default='datasets/DAVIS/test/set8',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default='datasets/DAVIS/test/set8',
                        help='input ground-truth test video folder')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/DSTNetPlus_L_denoise.pth',
                        help='input path of the pretrained checkpoints')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')
    args = parser.parse_args()

    save_dir = f'results/{args.testname}/{args.sigma}'
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)

    logger = get_root_logger(logger_name='Denoise', log_level=logging.INFO, log_file=osp.join(save_dir, f'test_log.log'))

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSTNetPlus_Final(num_feat=96, num_kernel_block=5, num_block=25, nonblind_denoise=True)
    pretrained_model = torch.load(args.ckpt_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    logger.info(f'Loading model from {args.ckpt_path}')
    model.eval()
    model = model.to(device)

    test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq, 'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

    logger.info('Start inference')
    for idx, batch in enumerate(test_loader):
        lq = batch['L'].to(device)
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        # inference
        with torch.no_grad():
            output = []
            max_seq_len = 30 # for saving memory
            for i in range(0, lq.size(1), max_seq_len):
                output.append(model(lq[:, i:i + max_seq_len]).cpu())
            output = torch.cat(output, dim=1)

        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        for i in range(output.shape[1]):
            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
            if args.save_result:
                seq_ = osp.basename(batch['lq_path'][i][0]).split('.')[0]
                os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}.png', img)

            # evaluate psnr/ssim
            if gt is not None:
                img_gt = gt[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = np.squeeze(img_gt)

                test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                if img_gt.ndim == 3:  # RGB image
                    img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                    img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                else:
                    test_results_folder['psnr_y'] = test_results_folder['psnr']
                    test_results_folder['ssim_y'] = test_results_folder['ssim']

        if gt is not None:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            logger.info('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                      format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

    # summarize psnr/ssim
    if gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        logger.info('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
              format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':
    main()