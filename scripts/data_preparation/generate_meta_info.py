import glob
from os import path as osp
from PIL import Image
from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = '/data0/lvjt/datasets/DAVIS/train/'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DAVIS_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_davis():
    '''generate meta_info_DAVIS_GT.txt for DAVIS
    '''
    gt_folder = '/data0/lvjt/datasets/DAVIS/'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DAVIS_GT.txt'

    f = open(meta_info_txt, "w+")
    file_list = sorted(glob.glob(osp.join(gt_folder, 'train/*')))
    total_frames = 0
    for idx, img_path in enumerate(file_list):
        name = osp.basename(img_path)
        frames = sorted(glob.glob(osp.join(img_path, '*')))
        img = Image.open(frames[0])
        width, height = img.size
        mode = img.mode
        if mode == 'RGB':
            n_channel = 3
        elif mode == 'L':
            n_channel = 1
        else:
            raise ValueError(f'Unsupported mode {mode}.')

        info = f'{name} {len(frames)} ({height},{width},{n_channel})'

        total_frames += len(frames)

        f.write(f'{info}\n')

    assert total_frames == 6208, f'DAVIS training set should have 6208 images, but got {total_frames} images'

if __name__ == '__main__':
    # generate_meta_info_div2k()
    generate_meta_info_davis()
