import math

import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
from scipy.misc import imread


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_psnr(y, x):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse == 0:
        return math.inf
    return 20 * math.log10(255.0 / math.sqrt(mse))

 
def compare_l1_loss(y, x):
   return np.sum(np.abs(x / 256. - y / 256.)) / 3.


if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    path_true = args.data_path
    path_pred = args.output_path

    psnr = []
    l1_loss = []
    names = []
    index = 1

    files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
    for fn in sorted(files):
        name = basename(str(fn))
        names.append(name)

        img_gt = imread(str(fn))
        img_pred = imread(path_pred + '/' + basename(str(fn)))


        if args.debug != 0:
            plt.subplot('121')
            plt.imshow(img_gt)
            plt.title('Groud truth')
            plt.subplot('122')
            plt.imshow(img_pred)
            plt.title('Output')
            plt.show()

        psnr.append(compare_psnr(img_gt, img_pred))
        l1_loss.append(compare_l1_loss(img_gt, img_pred))
        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.15f" % round(np.mean(psnr), 4),
                "L1: %.15f" % round(np.mean(l1_loss), 4),
            )
        index += 1

    print(
        "PSNR: %.15f" % round(np.mean(psnr), 4),
        "PSNR Variance: %.15f" % round(np.var(psnr), 4),
        "L1: %.15f" % round(np.mean(l1_loss), 4),
        "L1 Variance: %.15f" % round(np.var(l1_loss), 4),
    )
