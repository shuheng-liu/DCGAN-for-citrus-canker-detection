import argparse
import os

from crop_down import handle_folder

parser = argparse.ArgumentParser()
parser.add_argument('--inf', default='/imgs', help='input directory, under which several folders are nested')
parser.add_argument('--outf', default='/output', help='output directory, under which respective folders are created')
parser.add_argument('--maxImages', default=10, type=int, help='')
opt = parser.parse_args()

inf = opt.inf
outf = opt.outf
maxImages = int(opt.maxImages)

try: os.mkdir(outf)
except: pass
handle_folder(inf, outf, max_imgs=maxImages)
