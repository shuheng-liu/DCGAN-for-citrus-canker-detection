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

for folder in os.listdir(inf):
    if not os.path.isdir(os.path.join(inf, folder)): continue
    try:
        os.mkdir(os.path.join(outf, folder))
    except:
        pass
    print("handling %s" % folder)
    handle_folder(os.path.join(inf, folder), os.path.join(outf, folder), max_imgs=maxImages)
