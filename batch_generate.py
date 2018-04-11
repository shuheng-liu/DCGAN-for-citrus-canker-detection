import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', default='/checkpoints', help='path to checkpoints dir')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--output', default='/output', help='path to output dir')
opt = parser.parse_args()

checkpoints_dir = opt.checkpoints
output_dir = opt.output

checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
checkpoints_D = [os.path.join(checkpoints_dir, f) for f in checkpoints if f.startswith('netD')]
checkpoints_G = [os.path.join(checkpoints_dir, f) for f in checkpoints if f.startswith('netG')]
checkpoints_D.sort()
checkpoints_G.sort()


def filter(s):
    r = ''
    for ch in s.split('/')[-1]:  # ignore leading directory
        if ch.isdigit():
            r = r + ch
    return int(r)

print('netD paths are', checkpoints_D)
print('netG paths are', checkpoints_G)

for netD, netG in zip(checkpoints_D, checkpoints_G):
    print("netD == %s" % netD)
    print("netG == %s" % netG)
    epoch = filter(netD)
    assert epoch == filter(netG)
    try:
        epoch_path = os.path.join(output_dir, str(epoch))
    except FileExistsError:
        pass
    os.mkdir(epoch_path)
    os.system("python generate.py --netG %s --outf %s %s --nsample 10 --batchSize 128"
              % (netG, epoch_path, "--cuda" if opt.cuda else ""))

