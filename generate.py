""" Serving DCGAN
"""
from __future__ import print_function
import argparse, torch
from dcgan import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--netG', required=True, default='', help="path to netG (for generating images)")
parser.add_argument('--outf', default='/output', help='folder to output images')
parser.add_argument('--Zvector', help="path to Serialized Z vector")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--nsample', type = int, default=20, help='number of samples to be generated')
parser.add_argument('--dropoutG', type=float, default=None, help='implements dropout in netG')
parser.add_argument('--batchSize', type=int, default=1, help='size of batch')
parser.add_argument('--eval', action='store_true', help = 'set G model to eval mode rather than train mode')
opt = parser.parse_args()
print(opt)

zvector = None
batchSize = opt.batchSize
# Load a Z vector and Retrieve the N of samples to generate
if opt.Zvector is not None:
    zvector = torch.load(opt.Zvector)
    batchSize = zvector.size()[0]
    print('Overwriting batch size to be', batchSize)

outf = "/output"
if opt.outf:
    outf = opt.outf

# GPU and CUDA
cuda = None
if opt.cuda:
    cuda = opt.cuda
ngpu = int(opt.ngpu)
nsample = int(opt.nsample)

# Generate An Image from input json or default parameters
for index in range(nsample):
    if index % 100 == 0:
        print("%4d images of %d generated" % (index, nsample))
    Generator = DCGAN(netG=opt.netG, zvector=zvector, batchSize=batchSize, nz = opt.nz, ngf = opt.ngf,
                      outf=outf, cuda=cuda, ngpu=ngpu, dropoutG=opt.dropoutG)
    Generator.build_model()
    if opt.eval:
        Generator.set_G_eval()
    Generator.generate(img_name='generated_%04d.png' % index)