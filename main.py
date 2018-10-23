from __future__ import print_function
import argparse
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import _netG, _netD, weights_init
from metrics import Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for D, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for G, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--dropoutD', default=None, type=float, help='implements dropout in netD')
parser.add_argument('--dropoutG', default=None, type=float, help='implements dropout in netG')
parser.add_argument('--fixD', action='store_true', help='fix the Discriminator while training Generator')
parser.add_argument('--GtoDratio', default=1, type=int, help='How many G iters to run per D iter')
parser.add_argument('--c_rate', default=10, type=int, help='How many epochs to save a checkpoint')
parser.add_argument('--v_rate', default=10, type=int, help='How many epochs to save a generated visual sample')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
    # os.makedirs(opt.outf+"/model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset, 'dataset is not loaded, check if the --dataset flag is properly set'
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

netG = _netG(ngpu, dropout=opt.dropoutG, nz=nz, ngf=ngf, nc=nc)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = _netD(ngpu, dropout=opt.dropoutD, ndf=ndf, nc=nc)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
# Save fixed latent Z vector
torch.save(fixed_noise, '%s/fixed_noise.pth' % (opt.outf))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

met = Metrics(beta=0.9)
metric_names = ('LossD', 'LossG', 'D(x)', 'D(G(z))1', 'D(G(z))2')
met.update_metrics(metric_names, None)

if opt.fixD: netD.eval()  # set netD to eval while Training netG
for epoch in range(1, opt.niter + 1):
    LossD_epoch, LossG_epoch, D_x_epoch, D_G_z1_epoch, D_G_z2_epoch = 0., 0., 0., 0., 0.
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward(retain_graph=True)
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        # netG.eval()
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        for G_iter in range(int(opt.GtoDratio)):
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # netG.train() # default
            netG.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labelv)
            errG.backward(retain_graph=True)
            D_G_z2 = output.data.mean()
            optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        LossD_epoch += errD.data[0]
        LossG_epoch += errG.data[0]
        D_x_epoch += D_x
        D_G_z1_epoch += D_G_z1
        D_G_z2_epoch += D_G_z2
        if epoch == 1 and i == 0:
            vutils.save_image(real_cpu, '%s/real_samples.jpg' % (opt.outf), normalize=True)
    # calculate metrics_dict of this epoch, prepare to write to sys.stdout
    try:
        i += 1
        metric_values = (LossD_epoch / i, LossG_epoch / i, D_x_epoch / i, D_G_z1_epoch / i, D_G_z2_epoch / i)
        met.update_metrics(metric_names, metric_values)
    except NameError:
        pass

    if epoch % opt.v_rate == 0 or opt.niter - epoch <= 10:
        fake = netG(fixed_noise)
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%04d.jpg' % (opt.outf, epoch), normalize=True)
        met.write_metrics()

    # do checkpointing - Are saved in outf/model/
    if epoch % opt.c_rate == 0 or opt.niter - epoch <= 10:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
