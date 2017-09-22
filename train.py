# coding: utf-8

import os
import argparse
import glob
import time
import math
import skvideo.io
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from models import Discriminator_I, Discriminator_V, Generator_I, GRU


parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--ngpu', type=int, default=1,
                     help='set the number of gpu you use')
parser.add_argument('--batch-size', type=int, default=16,
                     help='set batch_size, default: 16')
parser.add_argument('--niter', type=int, default=500000,
                     help='set num of iterations, default: 500000')
parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')

args       = parser.parse_args()
cuda       = args.cuda
ngpu       = args.ngpu
batch_size = args.batch_size
n_iter     = args.niter
pre_train  = args.pre_train


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)


''' prepare dataset '''

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path+'/*')
videos = [ skvideo.io.vread(file) for file in files ]
# transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
videos = [ video.transpose(3, 0, 1, 2) / 255.0 for video in videos ]


''' prepare video sampling '''

n_videos = len(videos)
T = 16

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :]

# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    start = np.random.randint(0, noise.size(1) - (T+1))
    end = start + T
    return noise[:, start:end, :, :, :]

def random_choice():
    X = []
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        video = torch.Tensor(trim(video))
        X.append(video)
    X = torch.stack(X)
    return X

# video length distribution
video_lengths = [video.shape[1] for video in videos]


''' set models '''

img_size = 96
nc = 3
ndf = 64 # from dcgan
ngf = 64
d_E = 100 # guess
hidden_size = 100 # guess
d_C = 50
d_M = 10
nz  = d_C + d_M
# one sided label smoothing. 0.9 is a guess.
criterion = nn.BCELoss()

dis_i = Discriminator_I(nc, ndf, ngpu=ngpu)
dis_v = Discriminator_V(nc, ndf, T=T, ngpu=ngpu)
gen_i = Generator_I(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, d_M, gpu=cuda)
gru.initWeight()


''' prepare for train '''

label = torch.FloatTensor()

def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models')
def checkpoint(model, optimizer, epoch):
    filename = os.path.join(trained_path, '%s_epoch-%d' % (model.__class__.__name__, epoch))
    torch.save(model.state_dict(), filename + '.model')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)


''' adjust to cuda '''

if cuda == True:
    dis_i.cuda()
    dis_v.cuda()
    gen_i.cuda()
    gru.cuda()
    criterion.cuda()
    label = label.cuda()


# setup optimizer
lr = 0.0002
betas=(0.5, 0.999)
optim_Di  = optim.Adam(dis_i.parameters(), lr=lr, betas=betas)
optim_Dv  = optim.Adam(dis_v.parameters(), lr=lr, betas=betas)
optim_Gi  = optim.Adam(gen_i.parameters(), lr=lr, betas=betas)
optim_GRU = optim.Adam(gru.parameters(),   lr=lr, betas=betas)


''' use pre-trained models '''

if pre_train == True:
    dis_i.load_state_dict(torch.load(trained_path + '/Discriminator_I.model'))
    dis_v.load_state_dict(torch.load(trained_path + '/Discriminator_V.model'))
    gen_i.load_state_dict(torch.load(trained_path + '/Generator_I.model'))
    gru.load_state_dict(torch.load(trained_path + '/GRU.model'))
    optim_Di.load_state_dict(torch.load(trained_path + '/Discriminator_I.state'))
    optim_Dv.load_state_dict(torch.load(trained_path + '/Discriminator_V.state'))
    optim_Gi.load_state_dict(torch.load(trained_path + '/Generator_I.state'))
    optim_GRU.load_state_dict(torch.load(trained_path + '/GRU.state'))


''' calc grad of models '''

def bp_i(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_i(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    return err.data[0], outputs.data.mean()

def bp_v(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_v(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    return err.data[0], outputs.data.mean()


''' gen input noise for fake video '''

def gen_z(n_frames):
    z_C = Variable(torch.randn(batch_size, d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    eps = Variable(torch.randn(n_frames, batch_size, d_E))
    if cuda == True:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    return z.view(batch_size, n_frames, nz, 1, 1)


''' train models '''

start_time = time.time()

for epoch in range(1, n_iter+1):
    ''' prepare real images '''
    # real_videos.size() => (batch_size, nc, T, img_size, img_size)
    real_videos = random_choice()
    if cuda == True:
        real_videos = real_videos.cuda()
    real_videos = Variable(real_videos)
    real_img = real_videos[:, :, np.random.randint(0, T), :, :]

    ''' prepare fake images '''
    # note that n_frames is sampled from video length distribution
    n_frames = video_lengths[np.random.randint(0, n_videos)]
    Z = gen_z(n_frames)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
    # trim => (batch_size, T, nz, 1, 1)
    Z = trim_noise(Z)
    # generate videos
    Z = Z.contiguous().view(batch_size*T, nz, 1, 1)
    fake_videos = gen_i(Z)
    fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
    # transpose => (batch_size, nc, T, img_size, img_size)
    fake_videos = fake_videos.transpose(2, 1)
    # img sampling
    fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

    ''' train discriminators '''
    # video
    dis_v.zero_grad()
    err_Dv_real, Dv_real_mean = bp_v(real_videos, 0.9)
    err_Dv_fake, Dv_fake_mean = bp_v(fake_videos.detach(), 0)
    err_Dv = err_Dv_real + err_Dv_fake
    optim_Dv.step()
    # image
    dis_i.zero_grad()
    err_Di_real, Di_real_mean = bp_i(real_img, 0.9)
    err_Di_fake, Di_fake_mean = bp_i(fake_img.detach(), 0)
    err_Di = err_Di_real + err_Di_fake
    optim_Di.step()


    ''' train generators '''
    gen_i.zero_grad()
    gru.zero_grad()
    # video. notice retain=True for back prop twice
    err_Gv, _ = bp_v(fake_videos, 0.9, retain=True)
    # images
    err_Gi, _ = bp_i(fake_img, 0.9)
    optim_Gi.step()
    optim_GRU.step()

    if epoch % 100 == 0:
        print('[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f'
              % (epoch, n_iter, timeSince(start_time), err_Di, err_Dv, err_Gi, err_Gv, Di_real_mean, Di_fake_mean, Dv_real_mean, Dv_fake_mean))

    if epoch % 1000 == 0:
        save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)

    if epoch % 10000 == 0:
        checkpoint(dis_i, optim_Di, epoch)
        checkpoint(dis_v, optim_Dv, epoch)
        checkpoint(gen_i, optim_Gi, epoch)
        checkpoint(gru,   optim_GRU, epoch)
