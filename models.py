# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init


# Ref: _netD in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Discriminator_I(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1):
        super(Discriminator_I, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# Discriminator_Iに時間方向の畳み込みを追加
class Discriminator_V(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, ngpu=1):
        super(Discriminator_V, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x T/16  x 6 x 6
            Flatten(),
            nn.Linear(int((ndf*8)*(T/16)*6*6), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# Ref: _netG in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Generator_I(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=60, ngpu=1):
        super(Generator_I, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 24 x 24
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 48 x 48
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, gpu=True):
        super(GRU, self).__init__()

        self._gpu        = gpu
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 各layerの定義
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        '''
        gru_out.shape() =>  (seq_len, batch, hidden_size)
        outputs,shape() => (seq_len, batch, output_size)
        '''
        gru_out, self.hidden = self.gru(inputs, self.hidden)
        # 系列の要素ひとつずつに対して全結合を適用
        outputs = [ self.linear(self.drop(elm)) for elm in gru_out ]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            # weightをxavierで初期化
            if 'weight' in name:
                init.xavier_uniform(params)

            # 忘却しやすくなるようGRUのb_iz, b_hzを初期化
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)

            # それ以外のbiasを0に初期化
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(self.num_layers,
                      batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()


''' utils '''

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)