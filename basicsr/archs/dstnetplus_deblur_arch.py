import torch
import time
import torch.nn.functional as F
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, make_layer
from basicsr.archs.prog_dynconv import IDynamicDWConv
from einops import rearrange


@ARCH_REGISTRY.register()
class DSTNetPlus_Final(nn.Module):
    def __init__(self, num_feat=96, num_kernel_block=5, num_block=25, nonblind_denoise=False):
        super().__init__()
        self.num_feat = num_feat
        self.nonblind_denoise = nonblind_denoise

        # extractor & reconstruction
        self.feat_extractor = nn.Conv2d(4 if self.nonblind_denoise else 3, num_feat, 3, 1, 1)
        self.recons = nn.Conv2d(num_feat, 3, 3, 1, 1)

        # wave pro
        self.wave = HaarDownsampling(num_feat)
        self.x_wave_2_conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        self.x_wave_2_conv2 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)

        # propogation branch
        self.forward_propagation = ForwardProp(num_feat, num_kernel_block, num_block)
        self.backward_propagation = BackwardProp(num_feat, num_kernel_block, num_block)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True) #nn.GELU() #

    def spatial_padding(self, x, pad_size):
        """ Apply spatial pdding.

        Args:
            x (Tensor): Input blurry sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Padded blurry sequence with shape (n, t, c, h_pad, w_pad).

        """
        n, t, c, h, w = x.size()

        pad_h = (pad_size - h % pad_size) % pad_size
        pad_w = (pad_size - w % pad_size) % pad_size

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def forward(self, lrs):
        b, t, c, h, w = lrs.size()

        if h % 2 != 0 or w % 2 != 0:
            lrs = self.spatial_padding(lrs, pad_size=2)

        # [b, c, t, h, w] --> [bt, c, h, w]
        lrs_feature = self.feat_extractor(rearrange(lrs, 'b t c h w -> (b t) c h w'))
        # wavelet decomposition
        tf_wave1_l, tf_wave1_h = self.wave(lrs_feature)
        # high-frequency processing
        tf_wave1_h = self.x_wave_2_conv2(self.lrelu(self.x_wave_2_conv1(tf_wave1_h)))
        # low-frequency processing
        # backward propagation
        tf_wave1_l = self.backward_propagation(rearrange(tf_wave1_l, '(b t) c h w -> b t c h w', b=b))
        # forward propagation
        tf_wave1_l = rearrange(self.forward_propagation(tf_wave1_l), 'b t c h w -> (b t) c h w')
        # inverse wavelet
        pro_feat = self.wave(torch.cat([tf_wave1_l, tf_wave1_h], dim=1), rev=True)
        # reconstruction
        out = rearrange(self.recons(pro_feat), '(b t) c h w -> b t c h w', b=b)

        if self.nonblind_denoise:
            out = out.contiguous() + lrs[:, :, :3, ...]
        else:
            out = out.contiguous() + lrs
        return out[:, :, :, :h, :w]


class ResidualBlocks2D(nn.Module):
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, fea):
        return self.main(fea)


# Learnable pixel attention guided fusion
class PAGF(nn.Module):
    def __init__(self, num_fea):
        super().__init__()

        self.conv0 = nn.Conv2d(num_fea*2, num_fea*2, 1, 1, 0)
        self.conv1 = nn.Conv2d(num_fea, num_fea, 3, 1, 1)

    def forward(self, x_feat, feat_prop):
        q, k = self.conv0(torch.cat([x_feat, feat_prop], dim=1)).chunk(2, dim=1)
        sim = torch.sigmoid(q * k)
        x_feat = x_feat * (1-sim) + feat_prop * sim
        return self.conv1(x_feat)

class BackwardProp(nn.Module):
    def __init__(self, num_feat=64, num_kernel_block=3, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        self.fusion = PAGF(num_feat)
        self.kernel_conv_pixel = IDynamicDWConv(channels=num_feat, n_blocks=num_kernel_block,
                        kernel_size=3, group_channels=1, conv_group=1)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()
        backward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        # propagation
        for i in range(t - 1, -1, -1):
            x_feat = feature[:, i, :, :, :]
            feat_prop = self.fusion(x_feat, feat_prop)
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            backward_list.append(feat_prop)

        backward_list = backward_list[::-1]
        conv3d_feature = torch.stack(backward_list, dim=1)
        return conv3d_feature


class ForwardProp(nn.Module):
    def __init__(self, num_feat=64, num_kernel_block=3, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        self.fusion = PAGF(num_feat)
        self.kernel_conv_pixel = IDynamicDWConv(channels=num_feat, n_blocks=num_kernel_block,
                        kernel_size=3, group_channels=1, conv_group=1)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()
        forward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        for i in range(0, t):
            x_feat = feature[:, i, :, :, :]
            feat_prop = self.fusion(x_feat, feat_prop)
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            forward_list.append(feat_prop)

        conv3d_feature = torch.stack(forward_list, dim=1)
        return conv3d_feature


#----------------------------Haar wavelet------------------------------
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(in_channel), r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out[:,:self.channel_in,:,:],out[:,self.channel_in:self.channel_in*4,:,:]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)


def get_params(net):
    x = torch.randn(1, 1, 3, 256, 256).cuda()
    print(x.shape)
    from thop import profile
    flops,params = profile(net,inputs=(x,))
    print(f"flops:{flops/1e9}, params:{params/1e6}")


if __name__ == "__main__":
    # x = torch.rand(3, 45, 3, 80, 60).cuda()
    model = DSTNetPlus_Final(64, 3, 15, nonblind_denoise=False).cuda()
    print(model)
    get_params(model)

    # long sequence test
    # haar downsampling
    x = torch.rand(5, 35, 3, 320, 180)
    conv = nn.Conv2d(3, 64, 3, 1, 1)
    wave = HaarDownsampling(64)
    b, t, c, h, w = x.size()
    lr = conv(rearrange(x, 'b t c h w -> (b t) c h w'))
    if t > 30:
        n_seq = t // 30 + 1
        x_in = lr.chunk(n_seq, dim=0)
        wave_l, wave_h = [], []
        conv_x = []
        for xi in x_in:
            wl, wh = wave(xi)
            conv_x.append(xi)
            wave_l.append(wl)
            wave_h.append(wh)
        wave_l, wave_h = torch.cat(wave_l, dim=0), torch.cat(wave_h, dim=0)
    else:
        wave_l, wave_h = wave(lr)

    # wave2_l, wave2_h = wave(lr)

    # print(f'wave_h: {wave_h.shape}, wave_l: {wave_l.shape}')
    # print(f'wave2_h: {wave2_h.shape}, wave2_l: {wave2_l.shape}')
    # assert torch.allclose(wave_l, wave2_l, rtol=1e-7, atol=1e-7)
    # assert torch.allclose(wave_h, wave2_h, rtol=1e-7, atol=1e-7)
    # print(wave_h.eq(wave2_h))
    # print(f'diff: {torch.max(wave2_h - wave_h)}')

    # haar upsampling & reconstruction
    # b, t = 5, 35
    # wave_l = torch.rand(b*t, 64*3, 180, 90)
    # wave_h = torch.rand(b*t, 64, 180, 90)

    # wave = HaarDownsampling(64)
    # recons = nn.Conv2d(64, 64, 3, 1, 1)
    # if t > 30:
    #     n_seq = t // 30 + 1
    #     rev_l, rev_h = wave_l.chunk(n_seq, dim=0), wave_h.chunk(n_seq, dim=0)
    #     out = []
    #     for (rl, rh) in zip(rev_l, rev_h):
    #         pro_feat = wave(torch.cat([rl, rh], dim=1), rev=True)
    #         out.append(recons(pro_feat))
    #     out = rearrange(torch.cat(out, dim=0), '(b t) c h w -> b t c h w', b=b)
    #     # print('clip results')
    # else:
    #     pro_feat = wave(torch.cat([wave_l, wave_h], dim=1), rev=True)
    #     # reconstruction
    #     out = rearrange(recons(pro_feat), '(b t) c h w -> b t c h w', b=b)

    # feat = wave(torch.cat([wave_l, wave_h], dim=1), rev=True)
    # # reconstruction
    # out2 = rearrange(recons(feat), '(b t) c h w -> b t c h w', b=b)

    # print(f'out: {out.shape}, out2: {out2.shape}')
    # assert torch.allclose(out, out2, rtol=1e-7, atol=1e-7)
    # # print(f'is out equal out2? {out.eq(out2)}')
    # print(f'diff: {torch.max(out2 - out)}')

