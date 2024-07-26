import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import * 


class UNet2(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=128, num_layers_to_concat = 4, upsample_mode='bilinear', 
                 pad='zero', norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True):
        super(UNet2, self).__init__()

        self.conv_b_e = unetConv2(num_input_channels, num_input_channels, 3, 1, norm_layer, need_bias, pad)
        self.conv_b_d = unetConv2(num_input_channels + num_layers_to_concat, num_input_channels, 3, 1, norm_layer, need_bias, pad)
        self.down_r = unetConv2(num_input_channels, num_input_channels, 3, 2, norm_layer, need_bias, pad)
        self.conv_p_e = unetConv2(num_input_channels, num_layers_to_concat, 1, 1, norm_layer, need_bias, pad)
        self.up_p = unetUp(num_input_channels, upsample_mode, need_bias, pad, norm_layer)
        self.conv_p_d = unetConv2(num_input_channels, num_input_channels, 1, 1, norm_layer, need_bias, pad)
        self.final = conv(num_input_channels, 1, 1, 1, bias=need_bias, pad=pad)


    def forward(self, inputs):

        # encoder
        downs = [inputs]
        layers_to_concat = []
        for d in range(5):
            layers_to_concat.append(self.conv_p_e(downs[-1]))
            # print(layers_to_concat[-1].shape)
            down = self.down_r(downs[-1])
            # print(down.shape)
            downs.append(self.conv_b_e(down))

        # decoder
        ups = [down]
        for e in range(5):
            upsample = self.up_p(ups[-1], layers_to_concat[4-e])
            # print('CONCAT:', upsample.shape)
            ups.append(self.conv_p_d(self.conv_b_d(upsample)))
            # print('UPS:', ups[-1].shape)

        # print(ups[-1].shape)
        final = self.final(ups[-1])
        final -= final[:, :, :30, :30].mean()

        return final


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        if norm_layer is not None:
            self.conv = nn.Sequential(conv(in_size, out_size, kernel_size, stride=stride, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.PReLU(),)

        else:
            self.conv = nn.Sequential(conv(in_size, out_size, kernel_size, stride=stride, bias=need_bias, pad=pad),
                                       nn.PReLU(),)

    def forward(self, inputs):
        outputs= self.conv(inputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, norm_layer, same_num_filt=True):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)

        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   norm_layer(out_size))
            
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) if (inputs2.size(2) - in1_up.size(2)) < 0 else in1_up.size(2)
            diff3 = (inputs2.size(3) - in1_up.size(3)) if (inputs2.size(3) - in1_up.size(3)) < 0 else in1_up.size(3)
            # assuming in1_up.size >= inputs2.size 
            inputs1_ = in1_up[:, :, : diff2, : diff3]
            
        else:
            inputs1_ = in1_up

        # print(in1_up.shape, inputs1_.shape, inputs2.shape)
        output= torch.cat([inputs1_, inputs2], 1)

        return output