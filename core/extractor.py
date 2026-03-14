import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        if x.shape[1] != y.shape[1]:
            return y
       
        return self.relu(x+y) 

class BasicEncoder_resconv(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder_resconv, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)

        self.conv_after_1 = nn.Conv2d(96, 96, kernel_size=1)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv_after_2 = nn.Conv2d(128, 128, kernel_size=1)

        # 3 scales
        self.layer4 = self._make_layer(160, stride=2)
        self.conv_after_3 = nn.Conv2d(160, 160, kernel_size=1) 
            
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(in_planes=self.in_planes, planes=dim, norm_fn=self.norm_fn, stride=stride)
        layer2 = ResidualBlock(in_planes=dim, planes=dim, norm_fn=self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x, bw=True):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x) # h/2, 64

        x = self.layer1(x) # h/2, 64

        x = self.layer2(x) #h/4, 96
        enc_out2 = self.conv_after_1(x)
        x = self.layer3(x) #h/8, 128
        enc_out3 = self.conv_after_2(x)

    
        x = self.layer4(x) # h/16, 160
        x = self.conv_after_3(x)
        enc_out4 = x
        
        if bw:
            enc_out4_1, enc_out4_2 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            fw_bw_enc_out4 = (torch.cat((enc_out4_1, enc_out4_2), dim=0), torch.cat((enc_out4_2, enc_out4_1), dim=0))
        
            up2_out_1, up2_out_2 = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
            fw_bw_up2_out = (torch.cat((up2_out_1, up2_out_2), dim=0), torch.cat((up2_out_2, up2_out_1), dim=0))

            up1_out_1, up1_out_2 = torch.split(enc_out2, [batch_dim, batch_dim], dim=0)
            fw_bw_up1_out = (torch.cat((up1_out_1, up1_out_2), dim=0), torch.cat((up1_out_2, up1_out_1), dim=0))

            return [fw_bw_enc_out4, fw_bw_up2_out, fw_bw_up1_out]   
        
        else:

            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(enc_out2, [batch_dim, batch_dim], dim=0)
          
            return [enc_out4, up2_out, up1_out]   
        


class Basic_Context_Encoder_resconv_unet(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', size="M", dropout=0.0, shared=True):
        super(Basic_Context_Encoder_resconv_unet, self).__init__()
        self.norm_fn = norm_fn
        self.shared = shared
       
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=128)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(128)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(128)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 128
        self.top_down_res_layer= self._make_layer(128, stride=1)

        self.in_planes = 128 + 256
        self.consolidation_res = self._make_layer(128, stride=1)
        self.consolidation_conv = nn.Conv2d(128, 256, kernel_size=1)
        #only for finest upsampling
        self.consolidation_conv_finest = nn.Conv2d(128, 128, kernel_size=1)
    
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(in_planes=self.in_planes, planes=dim, norm_fn=self.norm_fn, stride=stride)
        layer2 = ResidualBlock(in_planes=dim, planes=dim, norm_fn=self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
      
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)#h,w
        x = self.top_down_res_layer(x)#h,w
        enc_out1 = x

        x = self.top_down_res_layer(x)#h/2,w/2
        cur_h, cur_w = list(x.size())[-2:]
        enc_out2 = TF.resize(x, (cur_h//2, cur_w//2))#h/4, w/4
    
        x = self.top_down_res_layer(enc_out2)#h/4,w/4
        cur_h, cur_w = list(x.size())[-2:]
        enc_out3 = TF.resize(x, (cur_h//2, cur_w//2))#h/8, w/8

        x = self.top_down_res_layer(enc_out3)#h/8,w/8
        cur_h, cur_w = list(x.size())[-2:]
        x = TF.resize(x, (cur_h//2, cur_w//2))#h/16, w/16
        enc_out4 = self.consolidation_conv(x)#h/16, w/16

        #uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.consolidation_conv(self.consolidation_res(up2layer_input))
        #uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.consolidation_conv(self.consolidation_res(up1layer_input))
        #uplayer1:#for finest upsampling
        cur_h, cur_w = list(enc_out1.size())[-2:]
        up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
        up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
        up0_out = self.consolidation_conv_finest(self.consolidation_res(up0layer_input))

        net, inp = torch.split(enc_out4, [128, 128], dim=1)
        net_4 = torch.tanh(net)
        inp_4 = torch.relu(inp)

        net, inp = torch.split(up2_out, [128, 128], dim=1)
        net_2 = torch.tanh(net)
        inp_2 = torch.relu(inp)

        net, inp = torch.split(up1_out, [128, 128], dim=1)
        net_1 = torch.tanh(net)
        inp_1 = torch.relu(inp)

        inp_0 = torch.tanh(up0_out)
        
        return [[net_4, inp_4], [net_2, inp_2], [net_1, inp_1], [inp_0]]    
       