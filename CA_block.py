from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import Tuple
import torch.nn.functional as F
# from torchsummary import summary
# from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

##CA BLOCK
class CABlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes,groups=groups)
        self.bn2 = norm_layer(planes)
        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1,bias=False),  # 32*33*33
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.downsample = downsample
        self.stride = stride
        self.planes=planes

    def forward(self, x):
        x, attn_last,if_attn =x##attn_last: downsampled attention maps from last layer as a prior knowledge
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out+identity)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True) #dim=1 表示对第一维求平均值，即[B,C,H,W]的C，使用keepdim保证仍是四维
        attn = torch.cat((avg_out, max_out), dim=1) #同理，channel方向做拼接，[B,2,H,W]
        attn = self.attn(attn) #C再次变为1
        if attn_last is not None:
            attn = attn_last * attn

        attn = attn.repeat(1, self.planes, 1, 1) #相对于把C维度扩展为下一个CABlock的输入维度，其他维度不变
        if if_attn:
            out = out *attn


        return out,attn[:, 0, :, :].unsqueeze(1),True #输出此次特征矩阵，并输出此次attn特征图，设置下一次if_attn=True



# ---------------------------------------------------------------------------
# Helper: Affine grid for in‑plane rotation around the feature‑map centre
# ---------------------------------------------------------------------------

def _rotation_grid(b: int, h: int, w: int, theta: torch.Tensor) -> torch.Tensor:  # (B) → (B,H,W,2)
    """Return an affine grid that rotates the input by **−theta** so that
    the desired orientation aligns with the vertical axis.  Align_corners=False
    is used for safety with arbitrary resolutions.
    """
    device = theta.device

    # Cos / sin have shape (B,1,1)
    cos_t: torch.Tensor = torch.cos(theta).view(b, 1, 1)
    sin_t: torch.Tensor = torch.sin(theta).view(b, 1, 1)

    # 2×3 affine matrices per batch element
    affine = torch.zeros(b, 2, 3, device=device)
    affine[:, 0, 0] = cos_t.squeeze()
    affine[:, 0, 1] = sin_t.squeeze()
    affine[:, 1, 0] = -sin_t.squeeze()
    affine[:, 1, 1] = cos_t.squeeze()

    return F.affine_grid(affine, torch.Size((b, 1, h, w)), align_corners=False)


# ---------------------------------------------------------------------------
# OrientationPooling – "OAP" without explicit for‑loops
# ---------------------------------------------------------------------------

class OrientationPooling(nn.Module):
    """Differentiable average‑pooling along a learnable orientation.

    Given an input feature map *x* ∈ ℝᴮ×ᶜ×ᴴ×ᵂ, the layer learns an angle θ
    (initialised randomly in (0,π)) and produces pooled features of shape
    (B,C,H,1) by rotating *x* by −θ, performing mean‑pooling across the
    horizontal axis, and returning to the original orientation implicitly.

    The operation is mathematically equivalent to the OAP definition in the
    proposal but implemented with `grid_sample`, which is natively vectorised
    and fully supported by autograd on both GPU and CPU.
    """

    def __init__(self, learnable: bool = True, init: float | None = None):
        super().__init__()
        # Random uniform θ ∈ (0,π) if not provided
        if init is None:
            init = float(torch.rand(1).item() * math.pi)
        theta = torch.tensor(init)
        self.theta = nn.Parameter(theta) if learnable else theta  # shape ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W) → (B,C,H,1)
        b, c, h, w = x.shape
        # Constrain θ to [0,π] for numeric stability using sigmoid → (0,1)
        theta = torch.sigmoid(self.theta) * math.pi

        # Build affine grid and rotate
        grid = _rotation_grid(b, h, w, theta.expand(b))  # broadcast θ to batch
        x_rot = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

        # Mean‑pool along the horizontal dimension (dim=3)
        pooled = x_rot.mean(dim=3, keepdim=True)  # (B,C,H,1)
        return pooled


# ---------------------------------------------------------------------------
# SOA – Single‑Orientation Attention block
# ---------------------------------------------------------------------------

class SOA(nn.Module):
    expansion = 1
    def __init__(self, inp, oup, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None, reduction: int = 16):
        super().__init__()
        # ------------- 预处理，与 CCA_Y 保持一致 -------------
        self.preprocess = nn.Sequential(
            conv3x3(inp, oup, stride, groups=groups),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            conv1x1(oup, oup, groups=groups),
            nn.BatchNorm2d(oup)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        # ------------- SOA 核心 -------------
        # 使用向量编码的可学习方向池化（初始90°垂直）
        self.pool = OrientationPoolingVec(init=2*math.pi/5)
        self.squeeze = nn.Conv2d(oup, max(8, oup // reduction), 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.excitate = nn.Conv2d(max(8, oup // reduction), oup, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 期望输入格式: (x, a_h_l, a_w_l, if_attn)
        x, a_h_l, a_w_l, if_attn = inputs
        identity = x

        # 预处理和残差
        out = self.preprocess(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        identity = out

        # SOA 注意力
        y = self.pool(identity)              # (B,C,H,1)
        y = self.squeeze(y)
        y = self.act(y)
        y = self.excitate(y)
        w = self.sigmoid(y)                  # (B,C,H,1)

        # 生成高度/宽度注意力图
        a_h = w if a_h_l is None else w * a_h_l       # (B,C,H,1)
        a_w_base = a_h.permute(0, 1, 3, 2)            # (B,C,1,H)
        a_w = a_w_base if a_w_l is None else a_w_base * a_w_l

        # 应用注意力
        out = identity * a_h if if_attn else identity

        return out, a_h, a_w, True



class CCA_Y(nn.Module):
    expansion = 1
    def __init__(self, inp, oup, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None, reduction=32):
        super(CCA_Y, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=True)
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(oup, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.preprocess = nn.Sequential(
            conv3x3(inp,oup,stride,groups=groups),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            conv1x1(oup,oup,groups=groups),
            nn.BatchNorm2d(oup)
        )
        self.downsample = downsample

    def forward(self, x):
        
        x, a_h_l,a_w_l,if_attn = x
        identity = x
        out = self.preprocess(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out+identity)
        identity = out
        
        b,c,h,w = out.size()
        x_h = self.pool_h(out)
        x_w = self.pool_w(out).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        
        if a_w_l is not None:
            #print('a_h:'+str(a_h.shape)+'a_h_l'+str(a_h_l.shape))
            a_h = a_h*a_h_l
            a_w = a_w*a_w_l
        out = identity * a_h
        return out,a_h,a_w,True

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    expansion = 1
    def __init__(self, inp, oup, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=True)
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(oup, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.preprocess = nn.Sequential(
            conv3x3(inp,oup,stride,groups=groups),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            conv1x1(oup,oup,groups=groups),
            nn.BatchNorm2d(oup)
        )
        self.downsample = downsample

    def forward(self, x):
        
        x, a_h_l,a_w_l,if_attn = x
        identity = x
        out = self.preprocess(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out+identity)
        identity = out
        
        b,c,h,w = out.size()
        x_h = self.pool_h(out)
        x_w = self.pool_w(out).permute(0, 1, 3, 2) #extract vertical information

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        
        if a_w_l is not None:
            #print('a_h:'+str(a_h.shape)+'a_h_l'+str(a_h_l.shape))
            a_h = a_h*a_h_l
            a_w = a_w*a_w_l
        out = identity * a_w * a_h
        return out,a_h,a_w,True


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=4, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128 #初始化输出层数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group #每个组为64
        self.conv1 = nn.Conv2d(in_channels = 180, out_channels = self.inplanes, kernel_size=3, stride=1,padding=1,
                               bias=False,groups=1) #（180，128，3，1，1）
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = nn.BatchNorm1d(512+21) #+21
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 128, layers[0],groups=1) #输出层数128
        #？？？
        self.inplanes = int(self.inplanes*1) 
        #？？？
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],groups=1)
        self.inplanes = int(self.inplanes * 1)
        self.conv2 = conv1x1(128,256)
        self.conv3 = conv1x1(256,512)



        self.MLP1 = nn.Linear(112,56)
        self.MLP2 = nn.Linear(56,28)
        self.MLP3 = nn.Linear(28,14)
        self.fc1 = nn.Linear(512*block.expansion*196, 512)
        self.fc2 = nn.Linear(512+21,128) #+21
        self.fc3 = nn.Linear(128,5) 
        self.drop = nn.Dropout(p=0.3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,groups=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,POS,au):##x->input of main branch; POS->position embeddings generated by sub branch
        #[B,180,112,112],[B,512,14,14]
        x = self.conv1(x) #[B,180,112,112] -> [B,120,112,112]
        if(x.size(0)>1):
            x = self.bn1(x)
        x = self.relu(x)
        ##main branch
        x,attn_h,attn_w,_ = self.layer1((x,None,None,True))
        # attn_h = self.MLP1(attn_h.permute(0,1,3,2)).permute(0,1,3,2)
        # attn_w = self.MLP1(attn_w)
        attn_h = self.maxpool(attn_h)
        attn_w = self.maxpool(attn_w)

        x ,attn_h,attn_w,_= self.layer2((x,attn_h,attn_w,True))
        # attn_h = self.MLP2(attn_h.permute(0,1,3,2)).permute(0,1,3,2)
        # attn_w = self.MLP2(attn_w)
        attn_h = self.maxpool(attn_h)
        attn_h = self.conv2(attn_h)
        attn_w = self.maxpool(attn_w)
        attn_w = self.conv2(attn_w)
        #attn2=self.maxpool(attn2)

        x ,attn_h,attn_w,_= self.layer3((x,attn_h,attn_w,True))
        # attn_h = self.MLP3(attn_h.permute(0,1,3,2)).permute(0,1,3,2)
        # attn_w = self.MLP3(attn_w)
        attn_h = self.maxpool(attn_h)
        attn_h = self.conv3(attn_h)
        attn_w = self.maxpool(attn_w)
        attn_w = self.conv3(attn_w)
        #attn3 = self.maxpool(attn3)
        x,attn_h,attn_w,_ = self.layer4((x,attn_h,attn_w,True))
        
        
        x=x+POS#fusion of motion pattern feature and position embeddings

        x = torch.flatten(x, 1) #(32,512*14*14)
        x = self.fc1(x)
        # x = self.drop(x)
        # x = x+au
        x = torch.cat([x,au],dim = 1) 
        if(x.size(0)>1):
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc2(x)
        if(x.size(0)>1):
            x = self.bn3(x)
        # x = self.relu(x)
        x = self.fc3(x)
        return x #后面的貌似是无用代码

    def forward(self, x,POS,au):
        return self._forward_impl(x,POS,au)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

##main branch consisting of CA blocks
def resnet18_pos_attention(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', SOA, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)


# ---------------------------------------------------------------------------
# OrientationPoolingVec – 向量编码版本，梯度更稳定
# ---------------------------------------------------------------------------

class OrientationPoolingVec(nn.Module):
    """Average-pooling along a learnable orientation using 2-D vector encoding.

    与上面的 OrientationPooling 保持接口兼容，但把单一标量 θ
    替换为 2-维向量 (u,v)。通过归一化 + atan2 得到 θ∈(0,π)，
    避免 sigmoid 饱和带来的梯度消失问题。
    """
    def __init__(self, learnable: bool = True, init: float | None = None):
        super().__init__()
        if init is None:
            init = float(torch.rand(1).item() * math.pi)  # 随机 (0,π)
        # 将角度转成向量坐标
        u0, v0 = math.cos(init), math.sin(init)
        vec = torch.tensor([u0, v0], dtype=torch.float32)
        self.vec = nn.Parameter(vec) if learnable else vec  # shape (2,)

    # 可供外部读取当前角度
    def angle(self) -> torch.Tensor:
        vec = F.normalize(self.vec, dim=0)                 # (2,)
        theta = torch.atan2(vec[1], vec[0])                # (-π, π]
        theta = theta + (theta < 0).float() * math.pi      # → (0, π)
        return theta                                        # 标量张量

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B,C,H,W) → (B,C,H,1)
        b, c, h, w = x.shape
        theta = self.angle()                               # scalar tensor
        grid = _rotation_grid(b, h, w, theta.expand(b))    # broadcast θ
        x_rot = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        return x_rot.mean(dim=3, keepdim=True)


if __name__ == "__main__":
    device = torch.device('cuda')
    model = CABlock(128,512).to(device)
