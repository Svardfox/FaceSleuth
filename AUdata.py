import pandas as pd
import os
import torch
import torch.nn as nn

df1 = pd.read_excel(('F:\dataset\CASMEII\CASME2_preprocessed_Li Xiaobai\CASME2-coding-20190701.xlsx'), usecols=[0, 1, 3, 4])
df1['Subject'] = df1['Subject'].apply(str)
Subject = df1.iloc[:,0].values
File_names = df1.iloc[:,1].values
Onset_num = df1.iloc[:, 2].values
Apex_num = df1.iloc[:, 3].values
df_AU = pd.DataFrame()
idx = 0

for(f,sub,onset,offset) in zip(File_names,Subject,Onset_num,Apex_num):
    filename = str(f)+".csv"
    sub = str(sub).zfill(2)
    sub_ = "sub"+sub
    df2 = pd.read_csv(os.path.join('F:\dataset\CASMEII_AU',sub_,filename))
    AU_info = df2.iloc[[int(onset)+1],5:]
    df_file = pd.DataFrame({'Subject':sub,'Filename':str(f)},index = [idx])
    AU_info.reset_index(drop=True, inplace=True)
    df_file.reset_index(drop=True, inplace=True)
    AU_info = pd.concat([df_file,AU_info],axis=1)
    df_AU = pd.concat([df_AU,AU_info],axis=0)
    idx += 1

df_AU.to_excel('AU_info.xlsx',index=False)
print('work done')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4, stride=1, downsample=None, groups=1):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )
        self.Sigmoid = nn.Sigmoid()
        self.conv1 = conv3x3(in_channels, out_channels, stride,groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channels, out_channels,groups=groups)
        self.downsample = downsample
        self.conv3 = conv1x1(2,1)


    def forward(self, x):
        x, attn_last, if_attn = x
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out+identity)
        identity = out

        b, c, h, w = out.shape
        x_permute = out.permute(0, 2, 3, 1).view(b, -1, c)  #[B,H*W,C]
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x_channel_att = self.Sigmoid(x_channel_att)

        out = out * x_channel_att

        x_spatial_att = self.spatial_attention(out).sigmoid()
        out = out * x_spatial_att
        avg_out = torch.mean(out,dim=1,keepdim=True)
        max_out,_ = torch.max(out, dim=1, keepdim=True)
        attn = torch.cat((avg_out,max),dim=1)
        attn = self.conv3(attn)

        if attn_last is not None:
            attn = attn_last * attn

        attn = attn.repeat(1, self.planes, 1, 1) 
        if if_attn:
            out = identity+attn

        return out, attn[:,0,:,:].unsqueeze(1),True