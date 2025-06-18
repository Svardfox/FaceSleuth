import math
import time
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
# import image_utils
import argparse, random
from functools import partial
from opencv_flow import OpticFlow
from resnet import ResNet,BasicBlock
from CA_block import resnet18_pos_attention
from timm.models import create_model
#from einops import rearrange, repeat
#from einops.layers.torch import Rearrange
from PC_module import VisionTransformer_POS
from swin_transformer import SwinTransformer
from torch.utils.data import ConcatDataset
# from gradcam import show_cam_on_image, GradCam, preprocess_image
#import confusion_matrix
from torchvision.transforms import Resize

torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='../autodl-tmp/CASMEII', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=7000, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, num_loso, transform=None, basic_aug=False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        SUBJECT_COLUMN = 0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6

        df = pd.read_excel(os.path.join(self.raf_path, 'CASME2-coding-20190701.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
        df['Subject'] = df['Subject'].apply(str) #将subject列元素转化为str类型？
        df_AU = pd.read_excel('./AU_info_obj.xlsx')
        df_AU['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject'] != num_loso]
            AU_info = df_AU.loc[df_AU['Subject'] != num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]
            AU_info = df_AU.loc[df_AU['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:,
                    LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = AU_info.iloc[:, 1:].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.sub = []
        self.file_names = []
        self.label_au = []
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        # use aligned images for training/testing
        for (f, sub, onset, apex, offset, label_all, label_au) in zip(File_names, Subject, Onset_num, Apex_num,
                                                                      Offset_num, Label_all, Label_au):

            if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':

                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.sub.append(sub)
                self.file_names.append(f)
                self.label_au.append(label_au)
                if label_all == 'happiness':
                    self.label_all.append(0)
                    a = a + 1
                elif label_all == 'repression':
                    self.label_all.append(1)
                    b = b + 1
                elif label_all == 'disgust':
                    self.label_all.append(2)
                    c = c + 1
                elif label_all == 'surprise':
                    self.label_all.append(3)
                    d = d + 1
                else:
                    self.label_all.append(4)
                    e = e + 1

                # label_au =label_au.split("+")
                # if isinstance(label_au, int):
                #     self.label_au.append([label_au])
                # else:
                #     label_au = label_au.split("+")
                #     self.label_au.append(label_au)

            ##label
        if self.phase == 'train':
            global cls_weights
            cls_weights = []
            for i in [a,b,c,d,e]:
                cls_weight = max(a,b,c,d,e)/i
                cls_weights.append(cls_weight)



        self.basic_aug = basic_aug
        # self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            on0 = str(random.randint(int(onset), int(onset + int(0.15 * (apex - onset) / 4))))
            # on0 = str(int(onset))
            """  
            on1 = str(
                random.randint(int(onset + int(0.9 * (apex - onset) / 4)), int(onset + int(1.1 * (apex - onset) / 4))))
            on2 = str(
                random.randint(int(onset + int(1.8 * (apex - onset) / 4)), int(onset + int(2.2 * (apex - onset) / 4))))
            on3 = str(random.randint(int(onset + int(2.7 * (apex - onset) / 4)), onset + int(3.3 * (apex - onset) / 4)))
            """
            # apex0 = str(apex)
            apex0 = str(
                random.randint(int(apex - int(0.15 * (apex - onset) / 4)), apex))
            
            """ 
            off0 = str(
                random.randint(int(apex + int(0.9 * (offset - apex) / 4)), int(apex + int(1.1 * (offset - apex) / 4))))
            off1 = str(
                random.randint(int(apex + int(1.8 * (offset - apex) / 4)), int(apex + int(2.2 * (offset - apex) / 4))))
            off2 = str(
                random.randint(int(apex + int(2.9 * (offset - apex) / 4)), int(apex + int(3.1 * (offset - apex) / 4))))
            off3 = str(random.randint(int(apex + int(3.8 * (offset - apex) / 4)), offset)) 
            """

            sub = "sub"+str(self.sub[idx]).zfill(2)
            f = str(self.file_names[idx])
        else:  ##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]

            on0 = str(onset)
            """ 
            on1 = str(int(onset + int((apex - onset) / 4)))
            on2 = str(int(onset + int(2 * (apex - onset) / 4)))
            on3 = str(int(onset + int(3 * (apex - onset) / 4))) 
            """
            
            apex0 = str(apex)
            """ 
            off0 = str(int(apex + int((offset - apex) / 4)))
            off1 = str(int(apex + int(2 * (offset - apex) / 4)))
            off2 = str(int(apex + int(3 * (offset - apex) / 4)))
            off3 = str(offset) 
            """

            sub = "sub"+str(self.sub[idx]).zfill(2)
            f = str(self.file_names[idx])

        on0 = 'reg_img' + on0 + '.jpg'
        """ 
        on1 = 'reg_img' + on1 + '.jpg'
        on2 = 'reg_img' + on2 + '.jpg'
        on3 = 'reg_img' + on3 + '.jpg' 
        """
        apex0 = 'reg_img' + apex0 + '.jpg'
        """ 
        off0 = 'reg_img' + off0 + '.jpg'
        off1 = 'reg_img' + off1 + '.jpg'
        off2 = 'reg_img' + off2 + '.jpg'
        off3 = 'reg_img' + off3 + '.jpg' 
        """
        path_on0 = os.path.join(self.raf_path, 'Cropped', sub, f, on0)
        """ path_on1 = os.path.join(self.raf_path, 'Cropped-updated/Cropped/', sub, f, on1)
        path_on2 = os.path.join(self.raf_path, 'Cropped-updated/Cropped/', sub, f, on2)
        path_on3 = os.path.join(self.raf_path, 'Cropped', sub, f, on3) """
        path_apex0 = os.path.join(self.raf_path, 'Cropped', sub, f, apex0)
        """ path_off0 = os.path.join(self.raf_path, 'Cropped', sub, f, off0)
        path_off1 = os.path.join(self.raf_path, 'Cropped-updated/Cropped/', sub, f, off1)
        path_off2 = os.path.join(self.raf_path, 'Cropped-updated/Cropped/', sub, f, off2)
        path_off3 = os.path.join(self.raf_path, 'Cropped-updated/Cropped/', sub, f, off3) """
        image_on0 = cv2.imread(path_on0)
        """ image_on1 = cv2.imread(path_on1)
        image_on2 = cv2.imread(path_on2)
        image_on3 = cv2.imread(path_on3) """
        image_apex0 = cv2.imread(path_apex0)
        """ image_off0 = cv2.imread(path_off0)
        image_off1 = cv2.imread(path_off1)
        image_off2 = cv2.imread(path_off2)
        image_off3 = cv2.imread(path_off3) """
        
        #optical flow
        # pre_gray = cv2.cvtColor(image_on0, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(image_apex0, cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(pre_gray, gray,None,
        #                                0.5, 3, 15, 3, 5, 1.2, 0)
        # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mask = np.zeros_like(image_on0)
  
        # #   Sets image saturation to maximum
        # mask[..., 1] = 255    
        # # Sets image hue according to the optical flow 
        # # direction
        # mask[..., 0] = angle * 180 / np.pi / 2
            
        #     # Sets image value according to the optical flow
        #     # magnitude (normalized)
        # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
        #     # Converts HSV to RGB (BGR) color representation
        # flow = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)   

        image_on0 = image_on0[:, :, ::-1]  # BGR to RGB
        """ mage_on1 = image_on1[:, :, ::-1]
        image_on2 = image_on2[:, :, ::-1]
        image_on3 = image_on3[:, :, ::-1]
        image_off0 = image_off0[:, :, ::-1]
        image_off1 = image_off1[:, :, ::-1]
        image_off2 = image_off2[:, :, ::-1]
        image_off3 = image_off3[:, :, ::-1] """
        image_apex0 = image_apex0[:, :, ::-1]

        label_all = self.label_all[idx]
        label_au = np.float32(self.label_au[idx])
        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)
            """ image_on1 = self.transform(image_on1)
            image_on2 = self.transform(image_on2)
            image_on3 = self.transform(image_on3)
            image_off0 = self.transform(image_off0)
            image_off1 = self.transform(image_off1)
            image_off2 = self.transform(image_off2)
            image_off3 = self.transform(image_off3) """
            image_apex0 = self.transform(image_apex0)
            ALL = torch.cat(
                (image_on0, image_apex0), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]
            """ image_on1 = ALL[3:6, :, :]
            image_on2 = ALL[6:9, :, :]
            image_on3 = ALL[9:12, :, :] """
            image_apex0 = ALL[3:6, :, :]
            """ image_off0 = ALL[15:18, :, :]
            image_off1 = ALL[18:21, :, :]
            image_off2 = ALL[21:24, :, :]
            image_off3 = ALL[24:27, :, :] """
            """ temp = torch.zeros(38)
            for i in label_au:
                temp[int(i) - 1] = 1 """

            return image_on0,image_apex0,label_all,label_au


def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def criterion2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=180, kernel_size=3, stride=2, padding=1, bias=False, groups=1), 
            #[B,3,224,224]->[B,180,112,112]
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),

        )
        self.pos = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

        )
        ##Position Calibration Module(subbranch)
        # self.vit_pos = VisionTransformer_POS(img_size=14,
        #                                      patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4,
        #                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                      drop_path_rate=0.)
        
        # self.vit_pos_224 = VisionTransformer_POS(img_size=224,
        #                                      patch_size=16, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4,
        #                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                      drop_path_rate=0.)

        self.swin_pos_224 = SwinTransformer(img_size=224, patch_size=4, embed_dim=128, depths=[2, 2, 6],
                                             num_heads=[4, 8, 16])
        
        #self.resnet = ResNet(BasicBlock,[3,4,6],include_top=False)

        self.resize = Resize([14, 14])
        ##main branch consisting of CA blocks
        self.main_branch = resnet18_pos_attention()

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 112 * 112, 38, bias=False),

        )

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x5, au):
        ##onset:x1 apex:x5
        B = x1.shape[0] #BatchSize
        # Position Calibration Module (subbranch)
        # POS = self.vit_pos(self.resize(x1)).transpose(1, 2).view(B, 512, 14, 14) #[B,512,14,14]
        # POS = self.vit_pos_224(x1).transpose(1,2).view(B,512,14,14)
        x = torch.cat([x1,x5],dim=0)
        Swin = self.swin_pos_224(x).transpose(1, 2).view(2*B, 512, 14, 14)
        Swin_Onset,Swin_Apex = torch.split(Swin,B,dim=0)
        POS = Swin_Apex + Swin_Onset
        act = x5 - x1
        act = self.conv_act(act)
        # flow = self.conv_act(flow)
        # main branch and fusion
        out= self.main_branch(act,POS,au) #[B,180,112,112],[B,512,14,14]

        return out


def run_training():
    args = parse_args()
    imagenet_pretrained = True

    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict=False)
    ##data normalization for both training set
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])
    ### data augmentation for training set only
    data_transforms_norm = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(3),
        transforms.RandomCrop(224, padding=15),

    ])

    ### data normalization for both teating set
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # leave one subject out protocal
    LOSO = ['2','16','9','20','5','3', '19', '1', '18', '10', '21', '22', '15', '6', '25', '7',
            '17', '26', '24',  '13', '4', '23', '11', '12', '8', '14', ]

    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(5)
    pos_label_ALL = torch.zeros(5)
    TP_ALL = torch.zeros(5)
    record = pd.DataFrame()
    idx = 0

    for subj in LOSO:
        train_dataset = RafDataSet(args.raf_path, phase='train', num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
        val_dataset = RafDataSet(args.raf_path, phase='test', num_loso=subj, transform=data_transforms_val)
        train_loader = MultiEpochsDataLoader(train_dataset,
                                                   batch_size=30,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   )
        val_loader = MultiEpochsDataLoader(val_dataset,
                                                 batch_size=30,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 )
        global cls_weights
        cls_weights = torch.FloatTensor(cls_weights).cuda()
        criterion = torch.nn.CrossEntropyLoss()

        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(5)
        max_pos_label = torch.zeros(5)
        max_TP = torch.zeros(5)
        ##model initialization
        net_all = MMNet()
        flag = True #跳出循环标志

        params_all = net_all.parameters()

        if args.optimizer == 'adam':
            optimizer_all = torch.optim.AdamW(params_all, lr=0.0008, weight_decay=0.6)
            ##optimizer for MMNet

        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, args.lr,
                                        momentum=args.momentum,
                                        weight_decay=1e-4)
        else:
            raise ValueError("Optimizer not supported.")
        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)

        net_all = net_all.cuda()
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        for i in range(1, 75):
            if not flag:
                break
            running_loss = 0.0
            correct_sum = 0
            running_loss_MASK = 0.0
            correct_sum_MASK = 0
            iter_cnt = 0
            accum_iter = 2
            net_all.train()
            
            sinceT = time.time()
            for batch_i, (
                    image_on0, image_apex0, label_all,label_au) in enumerate(train_loader):
                batch_sz = image_on0.size(0)
                b, c, h, w = image_on0.shape
                iter_cnt += 1
                image_on0 = image_on0.cuda()
                """ image_on1 = image_on1.cuda()
                image_on2 = image_on2.cuda()
                image_on3 = image_on3.cuda() """
                image_apex0 = image_apex0.cuda()
                """ image_off0 = image_off0.cuda()
                image_off1 = image_off1.cuda()
                image_off2 = image_off2.cuda()
                image_off3 = image_off3.cuda() """
                label_all = label_all.cuda()
                label_au = label_au.cuda()
                data_time = time.time() - sinceT
                starter.record()
                ##train MMNet
                ALL = net_all(image_on0, image_apex0, label_au)

                loss_all = criterion(ALL, label_all)
                loss_all = loss_all
                loss_all.backward()
                torch.cuda.synchronize() # 等待GPU任务完成
                ender.record()

                # if ((batch_i + 1) % accum_iter == 0) or (batch_i + 1 == len(train_loader)):
                optimizer_all.step()
                optimizer_all.zero_grad()
                    
                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all).sum()
                correct_sum += correct_num

            ## lr decay
            if i <= 50:
                scheduler_all.step()
            if i >= 0:
                acc = correct_sum.float() / float(train_dataset.__len__())

                running_loss = running_loss / iter_cnt
                curr_time = starter.elapsed_time(ender)
                total_time = time.time() - sinceT
                print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f Inference time: %.4f(py),Data time:%.4f Total time: %.4f' % 
                      (i, acc, running_loss,curr_time/1000,data_time,total_time))

            pos_label = torch.zeros(5)
            pos_pred = torch.zeros(5)
            TP = torch.zeros(5)

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                net_all.eval()
                # net_au.eval()
                for batch_i, (
                        image_on0, image_apex0,label_all,label_au) in enumerate(val_loader):
                    batch_sz = image_on0.size(0)
                    b, c, h, w = image_on0.shape

                    image_on0 = image_on0.cuda()
                    """ image_on1 = image_on1.cuda()
                    image_on2 = image_on2.cuda()
                    image_on3 = image_on3.cuda() """
                    image_apex0 = image_apex0.cuda()
                    """ image_off0 = image_off0.cuda()
                    image_off1 = image_off1.cuda()
                    image_off2 = image_off2.cuda()
                    image_off3 = image_off3.cuda() 
                    label_au = label_au.cuda()"""
                    label_all = label_all.cuda()
                    label_au = label_au.cuda()

                    ##test
                    ALL = net_all(image_on0,image_apex0,label_au)

                    loss = criterion(ALL, label_all)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(ALL, 1)
                    correct_num = torch.eq(predicts, label_all)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += ALL.size(0)

                    for cls in range(5):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in label_all:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, label_all):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1
                        # if pos_label != 0 or pos_pred != 0:
                        #     f1 = 2 * TP / (pos_pred + pos_label)
                        #     F1.append(f1)
                    count = 0
                    SUM_F1 = 0
                    for index in range(5):
                        if pos_label[index] != 0 or pos_pred[index] != 0:
                            count = count + 1
                            SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])

                    AVG_F1 = SUM_F1 / count

                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (i, acc, running_loss, AVG_F1))
                if acc == 1:
                    flag = False

        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        for index in range(5):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        print("[..........%s] correctnum:%d . zongshu:%d   " % (subj, max_corr, val_dataset.__len__()))
        print("[ALL_corr]: %d [ALL_val]: %d" % (num_sum, val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))
        df = pd.DataFrame({'LOSO_num':subj,'Correct_now':max_corr,"Total":val_dataset.__len__(),
                           'F1':max_f1,'Correct_all':num_sum,'Val_all':val_now,'F1_all':F1_ALL},index=[idx])
        record = pd.concat([record,df])
        idx += 1
    return record


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

if __name__ == "__main__":
    for i in range(3):
        record = run_training()
        localtime = time.asctime( time.localtime(time.time()) )
        record.to_excel('./training_record/'+str(i)+'.xlsx')

