# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
#
# Code is adapted from : https://github.com/layumi/Person_reID_baseline_pytorch
# ------------------------------

import os
import cv2
import math
import time
import yaml
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from PIL import Image

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

def load_network(network, weights):
    network.load_state_dict(torch.load(weights))
    return network

def fliplr(img, device):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().to(device=device)  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def image_loader(loader, image_name, fromcv2=False):
    if fromcv2:
        image = cv2.resize(image_name, (64,128)) 
        image = Image.fromarray(image_name) # Convert to PIL
    else:
        image = Image.open(image_name)
    image = loader(image).float()
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image

class PersonReid:
    def __init__(self, network_config, weights, threshold, device='cpu', verbose=False):
        self.device = torch.device(device)
        self.verbose = verbose
        self.threshold = threshold
        config_path = network_config
        with open(config_path, 'r') as stream:
                self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        model_structure = ft_net(class_num=self.config['nclasses'], droprate=0.5, stride=self.config['stride']).to(device=self.device)
        self.model = load_network(model_structure, weights)
        self.model.classifier.classifier = nn.Sequential()
        self.model.eval()
        self.transforms = transforms.Compose([
                transforms.Resize((256,128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def extract_features(self, images):
        '''
        Method to extract features by the ReID model
        '''
        features = torch.FloatTensor().to(device=self.device)
        for image in images:
            try:
                img = image_loader(self.transforms, image, True).to(device=self.device)
            except:
                continue
            ff = torch.FloatTensor(1, 512).zero_().to(device=self.device)
            for i in range(2):
                if (i == 1):
                    img = fliplr(img, self.device)
                input_img = Variable(img).to(device=self.device)
                output = self.model(input_img)
                ff += output
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff.data.to(device=self.device)), 0)
        return features
    
    def reid(self, qfeat, gfeat):
        '''
        Method to generate ReID score
        '''
        qScore_idx = {}
        for qidx in range(len(qfeat)):
            qf = qfeat[qidx]
            query = qf.view(-1,1)
            score = torch.mm(gfeat, query)
            score = score.squeeze(1).cpu()
            score = score.numpy()
            
            index = np.argsort(score)
            index = index[::-1]
            best_gindex = None
            best_gscore = 0.00
            # Use index[0] here to get the highest scoring index after argsort
            if score[index[0]] > self.threshold:
                qScore_idx[qidx] = index[0]
                best_gindex = index[0]
                best_gscore = score[index[0]]
            
            if self.verbose:
                outtxt = f'Query[{qidx}]\n'
                for sidx in range(len(score)):
                    outtxt += f' Gallery[{sidx}]: ({score[sidx]:.2f})'
                print(outtxt)
                print(f'Best Gallery Index [{best_gindex}], Score ({best_gscore:.2f})\n')
        return qScore_idx

# FOR DEBUG
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(prog='PersonReID.py')
#     parser.add_argument('-dc', '--disable-cuda', action='store_true', help='Flag to disable CUDA')
#     parser.add_argument('-qv', '--qvideos-path', type=str, default='video', help='Path to query videos')
#     parser.add_argument('-gv', '--gvideos-path', type=str, default='video', help='Path to gallery videos')
#     parser.add_argument('-s', '--save-vid', action='store_true', help='Save output videos')
#     args = parser.parse_args()

#     if not args.disable_cuda and torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     # Primary vid
#     qvid_ls = os.listdir(args.qvideos_path)
#     qvids = []
#     for filename in qvid_ls:
#         img = cv2.imread(os.path.join(args.qvideos_path, filename))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         qvids.append(img)

#     # Sec vid
#     gvid_ls = os.listdir(args.gvideos_path)
#     gvids = []
#     for filename in gvid_ls:
#         img = cv2.imread(os.path.join(args.gvideos_path, filename))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         gvids.append(img)

#     st = time.time()
#     pr = PersonReid(device)
#     with torch.no_grad():
#         qfeatures = pr.extract_features(qvids)
#         gfeatures = pr.extract_features(gvids)
#     print(qfeatures.dtype)
#     print(gfeatures.dtype)
#     et = time.time()
#     print(f'Time: {et - st}')

#     qf = qfeatures[0]
#     query = qf.view(-1,1)
#     print(query.shape)
#     score = torch.mm(gfeatures, query)
#     score = score.squeeze(1).cpu()
#     score = score.numpy()
#     print(f'score: {score}')

#     index = np.argsort(score)
#     print(f'presort index: {index}')
#     index = index[::-1]
#     print(f'index: {index}')