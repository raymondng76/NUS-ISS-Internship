# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
#
# Code is adapted from : https://github.com/KaiyangZhou/deep-person-reid
# ------------------------------

import os
import cv2
import time
import torch
import numpy as np
import torchreid
from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image

def image_loader(loader, image_name, fromcv2=False):
    if fromcv2:
        image = cv2.resize(image_name, (128,256)) 
        image = Image.fromarray(image_name) # Convert to PIL
    else:
        image = Image.open(image_name)
    image = loader(image).float()
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image

def fliplr(img, device):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().to(device=device)  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class DeepPersonReID:
    def __init__(self, model, weights_path, threshold, device='cpu', verbose=False):
        self.device = torch.device(device)
        self.verbose = verbose
        self.threshold = threshold
        self.model = torchreid.models.build_model(name=model, num_classes=1041).to(self.device)
        torchreid.utils.load_pretrained_weights(self.model, weights_path)
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
            img = image_loader(self.transforms, image, True).to(device=self.device)
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
            best_gscore = 0.0
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
    # parser = argparse.ArgumentParser(prog='DeepPersonReID.py')
    # parser.add_argument('-qv', '--qvideos-path', type=str, default='video', help='Path to query videos')
    # parser.add_argument('-gv', '--gvideos-path', type=str, default='video', help='Path to gallery videos')
    # parser.add_argument('-s', '--save-vid', action='store_true', help='Save output videos')
    # args = parser.parse_args()

    # img = Image.open('35.png')
    # dpReID = DeepPersonReID(os.path.join('model','osnet_ain_x1_0_mars_softmax_cosinelr','model.pth.tar-150'), torch.device('cuda'))
    # print(dpReID.extract_features(['35.png']))
