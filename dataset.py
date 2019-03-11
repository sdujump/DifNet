import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from PIL import Image, ImageOps, ImageFilter
import os
import numpy as np
import os.path
import math
import shutil
import torch
import scipy.io

from IPython.core import debugger
debug = debugger.Pdb().set_trace

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class JointRandomFlip(object):
    def __init__(self, rand=True):
        self.rand = rand
    def __call__(self, img, seg):
        randnum = np.random.random()
        if self.rand and randnum < 0.5:
            img = F.hflip(img)
            seg = F.hflip(seg)
        return img, seg 
        

class JointRandomScale(object):
    def __init__(self, opt_size, rand=True):
        self.rand = rand
        self.size = opt_size
    def __call__(self, img, seg):
        if self.rand:
            width, height = img.size
            f_scale = 0.5 + np.random.randint(0, 10) / 10.0            
            width, height = int(width * f_scale), int(height * f_scale)
            img = F.resize(img, (height, width), Image.BILINEAR)
            seg = F.resize(seg, (height, width), Image.NEAREST)
        return img, seg


class JointResize(object):
    def __init__(self, opt_size):
        self.size = opt_size
    def __call__(self, img, seg):
        img = F.resize(img, (self.size, self.size), Image.BILINEAR)
        seg = F.resize(seg, (self.size, self.size), Image.NEAREST)
        return img, seg    


class JointRandomCrop(object):
    def __init__(self, opt_size, rand=True):
        self.rand = rand
        self.size = (int(opt_size), int(opt_size))
    def __call__(self, img, seg):
        w, h = img.size
        th, tw = self.size
        if self.rand:
            i = 0 if h==th else np.random.randint(min(th-h, 0), max(th-h, 0))
            j = 0 if w==tw else np.random.randint(min(tw-w, 0), max(tw-w, 0))
            pad_params = (j, i, tw-w-j, th-h-i)        
            img = F.pad(img, pad_params)
            seg = F.pad(seg, pad_params, fill=0)
        else:
            i = int(round((th - h) / 2.))
            j = int(round((tw - w) / 2.))
            pad_params = (j, i, tw-w-j, th-h-i)        
            img = F.pad(img, pad_params)
            seg = F.pad(seg, pad_params, fill=255)
        return img, seg


class JointRandomRotate(object):
    def __init__(self, rand=True):
        self.rand = rand
    def __call__(self, img, seg):
        if self.rand:
            deg = np.random.uniform(-10, 10)
            img = img.rotate(deg, resample=Image.BILINEAR)
            seg = seg.rotate(deg, resample=Image.NEAREST)
            '''
            if fill_value != 0:
                seg_np = np.asarray(seg)
                seg_np = scipy.ndimage.rotate(seg_np, deg, reshape=False, order=0, mode='constant',cval=fill_value)    ####   pad 0 or -1(255) for mask?      
                seg = Image.fromarray(seg_np)
            else:
                seg = seg.rotate(deg, resample=Image.NEAREST)
            '''
        return img, seg 


class JointRandomBlur(object):
    def __init__(self, rand=True):
        self.rand = rand
    def __call__(self, img, seg):
        #randnum = np.random.random()
        #if self.rand and randnum < 0.5:
        #if np.random.random() and self.rand< 0.5:
        if self.rand:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(0,5)))
        return img, seg


class JointPad(object):
    def __init__(self, opt_size, pad_value=255, switch=True):
        self.pad_value = pad_value
        self.size = (int(opt_size), int(opt_size))
        self.switch = switch
    def __call__(self, img, seg):
        if self.switch:
            lr, tb = tuple(x-y for x, y in zip(self.size, img.size))
            left, right, top, bottom = lr//2, lr - (lr//2), tb//2, tb - (tb//2)
            img = F.pad(img, (left,top,right,bottom))
            seg = F.pad(seg, (left,top,right,bottom), fill=self.pad_value)
        return img, seg


class JointPostProcess(object):
    def __init__(self, norm=True):
        self.norm = norm
    def __call__(self, img, seg):
        img = F.to_tensor(img)
        img = (img*255.0).float()
        seg = F.to_tensor(seg)
        seg = (seg*255.0).long()
        if self.norm:
            # img mode 'BGR'
            img = F.normalize(img, [104.00698793,116.66876762,122.67891434], [1.0,1.0,1.0])
        return img, seg

        
class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(*img)
        return img 


class PadForTest(object):
    def __init__(self, opt_size, pad_value=0):
        self.pad_value = pad_value
        self.size = (int(opt_size), int(opt_size))
    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        i = int(round((th - h) / 2.))
        j = int(round((tw - w) / 2.))
        pad_params = (j, i, tw-w-j, th-h-i)        
        img = F.pad(img, pad_params, self.pad_value)
        #lr, tb = tuple(x-y for x, y in zip(self.size, img.size))
        #left, right, top, bottom = lr//2, lr - (lr//2), tb//2, tb - (tb//2)
        #img = F.pad(img, (left,top,right,bottom))
        return img, pad_params


class PostProcessForTest(object):
    def __init__(self, norm=True):
        self.norm = norm
    def __call__(self, img):
        img = F.to_tensor(img)
        img = (img*255.0).float()
        if self.norm:
            # img mode 'BGR'
            img = F.normalize(img, [104.00698793,116.66876762,122.67891434], [1.0,1.0,1.0])
        return img

    
def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


#because model will be initialized by pretrained caffe model, image should be loaded in BGR mode.
def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if mode == 'BGR':
                img = img.convert('RGB')
                r, g, b = img.split()
                img = Image.merge('RGB', (b, g, r))
            else:
                img = img.convert(mode)
            return img


class ImageFolderForVOC2012(data.Dataset):
    def __init__(self, root, data_list, opt_size, flip=True, scale=True, crop=True, rotate=True, blur=True, test=False):
        self.root = root
        self.test = test
        self.opt_size = opt_size     
        self.img_names = []
        self.lab_names = []
        self.flip = flip
        self.scale = scale
        self.crop = crop
        self.rotate = rotate
        self.blur = blur

        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(self.root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(self.root, i) for i in self.indices]
        self.lab_names = ['{}/SegmentationClassAug/{}.png'.format(self.root, i) for i in self.indices]

    def __getitem__(self, index):
        if not self.test:
            img_name = self.img_names[index]
            lab_name = self.lab_names[index]
            img_ori = pil_loader(img_name, 'BGR')
            lab_ori = pil_loader(lab_name, 'P')
          
            transform = JointCompose([JointRandomFlip(self.flip), 
                                            JointRandomScale(self.opt_size, self.scale), 
                                            JointRandomCrop(self.opt_size, self.crop),
                                            JointRandomRotate(self.rotate),
                                            JointRandomBlur(self.blur),
                                            JointPostProcess(),])
 
            img, lab = transform([img_ori, lab_ori])
            return img, lab, img_name
        else:
            img_name = self.img_names[index]
            img_ori = pil_loader(img_name, 'BGR')
            transformPad = PadForTest(self.opt_size, 0)
            img, pad_size = transformPad(img_ori)

            lab_name = self.lab_names[index]
            if os.path.exists(lab_name):
                lab_ori = pil_loader(lab_name, 'P')
                transformPad = PadForTest(self.opt_size, 255)
                lab, lab_pad_size = transformPad(lab_ori)
                transformPostProcess = JointCompose([JointPostProcess(),])
                img, lab = transformPostProcess([img, lab])
            else:
                transformPostProcess = PostProcessForTest()
                img = transformPostProcess(img)
                lab = False
            return img, lab, pad_size, img_name
    
    def __len__(self):
        return len(self.img_names)
