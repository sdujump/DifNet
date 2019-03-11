import os
import sys
import time
import shutil
import random
import numpy as np
import torchnet as tnt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.utils import data
from datetime import datetime
from tqdm import tqdm
import socket

import model_difnet as model_dif
import model_nodif as model_nodif
import basic_function as func
import dataset as CustomDataset

from IPython.core import debugger
debug = debugger.Pdb().set_trace

parserWarpper = func.MyArgumentParser()
parser = parserWarpper.get_parser()
args = parser.parse_args()
print [item for item in args.__dict__.items()]

opt_manualSeed = 1000
print("Random Seed: ", opt_manualSeed)
np.random.seed(opt_manualSeed)
random.seed(opt_manualSeed)
torch.manual_seed(opt_manualSeed)
torch.cuda.manual_seed_all(opt_manualSeed)

#cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


class Trainer():
    def __init__(self, args):
        self.args = args
        self.date = datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()
        self.best_pred = 0

        if args.dataset == 'VOC2012':
            train_dataset=CustomDataset.ImageFolderForVOC2012(args.dataset_path, 'train.txt', 321, flip=True, scale=True, crop=True, rotate=False, blur=False)
            val_dataset=CustomDataset.ImageFolderForVOC2012(args.dataset_path, 'val.txt', 505, flip=False, scale=False, crop=False, rotate=False, blur=False)

        self.train_loader = data.DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        self.val_loader = data.DataLoader(val_dataset, num_workers=args.workers, batch_size=int(args.batchsize/2), shuffle=False, pin_memory=True)

        resnet_sed = models.__dict__['resnet' + str(args.layers)](pretrained=True)
        resnet_dif = models.__dict__['resnet' + str(args.layers)](pretrained=True)
        if args.model_type == 'dif':
            self.model = model_dif.Res_Deeplab(num_classes=args.numclasses, layers=args.layers)
            self.model.model_sed = func.param_restore(self.model.model_sed, resnet_sed.state_dict())
            self.model.model_dif = func.param_restore(self.model.model_dif, resnet_dif.state_dict())
        elif args.model_type == 'nodif':
            self.model = model_nodif.Res_Deeplab(num_classes=args.numclasses, layers=args.layers)
            self.model = func.param_restore(self.model, resnet_sed.state_dict())    

        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        max_step = args.epochs * len(self.train_loader)
        self.optimizer = optim.SGD(self.model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 10 * ((1.0-float(step)/max_step)**0.9))
        self.criterion = func.SegLoss(255)


    def train(self, epoch):
        self.model.train()
        losses = func.AverageMeter()
        tbar = tqdm(self.train_loader)
        for i, batch in enumerate(tbar):
            cur_lr = self.scheduler.get_lr()[0]
            img, gt, img_name = batch

            batch_size = img.size()[0]
            input_size = img.size()[2:4]
            
            img_v = img.cuda(non_blocking=True)
            gt_v = gt.cuda(non_blocking=True)

            if self.args.model_type == 'dif':
                mask, seed, pred = self.model(img_v)
            else:
                pred = self.model(img_v)

            pred_sg_up = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=False)

            loss_sg = self.criterion(pred_sg_up, gt_v.squeeze(1))
            loss = loss_sg
            losses.update(loss.item(), img.size(0))

            self.scheduler.step()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train [{0}] Loss {loss.val:.3f} {loss.avg:.3f} Lr {lr:.5f} Best {best:.4f}'.format(epoch, loss=losses, lr=cur_lr, best=self.best_pred))


    def validate_tnt(self, epoch):
        confusion_meter = tnt.meter.ConfusionMeter(self.args.numclasses, normalized=False)
        losses = func.AverageMeter()
        tbar = tqdm(self.val_loader)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tbar):
                img, gt, img_path = batch

                batch_size = img.size()[0]
                input_size = img.size()[2:4]
                
                img_v = img.cuda(non_blocking=True)
                gt_v = gt.cuda(non_blocking=True)

                if self.args.model_type == 'dif':
                    mask, seed, pred = self.model(img_v)
                else:
                    pred = self.model(img_v)

                pred_sg_up = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=False)

                loss = self.criterion(pred_sg_up, gt_v.squeeze(1))
                
                valid_pixel = gt.ne(255)
                pred_sg_up_label = torch.max(pred_sg_up, 1, keepdim=True)[1]
                
                confusion_meter.add(pred_sg_up_label[valid_pixel], gt[valid_pixel])
                losses.update(loss.item(), img.size(0))
                tbar.set_description('Valid [{0}] Loss {loss.val:.3f} {loss.avg:.3f}'.format(epoch, loss=losses))

            if self.args.model_type == 'dif':
                print(self.model.module.get_alpha())
                print(self.model.module.get_beta())

            confusion_matrix = confusion_meter.value()
            inter = np.diag(confusion_matrix)
            union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

            mean_iou_ind = inter/union
            mean_iou_all = mean_iou_ind.mean()
            mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())
            print(' * IOU_All {iou}'.format(iou=mean_iou_all))
            print(' * IOU_Ind {iou}'.format(iou=mean_iou_ind))
            print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix))

        return mean_iou_all, mean_iou_ind, mean_acc_pix


trainer = Trainer(args)

for epoch in range(args.epochs):

    # train and validate
    trainer.train(epoch)
    iou_all, iou_ind, acc_pix = trainer.validate_tnt(epoch)

    # save checkpoint
    is_best = iou_all > trainer.best_pred
    trainer.best_pred = iou_all if is_best else trainer.best_pred

    if args.model_type == 'dif':
        func.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': trainer.model.state_dict(),
            'best_pred': (iou_all, iou_ind, acc_pix),
            'alpha': trainer.model.module.get_alpha(),
            'beta': trainer.model.module.get_beta(),
            'optimizer': trainer.optimizer.state_dict(),
        }, trainer.date, is_best)
    else:
        func.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': trainer.model.state_dict(),
            'best_pred': (iou_all, iou_ind, acc_pix),
            'optimizer': trainer.optimizer.state_dict(),
        }, trainer.date, is_best)
