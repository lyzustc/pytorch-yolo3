import torch
import numpy as np
import cv2
from torchvision import transforms
from cfg import *
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import *
from darknet import Darknet
voc_data = MyDataset('/home/lyzustc/programs/pytorch/data/VOCdevkit/2007_test.txt', shape=(480,480), transform = transforms.ToTensor(), use_da = False, dataset = 'voc', is_train=True)
voc_loader = DataLoader(dataset = voc_data, shuffle = False, batch_size = 1)
voc_iter = iter(voc_loader)

class_names = load_class_names('data/voc.names')
img, label = next(voc_iter)
img_np = (np.transpose(img[0].numpy(),(1,2,0)) * 255.0).astype(np.uint8)
img_np_o = img_np.copy()
img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.imshow('ori1',img_np)
img_gt_np = plot_gt_boxes(img_np.copy(), label[0].view(-1,5), class_names)
cv2.imshow('gt1',img_gt_np)
cv2.waitKey(0)

# class_names = load_class_names('data/coco.names')
# m = Darknet('cfg/yolo_v3.cfg')
# m.load_weights('weights/yolov3.weights')
# conf_thresh = 0.5
# nms_thresh = 0.6
# boxes = do_detect(m, img_np_o, conf_thresh, nms_thresh, len(class_names), use_cuda = False)
# img_det_np = plot_boxes(cv2.cvtColor(img_np_o, cv2.COLOR_RGB2BGR), boxes, class_names)
# cv2.imshow('det1', img_det_np)
# cv2.waitKey(0)