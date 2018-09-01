import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import itertools
import struct # get_image_size
import imghdr # get_image_size
import random
import cv2

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
        x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
        y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
        y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea/uarea)

def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if boxes.shape[0] == 0:
        return boxes

    det_confs = boxes[:,4] * boxes[:,5]               

    _, sortIds = torch.sort(det_confs, descending = True)
    boxes = boxes[sortIds]
    ind_mask = torch.ones(sortIds.shape).float()
    for i in range(boxes.shape[0] - 1):
        tmp_ind = torch.nonzero(ind_mask[i+1:]).squeeze(0)
        if tmp_ind.numel() != 0:
            tmp_boxes = boxes[tmp_ind + i + 1].view(-1, boxes.shape[1])
            ious = multi_bbox_ious(boxes[i].repeat(tmp_boxes.shape).t(), tmp_boxes.t(), x1y1x2y2 = False)
            tmp_ind = tmp_ind[ious > nms_thresh]
            ind_mask[i+1:][tmp_ind] = 0
    
    masked_inds = torch.nonzero(ind_mask).squeeze()
    return boxes[masked_inds].view(-1,7)

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_all_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False, use_cuda=True):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    tot = output[0]['x'].data.size(0)
    all_boxes = [[] for i in range(tot)]
    for i in range(len(output)):
        pred, anchors, num_anchors = output[i]['x'].data, output[i]['a'], output[i]['n'].item()
        b = get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)
        for t in range(tot):
            all_boxes[t].append(b[t])
    out_boxes = []
    for boxes in all_boxes:
        boxes = [box for box in boxes if len(box) != 0]
        if len(boxes) > 0:
            out_boxes.append(torch.cat(tuple(boxes),dim = 0))
        else:
            out_boxes.append(torch.Tensor([]))
        
    return out_boxes

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    anchors = anchors.to(device)
    anchor_step = anchors.size(0)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch*num_anchors*h*w

    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, cls_anchor_dim)

    grid_x = torch.linspace(0, w-1, w).repeat(batch*num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(cls_anchor_dim).to(device)
    ix = torch.LongTensor(range(0,2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(1, batch, h*w).view(cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(1, batch, h*w).view(cls_anchor_dim)

    xs, ys = torch.sigmoid(output[0]) + grid_x, torch.sigmoid(output[1]) + grid_y
    ws, hs = torch.exp(output[2]) * anchor_w.detach(), torch.exp(output[3]) * anchor_h.detach()
    det_confs = torch.sigmoid(output[4])

    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.sigmoid(output[5:5+num_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1).float()
    cls_max_ids = cls_max_ids.view(-1).float()
    
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs, ys = convert2cpu(xs), convert2cpu(ys)
    ws, hs = convert2cpu(ws), convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
        
    xs,ys = xs.view(batch,-1),ys.view(batch,-1)
    ws,hs = ws.view(batch,-1),hs.view(batch,-1)
    det_confs = det_confs.view(batch,-1)
    cls_max_confs = cls_max_confs.view(batch,-1)
    cls_max_ids = cls_max_ids.view(batch,-1).float()
    
    for b in range(batch):
        confs = det_confs[b,:]
        inds = (confs > conf_thresh)
        if torch.nonzero(inds).numel() == 0:
            all_boxes.append([])
            continue
        bcxs = xs[b,inds]/w
        bcys = ys[b,inds]/h
        bws = ws[b,inds]/w
        bhs = hs[b,inds]/h
        bc_det_confs = det_confs[b,inds]
        bc_cls_max_confs = cls_max_confs[b,inds]
        bc_cls_max_id = cls_max_ids[b,inds]
        
        
        boxes = torch.stack((bcxs, bcys, bws, bhs, bc_det_confs, bc_cls_max_confs, bc_cls_max_id),dim = 0)
        boxes = boxes.transpose(0,1).view(-1,7)

        all_boxes.append(boxes)
    return all_boxes

def get_color(c, x, max_val):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    ratio = float(x)/max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return int(r*255)

def plot_boxes(img, boxes, class_names=None, color=None):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        if box[3] < 0.01 or box[4] < 0.01:
            break
        x1 = int((box[0] - box[2]/2.0) * width)
        y1 = int((box[1] - box[3]/2.0) * height)
        x2 = int((box[0] + box[2]/2.0) * width)
        y2 = int((box[1] + box[3]/2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (0, 0, 255)
        # if len(box) >= 7 and class_names:
        #     cls_id = box[6].int()
        #     #print('%s: %f' % (class_names[cls_id], cls_conf))
        #     classes = len(class_names)
        #     offset = cls_id * 123457 % classes
        #     red   = get_color(2, offset, classes)
        #     green = get_color(1, offset, classes)
        #     blue  = get_color(0, offset, classes)
        #     if color is None:
        #         rgb = (red, green, blue)
        #     img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 3)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 3)
    
    return img

def plot_gt_boxes(img, boxes, class_names=None, color=None):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        if box[3] < 0.01 or box[4] < 0.01:
            break
        x1 = int((box[1] - box[3]/2.0) * width)
        y1 = int((box[2] - box[4]/2.0) * height)
        x2 = int((box[1] + box[3]/2.0) * width)
        y2 = int((box[2] + box[4]/2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 5 and class_names:
            cls_id = box[0].int()
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 3)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 3)
    
    return img

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r', encoding='utf8') as fp:
        lines = fp.readlines()
    for line in lines:
        class_names.append(line.strip())
    return class_names

def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img

import types
def do_detect(model, img, conf_thresh, nms_thresh, num_classes, use_cuda=True):
    model.eval()
    img = image2torch(img)

    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    out_boxes = model(img)
    boxes = get_all_boxes(out_boxes, conf_thresh, num_classes, use_cuda=use_cuda)[0]
    boxes = nms(boxes, nms_thresh)
    return boxes

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def adjust_learning_rate(lr, processed_batches, steps, scales):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for i in range(len(steps)):
        if processed_batches == steps[i]:
                break
    for j in range(0, i + 1):
        lr = lr * scales[j]

    return lr

def get_learning_rate(init_lr, processed_batches, burn_in, steps, scales):
    
    lr = init_lr
    if processed_batches < burn_in:
        lr = warm_up(processed_batches, burn_in, lr)
    else: 
        for i in range(len(steps)):
            if processed_batches >= steps[i]:
                lr = lr * scales[i]

    return lr

def warm_up(iteration, burn_in, learning_rate):
    lr = learning_rate * pow((iteration + 1)/burn_in, 4)
    return lr

def get_shape():
    width = (random.randint(0,9) + 10)*32
    shape = (width, width)

    return shape