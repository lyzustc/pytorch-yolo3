import torch
from darknet import Darknet
from dataset import MyDataset
from torchvision import datasets, transforms
from utils import get_all_boxes, bbox_iou, nms, get_image_size, read_data_cfg, load_class_names
import os
import time
import argparse

def valid(datacfg, cfgfile, weightfile, save_path, use_cuda = False, size = 416):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list = options['names']
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    prefix = save_path
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(cfgfile)

    m.load_weights(weightfile)
    num_classes = len(names)

    if use_cuda:
        m.cuda()
    m.eval()

    valid_dataset = MyDataset(valid_images, shape=(size, size),
                       is_train = False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 10
    assert(valid_batchsize > 1)
    
    if use_cuda:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    fps = [0]*num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(num_classes):
        buf = '%s/%s.txt' % (prefix, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.01
    nms_thresh = 0.5
    for batch_id, (data, target) in enumerate(valid_loader):
        if use_cuda:
            data = data.cuda()
        print('start processing batch{}'.format(batch_id))
        start1 = time.time()
        output = m(data)
        batch_boxes = get_all_boxes(output, conf_thresh, num_classes, only_objectness=0, validation=True, use_cuda = use_cuda)
        for i in range(data.size(0)):
            lineId = lineId + 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = get_image_size(valid_files[lineId])
            boxes = batch_boxes[i]
            if boxes.numel() == 0:
                continue
            for cls_id in range(num_classes):
                cls_ind = (boxes[:, 6] == cls_id)
                cls_boxes = nms(boxes[cls_ind],nms_thresh)
                if cls_boxes.numel == 0:
                    continue
                for box in cls_boxes:
                    x1 = (box[0] - box[2]/2.0) * width
                    y1 = (box[1] - box[3]/2.0) * height
                    x2 = (box[0] + box[2]/2.0) * width
                    y2 = (box[1] + box[3]/2.0) * height 
                    fps[cls_id].write('%s %f %f %f %f %f\n' %(fileId, box[4] * box[5], x1, y1, x2, y2))
        end1 = time.time()
        print('average time {}s'.format((end1 - start1) / len(data)))
        del data,target
    for i in range(num_classes):
        fps[i].close()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cfg/voc.data', help='data definition file')
    parser.add_argument('--weights', '-w', type=str, default='weights/yolo_v3.weights', help='weights file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolo_v3.cfg', help='net configure file')
    parser.add_argument('--save', '-s', type = str, default='results', help = 'save results path')
    parser.add_argument('--size', type=int, default = 416, help = 'the size of image, must be the times of 32')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda: True, else: False')
    args, _ = parser.parse_known_args()
    valid(args.data, args.config, args.weights, args.save, args.cuda, args.size)