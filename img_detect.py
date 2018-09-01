import sys
import time
import cv2
import numpy as numpy
from utils import *
from darknet import Darknet
import argparse

def detect_img(cfgfile, weightfile, imgfile, classnames_file, img_size, save_pth, conf_thresh, nms_thresh, label_path = None, use_cuda = False, dis=False):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    
    if use_cuda:
        m.cuda()

    class_names = load_class_names(classnames_file)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img.copy(), (args.size, args.size))
    if dis and label_path is None:
        cv2.imshow('origin_image', sized)
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    start = time.time()
    boxes = do_detect(m, sized, conf_thresh, nms_thresh, len(class_names), use_cuda)
    end = time.time()
    print('time {}s'.format(end-start))
    
    sized = cv2.cvtColor(sized, cv2.COLOR_RGB2BGR)
    img_det = plot_boxes(sized.copy(), boxes, class_names)
    if dis:
        cv2.imshow('detect_image', img_det)
        
    if label_path is not None:
        truths = np.loadtxt(label_path)
        gt_boxes = torch.from_numpy(truths.reshape(truths.size // 5, 5))
        img_gt = plot_gt_boxes(sized.copy(), gt_boxes, class_names)
        if dis:
            cv2.imshow('gt_image', img_gt)
    
    cv2.waitKey(0)

    if save_pth is not None:
        print("save plot results to %s" % save_pth)
        cv2.imwrite(save_pth, img_det)
    print('detection finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', type=str, default='images', help='specify the image path or directory')
    parser.add_argument('--weights', '-w', type=str, default='weights/yolo_v3.weights', help='weights file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolo_v3.cfg', help='net configure file')
    parser.add_argument('--save', '-s', type = str, default=None, help = 'save results path')
    parser.add_argument('--class_names', '-n', type = str, default='data/voc.names', help = 'specify the file which contains the class names')
    parser.add_argument('--size', type=int, default = 416, help = 'the size of image, must be the times of 32')
    parser.add_argument('--cuda', action='store_true', default=False, help='if specified, gpu will be used')
    parser.add_argument('--display', '-d', action='store_true', default=False, help='if specified, display images on the window')
    parser.add_argument('--conf_thresh', '-ct', type=float, default=0.3, help='confidence thresh')
    parser.add_argument('--nms_thresh', '-nt', type=float, default=0.4, help='nms thresh')
    parser.add_argument('--label_path', '-l', type=str, default=None, help='if specified, ground truth boxes will be drawed')
    args, _ = parser.parse_known_args()
    detect_img(args.config, args.weights, args.img, args.class_names, args.size, args.save, args.conf_thresh, args.nms_thresh, args.label_path, args.cuda, args.display)