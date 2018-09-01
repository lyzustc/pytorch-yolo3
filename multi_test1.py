from multiprocessing import Value, Process
from multiprocessing.managers import BaseManager as bm
import queue
import time
import argparse
from utils import *
from darknet import Darknet
import torch

def add_send_end():
    send_end.value += 1

def get_send_end():
    l = []
    l.append(send_end.value)
    return l

def add_pro_end():
    pro_end.value += 1

def get_pro_end():
    l = []
    l.append(pro_end.value)
    return l

def process_img(args):
    global m
    ori_q = m.get_ori()
    det_q = m.get_det()

    if args.detect:
        net = Darknet(args.config)
        net.load_weights(args.weights)
        conf_thresh = args.conf_thresh
        nms_thresh = args.nms_thresh
        class_names = load_class_names(args.class_names)
        use_cuda = args.cuda
        if use_cuda:
            net = net.cuda()
    while True:
        if ori_q.empty():
            if send_end_signal.value > 0:
                break
            else:
                continue
        if args.detect:
            frame = ori_q.get()
            sized = cv2.resize(frame, (args.size, args.size))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            boxes = do_detect(net, sized, conf_thresh, nms_thresh, len(class_names), use_cuda)
            frame_det = plot_boxes(frame, boxes, class_names)
            det_q.put(frame_det)
        else:
            img = ori_q.get()
            det_q.put(img)

    m.add_pro_end()
    while True:
        if m.get_pro_end().pop() == 2:
            break

    print('detect frames program exists!')

def super_send_end():
    while True:
        if m.get_send_end().pop() > 0:
            send_end_signal.value = 1
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', type=str, default='weights/yolov3.weights', help='weights file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolo_v3.cfg', help='net configure file')
    parser.add_argument('--cuda', action='store_true', default=False, help='if specified, gpu will be used')
    parser.add_argument('--detect', action='store_true', default=False, help='if specified, frames with detecting bounding box will be outputed')
    parser.add_argument('--save', '-s', type=str, default='video_frames', help='the path to save detection results')
    parser.add_argument('--conf_thresh', '-ct', type=float, default=0.3, help='confidence thresh')
    parser.add_argument('--nms_thresh', '-nt', type=float, default=0.4, help='nms thresh')
    parser.add_argument('--class_names', '-n', type = str, default='data/coco.names', help = 'specify the file which contains the class names')
    parser.add_argument('--size', type=int, default = 416, help = 'the size of image, must be the times of 32')
    args, _ = parser.parse_known_args()

    send_end_signal = Value('i', 0)
    send_end = Value('i', 0)
    pro_end = Value('i', 0)
    ori_q = queue.Queue()
    det_q = queue.Queue()
    bm.register('get_ori', callable = lambda: ori_q)
    bm.register('get_det', callable = lambda: det_q)
    bm.register('add_send_end', callable = add_send_end)
    bm.register('get_send_end', callable = get_send_end)
    bm.register('add_pro_end', callable = add_pro_end)
    bm.register('get_pro_end', callable = get_pro_end)
    m = bm(address=('10.66.30.45',10000),authkey = b'abc')
    m.start()
    
    p1 = Process(target = process_img, args = (args, ))
    p2 = Process(target = super_send_end)
    p1.start()
    p2.start()
    p1.join()
    p2.join()