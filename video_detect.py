import torch
import torch.nn as nn
import cv2
import argparse
from utils import *
from darknet import Darknet
import time

def detect_video(cfgfile, weightfile, videofile, classnames_file, frame_size, save_pth, conf_thresh, nms_thresh, use_cuda=False, dis=False):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    
    if use_cuda:
        m.cuda()
    class_names = load_class_names(classnames_file)

    if videofile is not None:
        cap = cv2.VideoCapture(videofile)
    else:
        cap = cv2.VideoCapture(0)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if save_pth is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_det = cv2.VideoWriter(save_pth, fourcc, 20, size)
    print('begin video detecting')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if dis:
                cv2.imshow("origin_img", frame)

            sized = cv2.resize(frame, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            start = time.time()
            boxes = do_detect(m, sized, conf_thresh, nms_thresh, len(class_names), use_cuda)
            end = time.time()
            print('time {}s'.format(end-start))
            frame_det = plot_boxes(frame, boxes, class_names)
            frame_det = cv2.flip(frame_det,0)
            if dis:
                cv2.imshow("det_img", frame_det)
            
            if save_pth is not None:
                video_det.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print('no frames captured')
            break
    
    cap.release()
    if save_pth is not None:
        video_det.release()
    cv2.destroyAllWindows()

def detect_video_frames(cfgfile, weightfile, frame_path, classnames_file, frame_size, save_pth, conf_thresh, nms_thresh, use_cuda=False, dis=False):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    
    if use_cuda:
        m.cuda()
    class_names = load_class_names(classnames_file)

    size = (frame_size, frame_size)

    if save_pth is not None:
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video_det = cv2.VideoWriter(save_pth, fourcc, 20, size)

    frame_no = 0
    sleep_times = 0
    cv2.namedWindow('origin_video')
    cv2.moveWindow('origin_video', 200, 100)
    cv2.namedWindow('detect_video')
    cv2.moveWindow('detect_video', 800, 100)
    while True:
        frame = cv2.imread(os.path.join(frame_path, 'frame{}.jpg'.format(frame_no)))
        if frame is None:
            time.sleep(1)
            sleep_times += 1
            if sleep_times >= 3:
                break
            continue
        sized = cv2.resize(frame, (frame_size, frame_size))
        if dis:
            cv2.imshow('origin_video', sized)
        boxes = do_detect(m, sized, conf_thresh, nms_thresh, len(class_names), use_cuda)
        frame_det = plot_boxes(frame, boxes, class_names)
        if dis:
            cv2.imshow('detect_video', frame_det)
        if save_pth is not None:
            video_det.write(frame_det)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str, default=None, help='specify the image path or directory')
    parser.add_argument('--weights', '-w', type=str, default='weights/yolo_v3.weights', help='weights file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolo_v3.cfg', help='net configure file')
    parser.add_argument('--save', '-s', type = str, default=None, help = 'save results path')
    parser.add_argument('--class_names', '-n', type = str, default='data/voc.names', help = 'specify the file which contains the class names')
    parser.add_argument('--size', type=int, default = 416, help = 'the size of image, must be the times of 32')
    parser.add_argument('--cuda', action='store_true', default=False, help='if specified, gpu will be used')
    parser.add_argument('--display', '-d', action='store_true', default=False, help='if specified, video will be displayed in window')
    parser.add_argument('--conf_thresh', '-ct', type=float, default=0.5, help='confidence thresh')
    parser.add_argument('--nms_thresh', '-nt', type=float, default=0.4, help='nms thresh')
    args, _ = parser.parse_known_args()
    detect_video_frames(args.config, args.weights, args.video, args.class_names, args.size, args.save, args.conf_thresh, args.nms_thresh, args.cuda, args.display)