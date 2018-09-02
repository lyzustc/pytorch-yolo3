from multiprocessing import Value, Process
from multiprocessing.managers import BaseManager as bm
import cv2
import queue
import argparse
import time
import os

def send_frames_video(videofile, frame_size):
    global m
    ori_q = m.get_ori()

    if videofile is not None:
        cap = cv2.VideoCapture(videofile)
    else:
        cap = cv2.VideoCapture(0)

    cv2.namedWindow('origin_video')
    cv2.moveWindow('origin_video', 200, 100)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            sized = cv2.resize(frame, (frame_size, frame_size))
            cv2.imshow("origin_video", sized)
            ori_q.put(sized)
        else:
            break
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    m.add_send_end()
    cap.release()
    cv2.destroyWindow("origin_video")
    print('getting and sending frames program exists!')

def send_frames(frame_path, frame_size):
    global m
    ori_q = m.get_ori()
    assert os.path.exists(frame_path) == True, 'cannot find frame files path!'
    frame_no = 0
    sleep_times = 0

    cv2.namedWindow('origin_video')
    cv2.moveWindow('origin_video', 200, 100)

    while True:
        frame = cv2.imread(os.path.join(frame_path, 'photo{}.jpg'.format(frame_no)))
        if frame is None:
            time.sleep(1)
            sleep_times += 1
            if sleep_times >= 3:
                if len(os.listdir(frame_path)) > 0:
                    frame_no += 1
                    sleep_times = 0
                else:
                    break
            continue
        sized = cv2.resize(frame, (frame_size, frame_size))
        cv2.imshow('origin_video', sized)
        ori_q.put(sized)

        os.remove(os.path.join(frame_path, 'photo{}.jpg'.format(frame_no)))
        frame_no += 1
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    m.add_send_end()
    cv2.destroyWindow("origin_video")
    print('getting and sending frames program exists!')

def rcv_frames(write_video = False, save_path = None, frame_size = 416):
    global m
    det_q = m.get_det()

    cv2.namedWindow('detect_video')
    cv2.moveWindow('detect_video', 800, 100)

    if write_video:
        assert save_path is not None, 'cannot find path to save video file'
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video_det = cv2.VideoWriter(save_path, fourcc, 20, (frame_size, frame_size))

    while True:
        if det_q.empty():
            if pro_end_signal.value == 1:
                break
            else:
                continue
        else:
            img = det_q.get()
            cv2.imshow("detect_video", img)
            if write_video:
                video_det.write(img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    m.add_pro_end()
    cv2.destroyWindow("detect_video")
    print('showing detection frames program exists!')

def super_pro_end():
    while True:
        if m.get_pro_end().pop() == 1:
            pro_end_signal.value = 1
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', action='store_true', default=False, help='if specified, read frames from a video file')
    parser.add_argument('--video_file_path', '-vp', type=str, default=None, help='specify the file path of video file, if not specified, webcam will be used')
    parser.add_argument('--frame_path', '-fp', type=str, default=None, help='specify the path of video frames')
    parser.add_argument('--frame_size', '-s', type=int, default=416, help='specify the size of video frames')
    parser.add_argument('--write_video', '-w', action='store_true', default=False, help='if specified, write video frames to a file')
    parser.add_argument('--save_path', '-sp', type=str, default=None, help='the path to save your detection video results')
    args, _ = parser.parse_known_args()

    pro_end_signal = Value('i', 0)

    bm.register('get_ori')
    bm.register('get_det')
    bm.register('add_send_end')
    bm.register('get_send_end')
    bm.register('add_pro_end')
    bm.register('get_pro_end')
    m = bm(address=('geeekvr.com',8014),authkey = b'abc')
    m.connect()

    if args.video:
        p1 = Process(target = send_frames_video, args = (args.video_file_path, args.frame_size))
    else:
        p1 = Process(target = send_frames, args = (args.frame_path, args.frame_size))  
    p2 = Process(target = rcv_frames, args = (args.write_video, args.save_path,args.frame_size))
    p3 = Process(target = super_pro_end)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()   
