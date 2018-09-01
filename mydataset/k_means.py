import numpy as np
import os
import argparse

class bounding_box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def iou_dis(a, b):
    left_x = max(a.x - a.w / 2.0, b.x - b.w / 2.0)
    right_x = min(a.x + a.w / 2.0, b.x + b.w / 2.0)
    down_y = max(a.y - a.h / 2.0, b.y - b.h / 2.0)
    up_y = min(a.y + a.h / 2.0, b.y + b.h / 2.0)

    if left_x >= right_x or down_y >= up_y:
        iou = 0
    else:
        interaction_area = (right_x - left_x) * (up_y - down_y)
        union_area = a.w * a.h + b.w * b.h - interaction_area
        iou = interaction_area / union_area

    return 1 - iou

def init_centroids(box_list, num_anchors):
    centroids = []
    num_box = len(box_list)

    init_box_id = np.random.choice(num_box, 1)[0]
    centroids.append(box_list[init_box_id])
    print('begin generating initial bounding box...')
    print(box_list[init_box_id].w, box_list[init_box_id].h)

    for _ in range(num_anchors - 1):
        dis_sum = 0.0
        dis_list = []

        for box in box_list:
            min_dis = 1.0
            for center_box in centroids:
                dis = iou_dis(box, center_box)
                if dis < min_dis:
                    min_dis = dis
            dis_sum += min_dis
            dis_list.append(min_dis)

        dis_thresh = dis_sum * np.random.random()
        dis_sum = 0

        for id, dis in enumerate(dis_list):
            dis_sum += dis
            if dis_sum >= dis_thresh:
                centroids.append(box_list[id])
                print(box_list[id].w, box_list[id].h)
                break
    print('finish generating initial labels!')
    return centroids

def do_kmeans(centroids, box_list, num_anchors):
    loss = 0.0
    groups = []
    for _ in range(num_anchors):
        groups.append([])
    
    for box in box_list:
        min_dis = 1.0
        min_group_id = 0
        for group_id, center_box in enumerate(centroids):
            dis = iou_dis(center_box, box)
            if dis < min_dis:
                min_dis = dis
                min_group_id = group_id
        groups[min_group_id].append(box)
        loss += min_dis
    
    new_centroids = []
    for _ in range(num_anchors):
        new_centroids.append(bounding_box(0,0,0,0))
    for centroids_id in range(num_anchors):
        w_list = list(group.w for group in groups[centroids_id])
        h_list = list(group.h for group in groups[centroids_id])
        new_centroids[centroids_id].w = np.mean(w_list)
        new_centroids[centroids_id].h = np.mean(h_list)

    return new_centroids, loss

def get_bounding_box(labpath, max_iteration, num_anchors, convergence_loss):
    box_list = []
    label_list = os.listdir(labpath)
    print('begin reading from label files...')
    for label_file in label_list:
        f = open(os.path.join(labpath, label_file))
        for line in f:
            temp = line.strip().split(" ")
            if len(temp) > 0:
                box = bounding_box(0,0,float(temp[3]),float(temp[4]))
                box_list.append(box)
    print('finish reading all labels!')
    centroids = init_centroids(box_list, num_anchors)
    iteration = 1
    print('begin k-means iterations...')
    centroids, old_loss = do_kmeans(centroids, box_list, num_anchors)
    print('iteration = {} || loss = {}'.format(iteration, old_loss))
    while True:
        iteration = iteration + 1
        centroids, loss = do_kmeans(centroids, box_list, num_anchors)
        print('iteration = {} || loss = {}'.format(iteration, loss))
        if abs(loss - old_loss) < convergence_loss or iteration > max_iteration:
            print('finish getting bounding box!')
            break
        old_loss = loss

    return centroids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', '-lp', type=str, default='/VOCdevkit/VOC2007/labels', help='the path to label txt files')
    parser.add_argument('--max_iteration', '-mi', type=int, default=100, help='max iteration times')
    parser.add_argument('--num_anchors', '-n', type=int, default=9, help='number of anchors')
    parser.add_argument('--convergence_loss', '-cl', type=float, default=1e-6, help='convergence loss to step iteration')
    args, _ = parser.parse_known_args()

    centroids = get_bounding_box(args.label_path, args.max_iteration, args.num_anchors, args.convergence_loss)
    
    print('begin writing to files...')
    f = open('bbox_size.txt', 'w')
    for box in centroids:
        f.write(str(int(box.w*416))+' '+str(int(box.h*416))+' '+'\n')
    f.close()
    print('end writing to files!')
