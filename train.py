import torch
import torch.nn as nn
import torch.optim as optim
from darknet import *
from utils import *
from cfg import *
import argparse
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from vis_tool import Visualization
import math

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, default='cfg/voc.data', help='data definition file path')
parser.add_argument('--weights', '-w', type=str, default='weights/yolo_v3.weights', help='initial weights file path')
parser.add_argument('--config', '-c', type=str, default='cfg/yolo_v3.cfg', help='net configure file path')
parser.add_argument('--vis', '-v', action='store_true', default=False, help='if specified, use visdom to visualize')
parser.add_argument('--cuda', action='store_true', default=False, help='if specified, use gpu to train')
parser.add_argument('--reset', '-r', action='store_true', default = False, help='if specified, start a new training')
parser.add_argument('--save', '-s', type=str, default='weights',help='weights save path')
parser.add_argument('--paral', '-p', action='store_true', default=False, help='if specified, use multi-gpu to train')
parser.add_argument('--data_augmentation', '-da', action='store_true', default=False, help='if specified, use data aumentation before loading data')
args, _ = parser.parse_known_args()

data_pth = args.data
weights_pth = args.weights
cfg_pth = args.config
save_pth = args.save
use_vis = args.vis
use_cuda = args.cuda
use_paral = args.paral
use_da = args.data_augmentation

data_options  = read_data_cfg(data_pth)
dataset_name = data_options['dataset']
train_pth = data_options['train']
test_pth = data_options['valid']
class_names = data_options['names']
num_workers = int(data_options['num_workers'])

net_options = parse_cfg(cfg_pth)[0]
learning_rate = float(net_options['learning_rate'])
momentum = float(net_options['momentum'])
decay = float(net_options['decay'])
burn_in = float(net_options['burn_in'])
batch_size = int(net_options['batch'])
subdivisions = int(net_options['subdivisions'])
steps = [float(step) for step in net_options['steps'].split(',')]
scales = [float(scale) for scale in net_options['scales'].split(',')]
nsamples = file_lines(train_pth)

epoch_batches = math.ceil(nsamples / batch_size)
try:
    max_epochs = int(net_options['max_epochs'])
    max_batches = max_epochs * epoch_batches
except KeyError:
    max_batches = int(net_options['max_batches'])

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = Darknet(cfg_pth, use_cuda=use_cuda)
model.load_weights(weights_pth)

if os.path.exists(save_pth) == False:
    os.mkdir(save_pth)

if use_cuda:
    model.to(device)
    if use_paral and torch.cuda.device_count() > 1:
        paral_model = nn.DataParallel(model)
        model = paral_model.module

if args.reset:
    model.seen = 0
    init_batch = 0
else:
    init_batch = model.seen//batch_size

loss_layers = model.loss_layers

if use_vis:
    vis = Visualization(visdom_env)

lr = get_learning_rate(learning_rate, init_batch, burn_in, steps, scales)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, dampening=0, weight_decay=decay)

dataset = MyDataset(txtfile = train_pth, dataset = dataset_name, is_train = True, use_da = use_da)
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
dataLoader = DataLoader(dataset = dataset, shuffle = True, batch_size = batch_size//subdivisions, **kwargs)
dataIter = iter(dataLoader)


for iteration in range(init_batch, max_batches):
    if iteration < burn_in:
        lr = warm_up(iteration, burn_in, learning_rate)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, dampening=0, weight_decay=decay)
    dataset.shape = get_shape()

    optimizer.zero_grad()
    loss_sum = 0
    loss_coord_sum = 0
    loss_conf_sum = 0
    loss_cls_sum = 0
    for i in range(subdivisions):
        try:
            data, target = next(dataIter)
        except:
            dataLoader = DataLoader(dataset = dataset, shuffle = True, batch_size = batch_size//subdivisions, **kwargs)
            dataIter = iter(dataLoader)
            data, target = next(dataIter)
        model.train()
        data, target = data.to(device), target.to(device)
        if use_cuda and use_paral:
            output = paral_model(data)
        else:
            output = model(data)

        losses_coord = []
        losses_conf = []
        losses_cls = []
        for i, l in enumerate(loss_layers):
            ol_coord, ol_conf, ol_cls=l(output[i]['x'], target)
            losses_coord.append(ol_coord)
            losses_conf.append(ol_conf)
            losses_cls.append(ol_cls)

        loss_coord = sum(losses_coord)/subdivisions
        loss_conf = sum(losses_conf) / subdivisions
        loss_cls = sum(losses_cls) / subdivisions
        loss = loss_coord + loss_conf + loss_cls

        loss_coord_sum += loss_coord.detach().item()
        loss_conf_sum += loss_conf.detach().item()
        loss_cls_sum += loss_cls.detach().item() 
        loss_sum += loss.detach().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1000)
        del data,target

    optimizer.step()
    model.seen += batch_size

    if iteration in steps:
        lr = adjust_learning_rate(lr, iteration, steps, scales)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, dampening=0, weight_decay=decay)

    if iteration % dis_interval == 0:
        print("iteration = {}, shape = {}|| loss = {}".format(iteration, dataset.shape, loss_sum))
        if use_vis:
            total_loss = loss_sum
            conf_loss = loss_conf_sum
            cls_loss = loss_cls_sum
            coord_loss = loss_coord_sum
            loss_dict = dict(total_loss = total_loss, conf_loss = conf_loss, cls_loss = cls_loss, coord_loss = coord_loss)
            vis.plot(iteration,loss_dict)

    if iteration > 0 and iteration % (save_epoch_interval * epoch_batches) == 0:
        print('saving weights...')
        model.save_weights(os.path.join(save_pth, 'epoch{}_{}.weights'.format(iteration // epoch_batches,dataset.dataset)))

model.save_weights(os.path.join(save_pth, 'final.weights'))