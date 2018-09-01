import torch
import torch.nn as nn
import torch.nn.functional as F
from load_save_tools import *
import numpy as np
from cfg import *
from darknet_components import *

class Darknet(nn.Module):

    def __init__(self, cfgfile, use_cuda = False):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.use_cuda = use_cuda
        self.models = self.create_layers(self.blocks)
        self.loss_layers = self.getLossLayers()

        self.header = torch.IntTensor([0,1,0,0])
        self.seen = 0

    def getLossLayers(self):
        loss_layers = []
        for m in self.models:
            if isinstance(m, YoloLayer):
                loss_layers.append(m)
        return loss_layers

    def create_layers(self, blocks):
        in_channels = None
        out_channels_record = []
        prev_scale = 1
        scale_record = []
        conv_id = 0
        models = nn.ModuleList()
        ind = -2
        for block in blocks:
            ind = ind + 1  
            if block['type'] == 'net':
                self.width = int(block['width'])
                self.height = int(block['height'])
                in_channels = int(block['channels'])
                continue 

            elif block['type'] == 'convolutional':
                batch_normlize = int(block['batch_normalize'])
                stride = int(block['stride'])
                filters = int(block['filters'])
                size = int(block['size'])
                is_pad = int(block['pad'])
                activation = block['activation']
                unfrozen = int(block['unfrozen'])
                model = nn.Sequential()

                if is_pad:
                    pad = (size-1)//2
                else:
                    pad = 0 

                if batch_normlize == 1:
                    model.add_module('conv{}'.format(conv_id), nn.Conv2d(in_channels, filters, size, stride, pad, bias = False))
                    model.add_module('bn{}'.format(conv_id), nn.BatchNorm2d(filters, momentum = 0.01))
                    # if unfrozen == 0:
                    #     model[1].bias.data.requires_weight = False
                    #     model[1].weight.data.requires_weight = False
                    #     model[1].running_mean.data.requires_weight = False
                    #     model[1].running_var.data.requires_weight = False
                else:
                    model.add_module('conv{}'.format(conv_id), nn.Conv2d(in_channels, filters, size, stride, pad, bias = True))

                if activation == 'leaky':
                    model.add_module('leaky{}'.format(conv_id), nn.LeakyReLU(0.1, inplace = True))
                elif activation == 'relu':
                    model.add_module('relu{}'.format(conv_id), nn.ReLU(inplace = True))
                
                models.append(model)

                in_channels = filters
                out_channels_record.append(filters)
                prev_scale = prev_scale * stride
                scale_record.append(prev_scale)

            elif block['type'] == 'maxpool':
                size = int(block['size'])
                stride = int(block['stride'])
                model = nn.MaxPool2d(size = size, stride = stride)
                models.append(model)
                out_channels_record.append(in_channels)
                prev_scale = prev_scale * stride
                scale_record.append(prev_scale)

            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_channels_record.append(in_channels)
                models.append(model)

            elif block['type'] == 'softmax':
                model = nn.Softmax()
                scale_record.append(prev_scale)
                out_channels_record.append(in_channels)
                models.append(model)

            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                in_channels = stride * stride * in_channels
                out_channels_record.append(in_channels)
                prev_scale = prev_scale * stride
                scale_record.append(prev_scale)                
                models.append(Reorg(stride))

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_channels_record.append(in_channels)
                prev_scale = prev_scale / stride
                scale_record.append(prev_scale)                
                models.append(Upsample(stride))

            elif block['type'] == 'shortcut':
                ind = len(models)
                in_channels = out_channels_record[ind-1]
                out_channels_record.append(in_channels)
                prev_scale = scale_record[ind-1]
                scale_record.append(prev_scale)
                models.append(EmptyModule())

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    in_channels = out_channels_record[layers[0]]
                    prev_scale = scale_record[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    in_channels = out_channels_record[layers[0]] + out_channels_record[layers[1]]
                    prev_scale = scale_record[layers[0]]
                out_channels_record.append(in_channels)
                scale_record.append(prev_scale)
                models.append(EmptyModule())

            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors)//yolo_layer.num_anchors
                try:
                    yolo_layer.rescore = int(block['rescore'])
                except:
                    pass
                yolo_layer.ignore_thresh = float(block['ignore_thresh'])
                yolo_layer.truth_thresh = float(block['truth_thresh'])
                yolo_layer.stride = prev_scale
                yolo_layer.nth_layer = ind
                yolo_layer.net_width = self.width
                yolo_layer.net_height = self.height
                out_channels_record.append(in_channels)
                scale_record.append(prev_scale)
                models.append(yolo_layer)          
            else:
                print('Unknown layers.')
        return models

    def forward(self, x):
        ind = -2
        self.loss_layers = None
        outputs = dict()
        out_boxes = dict()
        outno = 0
        for block in self.blocks:
            ind = ind + 1

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] in [ 'region', 'yolo']:
                boxes = self.models[ind].get_mask_boxes(x)
                out_boxes[outno]= boxes
                outno += 1
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x if outno == 0 else out_boxes

    def load_binfile(self, weightfile):
        fp = open(weightfile, 'rb')
       
        version = np.fromfile(fp, count=3, dtype=np.int32)
        version = [int(i) for i in version]
        if version[0]*10+version[1] >=2 and version[0] < 1000 and version[1] < 1000:
            seen = np.fromfile(fp, count=1, dtype=np.int64)
        else:
            seen = np.fromfile(fp, count=1, dtype=np.int32)
        self.header = torch.from_numpy(np.concatenate((version, seen), axis=0))
        self.seen = int(seen)
        body = np.fromfile(fp, dtype=np.float32)
        fp.close()
        return body

    def load_weights(self, weightfile):
        buf = self.load_binfile(weightfile)
        start = 0
        ind = -2
        is_weight_file = True
        for block in self.blocks:
            if start >= buf.size:
                is_weight_file = False
                #break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if block['unfrozen'] == 0:
                    if batch_normalize:
                        start = load_conv_bn(buf, start, model[0], model[1])
                    else:
                        start = load_conv(buf, start, model[0])
                else:
                    if batch_normalize:
                        nn.init.kaiming_normal_(model[0].weight, mode='fan_out')
                        model[1].weight.data = torch.ones_like(model[1].weight.data)
                        model[1].bias.data.zero_()
                        num_w = int(block['ori_in']) * int(block['ori_out'])
                        num_b = int(block['ori_out'])
                        start += 4 * num_b + num_w
                    else:
                        nn.init.kaiming_normal_(model[0].weight, mode='fan_out')
                        model[0].bias.data.zero_()
                        num_w = int(block['ori_in']) * int(block['ori_out'])
                        num_b = int(block['ori_out'])
                        start += num_w + num_b
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass                
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = np.array(self.header[0:3].numpy(), np.int32)
        header.tofile(fp)
        if (self.header[0]*10+self.header[1]) >= 2:
            seen = np.array(self.seen, np.int64)
        else:
            seen = np.array(self.seen, np.int32)
        seen.tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass                
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

