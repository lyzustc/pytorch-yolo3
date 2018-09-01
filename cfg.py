dis_interval = 10
visdom_env = 'yolov3-voc'
save_epoch_interval = 10
def parse_cfg(cfg_file):
    blocks = []
    block = None
    
    fp = open(cfg_file,'r')
    line = fp.readline()
    
    while line != '':
        if line[0] == '#' or line[0] == '\n':
            line = fp.readline()
            continue
        if line[0] == '[':
            if block is not None:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']\n')
            
        else:
            key, value = line.split('=')
            key = key.strip()
            if(key == 'type'):
                key = '_type'
            value = value.strip()
            block[key] = value
            if block['type'] == 'convolutional':
                try:
                    _ = block['batch_normalize']
                except:
                    block['batch_normalize'] = 0
                try:
                    _ = block['unfrozen']
                except:
                    block['unfrozen'] = 0
                try:
                    _ = block['ori_in']
                except:
                    block['ori_in'] = -1
                try:
                    _ = block['ori_out']
                except:
                    block['ori_out'] = -1
        line = fp.readline()
    
    if block is not None:
        blocks.append(block)
    
    fp.close()
    return blocks