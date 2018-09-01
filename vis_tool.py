import torch
import visdom
import numpy as np
class Visualization:
    def __init__(self, env = "main"):
        self.vis = visdom.Visdom(env = env)
        
    def plot(self,x,loss_dict,**kwargs):
        for key in loss_dict:
            v = loss_dict[key] 
            self.vis.line(Y=np.array([v]), X=np.array([x]),
                      win=key,
                      opts=dict(title=key),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )