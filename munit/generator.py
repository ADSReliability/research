"""
Generate images based MUNIT
"""

import utils
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from munit import MUNIT
import torchvision.utils as vutils


# 调用编码器和解码器
if __name__ == "__main__":
    config_path = ''
    checkpoint_path = ''
    config = utils.get_config(config_path)
    model = MUNIT(config)
    try:
        state_dict = torch.load(checkpoint_path)
        model.gen_a.load_state_dict(state_dict['a'])
        model.gen_b.load_state_dict(state_dict['b'])
    except:
        raise RuntimeError('load model failed')

    model.cuda()
    new_size = config['new_size']
    style_dim = config['gen']['style_dim']
    encode = model.gen_a.encode
    style_encode = model.gen_b.encode
    decode = model.gen_b.decode
