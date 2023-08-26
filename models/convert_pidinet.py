import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import config_model_converted


def convert_pdc(op, weight):
    if op == 'cv':  # conv vanilla
        return weight
    elif op == 'cp':  # cur point
        return weight
    elif op == 'cd':  # central difference
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight
    elif op == 'ad':  # angular difference
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv
    elif op == 'rd':  # radial difference
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        # 这里为什么是weight[:, :, 1:]，按照原理应该是weight[:, :, ：4]和weight[:, :, 5:]
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    raise ValueError("wrong op {}".format(str(op)))


def convert_pidinet(state_dict, config):
    pdcs = config_model_converted(config)
    new_dict = {}
    for pname, p in state_dict.items():
        if 'init_block.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[0], p)
        elif 'block1_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[1], p)
        elif 'block1_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[2], p)
        elif 'block1_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[3], p)
        elif 'block2_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[4], p)
        elif 'block2_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[5], p)
        elif 'block2_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[6], p)
        elif 'block2_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[7], p)
        elif 'block3_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[8], p)
        elif 'block3_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[9], p)
        elif 'block3_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[10], p)
        elif 'block3_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[11], p)
        elif 'block4_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[12], p)
        elif 'block4_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[13], p)
        elif 'block4_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[14], p)
        elif 'block4_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[15], p)
        else:
            new_dict[pname] = p

    return new_dict


# convert kernel size form 3*3 to 5*5
def converti(w, size=5):
    i, o, s1, s2 = w.size()
    new_w = torch.zeros(i, o, size, size, device=w.device)
    new_w[:, :, (size - s1) // 2:size - (size - s1) // 2, (size - s1) // 2:size - (size - s1) // 2] = w
    return new_w


def convert_pidinet_v2(state_dict, config=None):
    new_dict = {}

    new_dict["module.init_block.init_block.weight"] = \
        (converti(convert_pdc("cv", state_dict['module.init_block.init_block_cv.weight'])) + \
         converti(convert_pdc("cp", state_dict['module.init_block.init_block_cp.weight'])) + \
         converti(convert_pdc("cd", state_dict['module.init_block.init_block_cd.weight'])) + \
         converti(convert_pdc("ad", state_dict['module.init_block.init_block_ad.weight'])) + \
         convert_pdc("rd", state_dict['module.init_block.init_block_rd.weight'])) / 5

    for i in range(1, 5):
        for j in range(1, 5):
            if "module.block{}_{}.conv_cv.weight".format(i, j) in state_dict.keys():
                new_dict["module.block{}_{}.conv.weight".format(i, j)] = \
                    (converti(convert_pdc("cv", state_dict['module.block{}_{}.conv_cv.weight'.format(i, j)])) + \
                     converti(convert_pdc("cp", state_dict['module.block{}_{}.conv_cp.weight'.format(i, j)])) + \
                     converti(convert_pdc("cd", state_dict['module.block{}_{}.conv_cd.weight'.format(i, j)])) + \
                     converti(convert_pdc("ad", state_dict['module.block{}_{}.conv_ad.weight'.format(i, j)])) + \
                     convert_pdc("rd", state_dict['module.block{}_{}.conv_rd.weight'.format(i, j)])) / 5
                new_dict["module.block{}_{}.conv2.weight".format(i, j)] \
                    = state_dict["module.block{}_{}.conv2.weight".format(i, j)]
                new_dict["module.block{}_{}.conv2.bias".format(i, j)] \
                    = state_dict["module.block{}_{}.conv2.bias".format(i, j)]

    for pname, p in state_dict.items():
        if ("predict_heads" in pname) or ("classifier" in pname) or ("shortcut" in pname):
            new_dict[pname] = p

    return new_dict
