# python test.py /workspace/pidinet-master/path/to/0421-pidinet-bsds-pascal BSDS-PASCAL 1 pidinet_converted_v2_test

import sys
import os
import glob

root = sys.argv[1]
dataset = sys.argv[2]
gpu = sys.argv[3]
model= sys.argv[4]

pths = glob.glob(os.path.join(root, "save_models", "*.pth.tar"))

assert dataset in ['BSDS', 'BSDS-PASCAL', 'NYUD-image', 'NYUD-hha', 'Multicue-boundary-1', 'Multicue-boundary-2',
                   'Multicue-boundary-3', 'Multicue-edge-1', 'Multicue-edge-2', 'Multicue-edge-3', 'Custom',
                   "BIPED"]
# exist_result = os.listdir(os.path.join(root,"eval_results",dataset,"SS"))

for ckpt in pths:

    cmd = "python main.py " \
          "--model {} " \
          "--config carv4 " \
          "--sa " \
          "--dil " \
          "--gpu {} " \
          "--savedir {}" \
          " --dataset {} " \
          "--evaluate {} " \
          "--evaluate-converted " \
          "--act RReLU " \
          "--note nothing".format(model,gpu,root, dataset, ckpt) \

    os.system(cmd)
