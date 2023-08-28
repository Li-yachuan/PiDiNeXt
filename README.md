# PiDiNeXt
Code of paper "PiDiNeXt: An Efficient Edge Detector based on Parallel Pixel Difference Networks"

## preparing data
following PiDiNet (https://github.com/hellozhuo/pidinet)
Specify the data set location in the data loading code

## Training PiDiNeXt on BSDS500
```bash
python main.py --model pidinet_v2 --config carv4 --sa --dil --iter-size 24 --gpu 0 --epochs 18 --lr 0.005 --lr-type multistep --lr-steps 10-14 --wd 1e-4 --savedir ./path/pidinet-bsds-pascal --dataset BSDS-PASCAL --seed 1334 --note rebuild pidinet_v2 on BSDS --act RReLU --opt adam
```
and model can be selected from "pidinet_v2", pidinet_tiny_v2, pidinet_small_v2"  

## Eval PiDiNeXt on BSDS500
similair to training but loading pretrained model and using  ```--evaluate-converted```
```model can be selected from ```pidinet_tiny_v2_converted, pidinet_small_v2_converted, pidinet_v2_converted```  


