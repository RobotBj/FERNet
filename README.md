# FERNet
#1. Requirements

     matplotlib
   
     opencv-python
   
     torch
   
     cython
   
     pillow
   
     torchvision
   
     numpy
   
     pycocotools
   
#2. Install

     sh complie.sh
   
     sh make.sh
 
 #3. test
 
      python test FER.py --size $img_size$ --dataset $VOC$ 
      
 #4. train
 
     python train FERNet.py --lr 0.002 --batch_szie 32 --max_epoch 160 --dataset URPC --size 512
