# Dual Refinement Underwater Object Detection Network

By Baojie Fan, Wei Chen, Yang Cong, and Jiandong Tian

# Introduction

Due to the complex underwater environment, underwater
imaging often encounters some problems such as blur, scale variation,
color shift, and texture distortion. Generic detection algorithms can not
work well when we use them directly in the underwater scene. To ad
dress these problems, we propose an underwater detection framework
with feature enhancement and anchor refinement. 

![image](https://github.com/RobotBj/FERNet/blob/main/datasets/ref2.png)

[Paper : https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650273.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650273.pdf)

#**1. Requirements**

     matplotlib
   
     opencv-python
   
     torch==1.3.1
   
     cython
   
     pillow
   
     torchvision
   
     numpy
   
     pycocotools
   
#**2. Install**

     sh complie.sh
   
     sh make.sh
 
 #**3. Test**
 
      python test FER.py --size $img_size$ --dataset $DatasetType$ --trained_model $model_path$
      
 ![image](https://github.com/RobotBj/FERNet/blob/main/datasets/ref1.png)
      
 #**4. Train**
 
     python train FERNet.py --lr 0.002 --batch_szie 32 --max_epoch 160 --dataset URPC --size 512
     
# Citation
If you use this toolbox or benchmark in your research, please cite this project.

     @inproceedings{fan2020dual,
       title={Dual refinement underwater object detection network},
       author={Fan, Baojie and Chen, Wei and Cong, Yang and Tian, Jiandong},
       booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XX 16},
       pages={275--291},
       year={2020},
       organization={Springer}
       }
