# Detection and Segmentation of Surgical Instruments and their Grasp Points
## Description
Detection and Segmentation of surgical instruments in diverse lighting conditions, and backgrounds. The approximate grasp points (where to place the gripper) are estimated using a fully convolutional network. 
<p align="center">
<img src="images/intro.gif" width="540" height="540">
</p>

## Grasp Point Detection
- Download and arrange our data in the `data` folder from here: [Google drive]()
- Download our pre-trained checkpoints from here: [Google drive]()
- For Training, refer to the `train.ipynb` notebook.
- For Inference, refer to the `inference.ipynb` notebook.

<p align="center">
<img src="images/pic.png" width="600" height="400">
</p>



## Instrument Detection and Segmentation
- Refer to the `./ins_det_seg` directory.

<p align="center">
<img src="images/Picture1.png" width="600" height="600">
</p>

The detection and segmentation network is based on Mask R-CNN from [detectron2](https://github.com/facebookresearch/detectron2). 
