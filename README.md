# ConditionDETR-for-tsr-detection
Train a TSR (Traffic Sign Recognition) object detection model using ConditionDETR and perform inference.

## Inference result

We use TSR object detection models trained based on YOLOv5n and Condition DETR to infer the same image and display the inference results.

Condition DETR             |  YOLOv5
:-------------------------:|:-------------------------:
![conddter](https://github.com/allrivertosea/ConditionDETR-for-tsr-detection/blob/main/conddter_result.jpg?raw=true)  |  ![yolov5n](https://github.com/allrivertosea/ConditionDETR-for-tsr-detection/blob/main/yolo_result.jpg?raw=true)

## Installation

### Requirements
- Python >= 3.7, CUDA >= 10.1
- PyTorch >= 1.7.0, torchvision >= 0.6.1
- Cython, COCOAPI, scipy, termcolor

The code is developed using Python 3.8 with PyTorch 1.13.1.

```shell
cd ConditionalDETR
pip install -r requirements.txt
```



## Usage

### Data preparation

Prepare your data in the structure and format of COCO 2017.
We expect the directory structure to be the following:
```
path/to/coco/
├── annotations/  # annotation json files
└── images/
    ├── train2017/    # train images
    └── val2017/      # val images
```
In this project, we use our own tsr_data, and design YOLO label file conversion to COCO format, which can be implemented using the yolo2coco.py script.

### Training

In this project, we train conditional DETR-R50 for 50 epochs:
```shell
python main.py \
    --resume auto \
    --coco_path /path/to/coco \
    --output_dir output/conddetr_r50_epoch50
```
The training process takes around 24 hours on a single machine with 1 RTX3090 card.

Same as DETR training setting, we train conditional DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone. Horizontal flips, scales and crops are used for augmentation. Images are rescaled to have min size 800 and max size 1333. The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

### Evaluation
To evaluate conditional DETR-R50 on COCO *val* with 1 GPUs run:
```shell
python main.py \
    --batch_size 2 \
    --eval \
    --resume <checkpoint.pth> \
    --coco_path /path/to/coco \
    --output_dir output/<output_path>
```

### Inference


```bash
python inference_pth.py --img_path ./0267922.jpg
python inference_pth_video.py --video_path ./test.mp4
```

## License

Conditional DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.



## References

```bibtex
@inproceedings{meng2021-CondDETR,
  title       = {Conditional DETR for Fast Training Convergence},
  author      = {Meng, Depu and Chen, Xiaokang and Fan, Zejia and Zeng, Gang and Li, Houqiang and Yuan, Yuhui and Sun, Lei and Wang, Jingdong},
  booktitle   = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year        = {2021}
}
```
