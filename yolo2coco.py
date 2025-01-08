import os
import json
from PIL import Image

# 设置数据集路径
output_dir = "D:\\coco"  # 修改你想输出的coco格式数据集路径
dataset_path = "D:\\yolo"  # 修改为YOLO格式的数据集路径
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# 类别映射
category_mapping = {"pl5": 1, "pl10": 2, "pl15": 3, "pl20": 4, "pl25": 5, "pl30": 6, "pl40": 7,
                    "pl50": 8, "pl60": 9, "pl70": 10, "pl80": 11, "pl90": 12, "pl100": 13}

categories = [
    {"supercategory": "none", "id": category_mapping[name], "name": name} for name in category_mapping
]

# YOLO格式转COCO格式的函数
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    xmin = (x_center - width / 2) * img_width
    ymin = (y_center - height / 2) * img_height
    bbox_width = width * img_width
    bbox_height = height * img_height
    return [xmin, ymin, bbox_width, bbox_height]

# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }

# 处理每个数据集分区
for split in ['train', 'val']:
    coco_format = init_coco_format()
    annotation_id = 1

    for img_name in os.listdir(os.path.join(images_path, split)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_path, split, img_name)
            label_path = os.path.join(labels_path, split, img_name.replace(".jpg", ".txt"))

            img = Image.open(img_path)
            img_width, img_height = img.size
            base_name = os.path.splitext(img_name)[0]
            image_id = len(coco_format["images"]) + 1

            image_info = {
                "file_name": base_name + ".jpg",
                "height": img_height,
                "width": img_width,
                "id": image_id
            }
            coco_format["images"].append(image_info)

            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                        annotation = {
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "image_id": image_id,
                            "bbox": bbox,
                            "category_id": int(class_id) + 1,
                            "id": annotation_id,
                            "ignore": 0,
                            "segmentation": []
                        }
                        coco_format["annotations"].append(annotation)
                        annotation_id += 1

    # 为每个分区保存JSON文件
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{split}_coco_format.json"), "w") as json_file:
        json.dump(coco_format, json_file, indent=4)
