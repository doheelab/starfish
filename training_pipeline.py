import warnings
warnings.filterwarnings("ignore")

import ast
import os
import json
import pandas as pd
import torch
import importlib
import cv2 

from shutil import copyfile
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.model_selection import GroupKFold
from PIL import Image
from string import Template
from IPython.display import display

TRAIN_PATH = '../input/tensorflow-great-barrier-reef'

# # 1. INSTALL YOLOX
# !git clone https://github.com/Megvii-BaseDetection/YOLOX -q
# !pip install -U pip && pip install -r requirements.txt
# !pip install -v -e . 
# !pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# # 2. PREPARE COTS DATASET FOR YOLOX
# This section is taken from  notebook created by Awsaf [Great-Barrier-Reef: YOLOv5 train](https://www.kaggle.com/awsaf49/great-barrier-reef-yolov5-train)
# ## A. PREPARE DATASET AND ANNOTATIONS
def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_path(row):
    row['image_path'] = f'{TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
df.head(5)

# Taken only annotated photos
df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
df_train = df[df["num_bbox"]>0]

#Annotations 
df_train['annotations'] = df_train['annotations'].progress_apply(lambda x: ast.literal_eval(x))
df_train['bboxes'] = df_train.annotations.progress_apply(get_bbox)

#Images resolution
df_train["width"] = 1280
df_train["height"] = 720

#Path of images
df_train = df_train.progress_apply(get_path, axis=1)

# %%
kf = GroupKFold(n_splits = 5) 
df_train = df_train.reset_index(drop=True)
df_train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y = df_train.video_id.tolist(), groups=df_train.sequence)):
    df_train.loc[val_idx, 'fold'] = fold

df_train.head(5)

HOME_DIR = '../input/' 
DATASET_PATH = 'dataset/images'

# import os

# os.mkdir(f"{HOME_DIR}dataset")
# os.mkdir(f"{HOME_DIR}{DATASET_PATH}")
# os.mkdir(f"{HOME_DIR}{DATASET_PATH}/train2017")
# os.mkdir(f"{HOME_DIR}{DATASET_PATH}/val2017")
# os.mkdir(f"{HOME_DIR}{DATASET_PATH}/annotations")


SELECTED_FOLD = 4

for i in tqdm(range(len(df_train))):
    row = df_train.loc[i]
    if row.fold != SELECTED_FOLD:
        copyfile(f'{row.image_path}', f'{HOME_DIR}{DATASET_PATH}/train2017/{row.image_id}.jpg')
    else:
        copyfile(f'{row.image_path}', f'{HOME_DIR}{DATASET_PATH}/val2017/{row.image_id}.jpg') 

# %%
print(f'Number of training files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/train2017/"))}')
print(f'Number of validation files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/val2017/"))}')

# ## B. CREATE COCO ANNOTATION FILES

def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)

annotion_id = 0

def dataset2coco(df, dest_path):
    
    global annotion_id
    
    annotations_json = {
        "info": [],
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    info = {
        "year": "2021",
        "version": "1",
        "description": "COTS dataset - COCO format",
        "contributor": "",
        "url": "https://kaggle.com",
        "date_created": "2021-11-30T15:01:26+00:00"
    }
    annotations_json["info"].append(info)
    
    lic = {
            "id": 1,
            "url": "",
            "name": "Unknown"
        }
    annotations_json["licenses"].append(lic)

    classes = {"id": 0, "name": "starfish", "supercategory": "none"}

    annotations_json["categories"].append(classes)

    
    for ann_row in df.itertuples():
            
        images = {
            "id": ann_row[0],
            "license": 1,
            "file_name": ann_row.image_id + '.jpg',
            "height": ann_row.height,
            "width": ann_row.width,
            "date_captured": "2021-11-30T15:01:26+00:00"
        }
        
        annotations_json["images"].append(images)
        
        bbox_list = ann_row.bboxes
        
        for bbox in bbox_list:
            b_width = bbox[2]
            b_height = bbox[3]
            
            # some boxes in COTS are outside the image height and width
            if (bbox[0] + bbox[2] > 1280):
                b_width = bbox[0] - 1280 
            if (bbox[1] + bbox[3] > 720):
                b_height = bbox[1] - 720 
                
            image_annotations = {
                "id": annotion_id,
                "image_id": ann_row[0],
                "category_id": 0,
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0
            }
            
            annotion_id += 1
            annotations_json["annotations"].append(image_annotations)
        
        
    print(f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
    return annotations_json

# Convert COTS dataset to JSON COCO
train_annot_json = dataset2coco(df_train[df_train.fold != SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/train2017/")
val_annot_json = dataset2coco(df_train[df_train.fold == SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/val2017/")

# Save converted annotations
save_annot_json(train_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/train.json")
save_annot_json(val_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/valid.json")

# # 3. PREPARE CONFIGURATION FILE
config_file_template = '''

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Define yourself dataset path
        self.data_dir = "../input/dataset/images"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"

        self.num_classes = 1

        self.max_epoch = $max_epoch
        self.data_num_workers = 2
        self.eval_interval = 1
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 2
        
        self.input_size = (800, 1280) # (960, 960)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (800, 1280) # (960, 960)
'''

# <strong> For YOLOX_nano I use input size 460x460 but you can change it for your experiments.</strong> 
NANO = False
if NANO:
    config_file_template = '''

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = "../input/dataset/images"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"

        self.num_classes = 1

        self.max_epoch = $max_epoch
        self.data_num_workers = 2
        self.eval_interval = 1

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth,
                                 self.width,
                                 in_channels=in_channels,
                                 depthwise=True)
            head = YOLOXHead(self.num_classes,
                             self.width,
                             in_channels=in_channels,
                             depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

'''

# <div class="alert alert-warning">
# <strong> I trained model for 20 EPOCHS only .... This is for DEMO purposes only.</strong> 
# </div>

PIPELINE_CONFIG_PATH='cots_config.py'

pipeline = Template(config_file_template).substitute(max_epoch = 20)

with open(PIPELINE_CONFIG_PATH, 'w') as f:
    f.write(pipeline)

# ./yolox/data/datasets/voc_classes.py

voc_cls = '''
VOC_CLASSES = (
  "starfish",
)
'''
with open('../input/YOLOX/data/datasets/voc_classes.py', 'w') as f:
    f.write(voc_cls)

# ./yolox/data/datasets/coco_classes.py

coco_cls = '''
COCO_CLASSES = (
  "starfish",
)
'''
with open('../input/YOLOX/data/datasets/coco_classes.py', 'w') as f:
    f.write(coco_cls)


# # 4. DOWNLOAD PRETRAINED WEIGHTS

# List of pretrained models:
# * YOLOX-s
# * YOLOX-m
# * YOLOX-nano for inference speed (!)
# * etc.

sh = 'wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth'
MODEL_FILE = '../input/yolox_l.pth'

if NANO:
    sh = '''
    wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth
    '''
    MODEL_FILE = 'yolox_nano.pth'

with open('script.sh', 'w') as file:
  file.write(sh)


# # 5. TRAIN MODEL
!python train.py \
    -f cots_config.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o \
    -c {MODEL_FILE} 
    # Remember to chenge this line if you take different model eg. yolo_nano.pth, yolox_s.pth or yolox_m.pth
# %% [markdown]
# # 6. RUN INFERENCE
# 
# ## 6A. INFERENCE USING YOLOX TOOL

# %%
# I have to fix demo.py file because it:
# - raises error in Kaggle (cvWaitKey does not work) 
# - saves result files in time named directory eg. /2021_11_29_22_51_08/ which is difficult then to automatically show results

%cp ../../input/YOLOX-kaggle-fix-for-demo-inference/demo.py tools/demo.py

TEST_IMAGE_PATH = "../input/dataset/images/val2017/0-4614.jpg"
MODEL_PATH = "./YOLOX_outputs/cots_config/best_ckpt.pth"

!python ../input/YOLOX/tools/demo.py image \
    -f cots_config.py \
    -c {MODEL_PATH} \
    --path {TEST_IMAGE_PATH} \
    --conf 0.1 \
    --nms 0.45 \
    --tsize 960 \
    --save_result \
    --device gpu 

OUTPUT_IMAGE_PATH = "./YOLOX_outputs/cots_config/vis_res/0-4614.jpg" 
Image.open(OUTPUT_IMAGE_PATH)

# ## 6B. INFERENCE USING CUSTOM SCRIPT (IT WOULD BE USED FOR COTS INFERENCE PART)
# 
# ### 6B.1 SETUP MODEL

# %%
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

COCO_CLASSES = (
  "starfish",
)

# get YOLOX experiment
current_exp = importlib.import_module('cots_config')
exp = current_exp.Exp()

# set inference parameters
test_size = (960, 960)
num_classes = 1
confthre = 0.1
nmsthre = 0.45


# get YOLOX model
model = exp.get_model()
model.cuda()
model.eval()

# get custom trained checkpoint
ckpt_file = "./YOLOX_outputs/cots_config/best_ckpt.pth"
ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])

# ### 6B.2 INFERENCE BBOXES
def yolox_inference(img, model, test_size): 
    bboxes = []
    bbclasses = []
    scores = []
    
    preproc = ValTransform(legacy = False)

    tensor_img, _ = preproc(img, None, test_size)
    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
    tensor_img = tensor_img.float()
    tensor_img = tensor_img.cuda()

    with torch.no_grad():
        outputs = model(tensor_img)
        outputs = postprocess(
                    outputs, num_classes, confthre,
                    nmsthre, class_agnostic=True
                )

    if outputs[0] is None:
        return [], [], []
    
    outputs = outputs[0].cpu()
    bboxes = outputs[:, 0:4]

    bboxes /= min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    bbclasses = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    
    return bboxes, bbclasses, scores

# %% [markdown]
# ### 6B.3 DRAW RESULT

# %%
def draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, classes_dict):
    for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(bbclasses[i])
            score = scores[i]
            if score < confthre:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img, '{}:{:.1f}%'.format(classes_dict[cls_id], score * 100), (x0, y0 - 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), thickness = 1)
    return img

# %% [markdown]
# ### 6B.4 ALL PUZZLES TOGETHER

# %%
TEST_IMAGE_PATH = "../input/dataset/images/val2017/0-4614.jpg"
img = cv2.imread(TEST_IMAGE_PATH)

# Get predictions
bboxes, bbclasses, scores = yolox_inference(img, model, test_size)

# Draw predictions
out_image = draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, COCO_CLASSES)

# Since we load image using OpenCV we have to convert it 
out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
display(Image.fromarray(out_image))

# <div class="alert alert-success" role="alert">
#     Find this notebook helpful? :) Please give me a vote ;) Thank you
#  </div>

# # 7. SUBMIT TO COTS COMPETITION AND EVALUATE

# %%
import greatbarrierreef

env = greatbarrierreef.make_env()   # initialize the environment
iter_test = env.iter_test()  

# %%
submission_dict = {
    'id': [],
    'prediction_string': [],
}

for (image_np, sample_prediction_df) in iter_test:
 
    bboxes, bbclasses, scores = yolox_inference(image_np, model, test_size)
    
    predictions = []
    for i in range(len(bboxes)):
        box = bboxes[i]
        cls_id = int(bbclasses[i])
        score = scores[i]
        if score < confthre:
            continue
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        predictions.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))
    
    prediction_str = ' '.join(predictions)
    sample_prediction_df['annotations'] = prediction_str
    env.predict(sample_prediction_df)

    print('Prediction:', prediction_str)

# %%
sub_df = pd.read_csv('submission.csv')
sub_df.head()


