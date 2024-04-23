# YOLOv9 Waste Detection
[YOLOv9](https://github.com/WongKinYiu/yolov9) Python interface for using with custom datasets. TODO:
## Installation
1. Clone repository
```bash
git clone https://github.com/mgmike1011/yolo_waste_detection.git
```
2. Enter directory and create other directories
```bash
cd yolo_waste_detection
mkdir detect_results 
```
3. Prepare python environment
```bash
python3 -m venv venv
source venv/bin/activate
```
4. Install dependencies
```bash
pip install -r requirements.txt
```
5. Download `yolov9` models
```bash
mkdir models
wget -P models https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt
wget -P models https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt
```
**Note**: 04.2024 - Only C and E models are available, T, S, M will be published after the paper review and publication.
6. Confirm installation by running test detection
```
python detection_test.py
```
Watch output in `detection_results`.

## Detection
Inference implementation: [`detect.py`](detect.py)
```bash
python detect.py --input path/to/input/img/or/directory \
  --output path/to/output/directory \
  --model path/to/model.pt \
  --conf 0.25 \
  --iou 0.7 \
  --imgsz 640 \
  --device "0" \
  --savetxt False \
  --saveconf False \
  --saveimg True \
  --savecfg False
```
**Params:**
* input - Input image or path to directory with images - *Required*,
* output - Path to directory where data will be saved - *Required*,
* model - Path to model - *Required*,
* conf - Minimum confidence threshold for detections,
* iou - Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS),
* imgsz - Image size for inference,
* device - Device for inference (e.g., cpu, cuda:0 or 0),
* savetxt - Save results to txt result file,
* saveconf - Save confidence score to txt result file,
* saveimg - Save annotated image,
* savecfg - Save config file.
## Training
Train implementation: [`train.py`](train.py) \
**Dataset structure:**
```
dataset/
    |
    ->train/
    |    |
    |    ->images/
    |    ->labels/
    |
    ->test/
    |    |
    |    ->images/
    |    ->labels/
    |
    ->valid/
    |    |
    |    ->images/
    |    ->labels/
    |
    ->data.yaml
```
**Note**: test/ is optional. \
Example `data.yaml`:
```yaml
train: /path/to/dataset/train/images
test: /path/to/dataset/test/images
val: /path/to/dataset/valid/images

nc: 1
names: ['class_name']
```
```bash
python train.py #TODO:
```
**Params:**
* a
* b 
TODO:
## Validation
Validation implementation: [`val.py`](val.py) 
```bash
python val.py  #TODO:
```
**Params:**
* a - 
* b -
TODO:
## Synthetic data generator
[Synthetic generator](https://github.com/AgniechaP/synthetic_data_generation) - Agnieszka Piórkowska, Miłosz Gajewski
## YOLOv9 credits
* [Ultralitics](https://docs.ultralytics.com/models/yolov9)
* [WongKinYiu](https://github.com/WongKinYiu/yolov9/)
#### Agnieszka Piórkowska, Miłosz Gajewski
##### Politechnika Poznańska