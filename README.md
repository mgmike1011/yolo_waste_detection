# YOLOv9 Waste Detection
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
3. **BASH:** Clone yolov9 repository
```bash
git clone https://github.com/WongKinYiu/yolov9/
```
4. Prepare python environment
```bash
python3 -m venv venv
source venv/bin/activate
```
5. Install dependencies
```bash
pip install -r requirements.txt
```
6. **BASH:** Install `yolov9` python packages
```bash
pip install -r yolov9/requirements.txt 
```
7. Download `yolov9` models
**BASH:**
```bash
mkdir models
wget -P models https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
wget -P models https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt
```
**PYTHON:**
```bash
mkdir models
wget -P models https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt
wget -P models https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt
```
**Note**: 04.2024 - Only C and E models are available, T, S, M will be published after the paper review and publication. \
8. Confirm installation by running test detection
**BASH:**
```bash
bash detection_test.sh
```
**PYTHON:**
```
python detection_test.py
```
Watch output in `detection_results`.
## Detection
TODO:
## Train
1. Prepare data directories
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
## YOLOv9 implementations
* [Ultralitics](https://docs.ultralytics.com/models/yolov9)
* [WongKinYiu](https://github.com/WongKinYiu/yolov9/)
#### Agnieszka Piórkowska, Miłosz Gajewski
##### Politechnika Poznańska