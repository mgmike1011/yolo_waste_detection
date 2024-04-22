echo "--- Detection test ---"

python yolov9/detect.py \
    --weights models/yolov9-e-converted.pt \
    --source test_img/test_img_1.jpg \
    --device 0 \
    --project detect_results \
    --img 640 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --save-txt

echo "--- --- ---"