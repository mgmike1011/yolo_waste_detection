import os

from ultralytics import YOLO

model = YOLO('models/yolov9c.pt')
model.info()

if not os.path.isdir("detect_results/pyexp"):
    os.mkdir("detect_results/pyexp")

result = model.predict(source=["test_img/test_img_1.jpg"], conf=0.25)

for i, r in enumerate(result):
    result[i].save(filename=f"detect_results/pyexp/test_img_{i}.jpg")
    for box in result[i].boxes:
        out = [int(box.cls.cpu().numpy()[0])]
        for xywhn in box.xywhn.cpu().numpy()[0]:
            out.append(xywhn)
        print(out)
