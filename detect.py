import argparse
import json
import os
import sys

from ultralytics import YOLO

from utilities.parsing_vaildator import dir_path, file_path

DEFAULT_MODEL_PATH = "models/yolov9c.pt"
OUTPUT_DIRECTORY_PATH_SUBDIRECTORY = "pyexp"
DEFAULT_CONF_VALUE = 0.25
DEFAULT_IOU_VALUE = 0.7
DEFAULT_IMGSZ_VALUE = 640
DEFAULT_DEVICE_VALUE = "0"
DEFAULT_SAVE_TXT_VALUE = False
DEFAULT_SAVE_CONF_VALUE = False
DEFAULT_SAVE_IMG_VALUE = True
DEFAULT_SAVE_CFG_VALUE = False
CONFIG_JSON_NAME = "cfg.json"


def main(input_path: str,
         output_path: dir_path,
         model_path: file_path,
         conf: float,
         iou: float,
         imgsz: int,
         device: str,
         save_conf: bool,
         save_txt: bool,
         save_img: bool,
         save_cfg: bool
         ):
    # Prepare input data
    input_data = []
    is_file = os.path.isfile(input_path)
    is_dir = os.path.isdir(input_path)
    if is_file and not is_dir:
        input_data.append(input_path)
    elif is_dir and not is_file:
        for file in sorted(os.listdir(input_path)):
            if file.endswith(".jpg") or file.endswith(".png"):
                input_data.append(os.path.join(input_path, file))
    else:
        print("Wrong input argument!")
        sys.exit()
    # Load model
    model = YOLO(model_path)
    model.info()
    # Prepare output directory
    output_directory_path = os.path.join(output_path, OUTPUT_DIRECTORY_PATH_SUBDIRECTORY)
    expiter = 0
    while os.path.isdir(output_directory_path):
        expiter += 1
        output_directory_path = os.path.join(output_path, OUTPUT_DIRECTORY_PATH_SUBDIRECTORY + str(expiter))
    os.mkdir(output_directory_path)
    # Prepare json config file
    config_json_path = os.path.join(output_directory_path, CONFIG_JSON_NAME)
    if save_cfg:
        with open(config_json_path, "w") as file:
            json.dump([], file)
    # Predict
    results = model.predict(source=input_data, conf=conf, iou=iou, imgsz=imgsz, device=device)
    # Save results
    for result in results:
        image = os.path.basename(result.path)
        image_name = os.path.splitext(image)[0]
        result_txt_path = os.path.join(output_directory_path, (image_name + ".txt"))
        result_img_path = os.path.join(output_directory_path, image)
        # Prepare data
        if save_txt:
            result.save_txt(txt_file=result_txt_path, save_conf=save_conf)
        # Save annotated image
        if save_img:
            result.save(filename=result_img_path)
        # Save config json file
        if save_cfg:
            height, width = result.orig_shape
            xywh = [box for box in result.boxes.xywh.cpu().numpy().tolist()]
            xywhn = [box for box in result.boxes.xywhn.cpu().numpy().tolist()]
            cls = [cl for cl in result.boxes.cls.cpu().numpy().tolist()]
            out_obj = {
                "name": result.names,
                "speed": result.speed,
                "path": result.path,
                "xywhn": xywhn,
                "xywh": xywh,
                "cls": cls,
                "width": width,
                "height": height
            }
            with open(config_json_path, "r+") as file:
                file_data = json.load(file)
                file_data.append(out_obj)
                file.seek(0)
                json.dump(file_data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO detect on given input")

    parser.add_argument("--input", type=str, help="Input image or path to directory with images.", required=True)
    parser.add_argument("--output", type=dir_path, help="Path to directory where data will be saved.", required=True)
    parser.add_argument("--model", type=file_path, help="Path to model.", default=DEFAULT_MODEL_PATH, required=True)
    parser.add_argument("--conf", type=float, help="Minimum confidence threshold for detections.",
                        default=DEFAULT_CONF_VALUE, required=False)
    parser.add_argument("--iou", type=float, help="Intersection Over Union (IoU) threshold for Non-Maximum "
                                                  "Suppression (NMS).", default=DEFAULT_IOU_VALUE, required=False)
    parser.add_argument("--imgsz", type=int, help="Image size for inference.",
                        default=DEFAULT_IMGSZ_VALUE, required=False)
    parser.add_argument("--device", type=str, help="Device for inference (e.g., cpu, cuda:0 or 0).",
                        default=DEFAULT_DEVICE_VALUE, required=False)
    parser.add_argument("--savetxt", type=bool, help="Save results to txt result file.",
                        default=DEFAULT_SAVE_TXT_VALUE, required=False)
    parser.add_argument("--saveconf", type=bool, help="Save confidence score to txt result file.",
                        default=DEFAULT_SAVE_CONF_VALUE, required=False)
    parser.add_argument("--saveimg", type=bool, help="Save annotated image.",
                        default=DEFAULT_SAVE_IMG_VALUE, required=False)
    parser.add_argument("--savecfg", type=bool, help="Save config file.",
                        default=DEFAULT_SAVE_CFG_VALUE, required=False)

    args = parser.parse_args()

    main(input_path=args.input,
         output_path=args.output,
         model_path=args.model,
         conf=args.conf,
         iou=args.iou,
         imgsz=args.imgsz,
         device=args.device,
         save_conf=args.saveconf,
         save_txt=args.savetxt,
         save_img=args.saveimg,
         save_cfg=args.savecfg)
