import os
import sys
import argparse
import json

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.parsing_vaildator import dir_path, file_path, str2bool


OUTPUT_DIRECTORY_PATH_SUBDIRECTORY = "pyexp"
CONFIG_JSON_NAME = "cfg.json"
DEFAULT_CONF_VALUE = 0.5
DEFAULT_OVERLAP_VALUE = 0.1
DEFAULT_IMGSZ_VALUE = 640
DEFAULT_DEVICE_VALUE = "cuda:0"
DEAFULT_VISUAL_VALUE = False


def main(input_path: str,
         output_path: dir_path,
         model_path: file_path,
         coco_gt: file_path,
         conf: float,
         overlap: float,
         imgsz: int,
         visual: bool):
    # Load input data
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
    # Load base coco
    image_id = {}
    with open(coco_gt, 'r') as file:
        coco_data = json.load(file)
        for image in coco_data["images"]:
            image_id[image["file_name"]] = image["id"]
    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=conf,
        device='cuda:0'
    )
    # Prepare output directory
    output_directory_path = os.path.join(output_path, OUTPUT_DIRECTORY_PATH_SUBDIRECTORY)
    expiter = 0
    while os.path.isdir(output_directory_path):
        expiter += 1
        output_directory_path = os.path.join(output_path, OUTPUT_DIRECTORY_PATH_SUBDIRECTORY + str(expiter))
    os.mkdir(output_directory_path)
    # Prepare output dict
    output_list = []
    # Calculate predictions
    for image in tqdm(input_data):
        curr_image_id = None
        basename = os.path.basename(image)
        if basename in image_id:
            curr_image_id = image_id[os.path.basename(image)]
        result = get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_width=imgsz,
            slice_height=imgsz,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type='NMS',
            verbose=0
        )
        coco_output = result.to_coco_predictions(curr_image_id)
        for coco in coco_output:
            output_list.append(coco)
        if visual:
            result.export_visuals(export_dir=output_directory_path, file_name=basename)
    # Prepare json config file
    config_json_path = os.path.join(output_directory_path, CONFIG_JSON_NAME)
    with open(config_json_path, "w") as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO detect on given input with SAHI slicing")

    parser.add_argument("--input", type=str, help="Input image or path to directory with images.", required=True)
    parser.add_argument("--output", type=dir_path, help="Path to directory where data will be saved.", required=True)
    parser.add_argument("--model", type=file_path, help="Path to model.", required=True)
    parser.add_argument("--coco_gt", type=file_path, help="Path to a COCO ground truth file.", required=True)
    parser.add_argument("--conf", type=float, help="Minimum confidence threshold for detections.",
                        default=DEFAULT_CONF_VALUE, required=False)
    parser.add_argument("--overlap", type=float, help="Overlap width/height ratio.", default=DEFAULT_OVERLAP_VALUE,
                        required=False)
    parser.add_argument("--imgsz", type=int, help="Image size for inference.", default=DEFAULT_IMGSZ_VALUE,
                        required=False)
    parser.add_argument("--visual", type=str2bool, help="Save the results as PNG images.",
                        default=DEAFULT_VISUAL_VALUE, required=False)

    args = parser.parse_args()

    main(input_path=args.input,
         output_path=args.output,
         model_path=args.model,
         coco_gt=args.coco_gt,
         conf=args.conf,
         overlap=args.overlap,
         imgsz=args.imgsz,
         visual=args.visual)
