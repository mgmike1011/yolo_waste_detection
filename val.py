import argparse

from ultralytics import YOLO

from utilities.parsing_vaildator import file_path, str2bool

# Parameters - https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation
DEFAULT_IMGSZ_VALUE = 640
DEFAULT_BATCH_VALUE = 100
DEFAULT_SAVE_JSON_VALUE = False
DEFAULT_SAVE_HYBRID_VALUE = False
DEFAULT_CONF_VALUE = 0.001
DEFAULT_IOU_VALUE = 0.6
DEFAULT_MAX_DET_VALUE = 300
DEFAULT_HALF_VALUE = True
DEFAULT_DEVICE_VALUE = "0"
DEFAULT_DNN_VALUE = False
DEFAULT_PLOTS_VALUE = False
DEFAULT_RECT_VALUE = False
DEFAULT_SPLIT_VALUE = "val"


def main(data_path: file_path,
         model_path: file_path,
         device: str,
         imgsz: int,
         batch: int,
         save_json: bool,
         save_hybrid: bool,
         conf: float,
         iou: float,
         max_det: int,
         half: bool,
         dnn: bool,
         plots: bool,
         rect: bool,
         split: str):
    # Load model and validate
    model = YOLO(model_path)
    print("--- Validation start ---")
    model.val(data=data_path, imgsz=imgsz, batch=batch, device=device, save_json=save_json,
              save_hybrid=save_hybrid, conf=conf, iou=iou, max_det=max_det, half=half,
              dnn=dnn, plots=plots, rect=rect, split=split)
    print("--- Validation end ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO validate model performance on given input data.")

    parser.add_argument("--data", type=file_path, help="Input data YAML file.", required=True)
    parser.add_argument("--model", type=file_path, help="Path to a model.", required=True)
    parser.add_argument("--device", type=str, help="Device for inference (e.g., cpu, cuda:0 or 0).",
                        default=DEFAULT_DEVICE_VALUE, required=False)
    parser.add_argument("--imgsz", type=int, help="Image size for inference.",
                        default=DEFAULT_IMGSZ_VALUE, required=False)
    parser.add_argument("--batch", type=int, help="Number of images per batch.",
                        default=DEFAULT_BATCH_VALUE, required=False)
    parser.add_argument("--savejson", type=str2bool, help="Save the results to a JSON file.",
                        default=DEFAULT_SAVE_JSON_VALUE, required=False)
    parser.add_argument("--savehybrid", type=str2bool,
                        help="Save a hybrid version of labels that combines original annotations with additional "
                             "model predictions.",
                        default=DEFAULT_SAVE_HYBRID_VALUE, required=False)
    parser.add_argument("--conf", type=float, help="Minimum confidence threshold for detections.",
                        default=DEFAULT_CONF_VALUE, required=False)
    parser.add_argument("--iou", type=float, help="Intersection Over Union (IoU) threshold for Non-Maximum "
                                                  "Suppression (NMS).", default=DEFAULT_IOU_VALUE, required=False)
    parser.add_argument("--maxdet", type=int, help="Maximum number of detections per image.",
                        default=DEFAULT_MAX_DET_VALUE, required=False)
    parser.add_argument("--half", type=str2bool, help="Enables half-precision (FP16).",
                        default=DEFAULT_HALF_VALUE, required=False)
    parser.add_argument("--dnn", type=str2bool, help="Use the OpenCV DNN module for ONNX model inference.",
                        default=DEFAULT_DNN_VALUE, required=False)
    parser.add_argument("--plots", type=str2bool,
                        help="Generate and save plots of predictions versus ground truth for visual evaluation of the "
                             "model's performance.",
                        default=DEFAULT_PLOTS_VALUE, required=False)
    parser.add_argument("--rect", type=str2bool, help="Use rectangular inference for batching.",
                        default=DEFAULT_RECT_VALUE, required=False)
    parser.add_argument("--split", type=str,
                        help="Determines the dataset split to use for validation (val, test, or train)",
                        default=DEFAULT_SPLIT_VALUE, required=True)

    args = parser.parse_args()

    main(data_path=args.data,
         model_path=args.model,
         device=args.device,
         imgsz=args.imgsz,
         batch=args.batch,
         save_json=args.savejson,
         save_hybrid=args.savehybrid,
         conf=args.conf,
         iou=args.iou,
         max_det=args.maxdet,
         half=args.half,
         dnn=args.dnn,
         plots=args.plots,
         rect=args.rect,
         split=args.split)
