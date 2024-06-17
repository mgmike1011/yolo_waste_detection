import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utilities.parsing_vaildator import file_path


def calculate_mAP(annotations_file: str, results_file: str):
    # Load the ground truth annotations
    coco_gt = COCO(annotations_file)

    # Load the detection results
    coco_dt = coco_gt.loadRes(results_file)

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mAP value
    mAP = coco_eval.stats[1]

    print(f"Mean Average Precision (mAP@50): {mAP:.4f}")

    return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO detect on given input with SAHI slicing")

    parser.add_argument("--input", type=file_path, help="Path to ground truth COCO annotations file.", required=True)
    parser.add_argument("--output", type=file_path, help="Path to detection results file.", required=True)

    args = parser.parse_args()

    calculate_mAP(annotations_file=args.annotations, results_file=args.result)
