"""
This script extracts information about images from a source COCO file based on their appearance in a result COCO file.
The script identifies and extracts metadata for images that are present in both the source and reference COCO files.
This can be useful for filtering images from a larger dataset based on a subset defined in another dataset.
@author Miłosz Gajewski
@date 06.2024
"""
import json

annotations_file = 'path/to/annotations.json'  # Replace with the path to your COCO annotations file
results_file = 'path/to/cfg.json'  # Replace with the path to your detection results file
new_COCO_file = "path/to/new_annotations.json"

# Read result file for extracting id
with open(results_file, "r") as f:
    results = json.load(f)

# Extract image_id from files in "detected" COCO
all_id = []
for result in results:
    if result["image_id"] not in all_id:
        all_id.append(result["image_id"])

# Load ground truth
with open(annotations_file, "r") as f:
    annotations = json.load(f)

# Create and copy constant information to output COCO file
output_coco = {"images": [], "categories": annotations["categories"], "annotations": [],
               "licenses": annotations["licenses"], "info": annotations["info"]}

# Extract and copy necessary data
for id in all_id:
    # Extract images data
    for image in annotations["images"]:
        if image["id"] == id:
            output_coco["images"].append(image)
    for annotation in annotations["annotations"]:
        if annotation["image_id"] == id:
            output_coco["annotations"].append(annotation)

# Save output COCO file
with open(new_COCO_file, "w") as f:
    json.dump(output_coco, f)
