from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

# Load the ground truth annotations
coco_gt = COCO('/media/roderickmajoor/TREKSTOR/Test/annotations_only_extended_cols_mask.json')  # Ground truth annotations

# Load the predicted annotations
coco_pred = COCO('/media/roderickmajoor/TREKSTOR/Test/annotations_pred_updated.json')  # Predicted annotations

# Convert the predicted annotations to the results format expected by COCOeval
def coco_annotations_to_results(coco, anns):
    results = []
    for ann in anns:
        result = {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": 1.0,  # Assign a dummy confidence score of 1.0 for all predictions
            "segmentation": ann.get("segmentation", None)
        }
        results.append(result)
    return results

pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds())
pred_results = coco_annotations_to_results(coco_pred, pred_anns)

# Save the converted results to a temporary JSON file
with open('temp_annotations_pred.json', 'w') as f:
    json.dump(pred_results, f)

# Load the results using the COCO API
coco_dt = coco_gt.loadRes('temp_annotations_pred.json')

# Initialize COCOeval object for bbox
coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')

# Run evaluation for bbox
coco_eval_bbox.evaluate()
coco_eval_bbox.accumulate()
coco_eval_bbox.summarize()

# Extract precision and recall values
precision_values = coco_eval_bbox.eval['precision']
recall_values = coco_eval_bbox.eval['recall']

mean_precision = coco_eval_bbox.stats[1]
mean_recall = coco_eval_bbox.stats[9]

print('Mean Precision at IoU=0.5: ', mean_precision)
print('Mean Recall at IoU=0.5: ', mean_recall)

# Initialize COCOeval object for segm
coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')

# Run evaluation for segm
coco_eval_segm.evaluate()
coco_eval_segm.accumulate()
coco_eval_segm.summarize()
