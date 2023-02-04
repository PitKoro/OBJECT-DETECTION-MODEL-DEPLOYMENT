import fiftyone as fo
IMG_PATH = "/app/dataset/val2017"
LABELS_PATH = "/app/dataset/annotations/instances_val2017.json"
# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path = IMG_PATH,
    labels_path = LABELS_PATH,
    include_id=True,
)

print(coco_dataset.default_classes)
print(coco_dataset)

session = fo.launch_app(coco_dataset, address="jupyter", port=9000)