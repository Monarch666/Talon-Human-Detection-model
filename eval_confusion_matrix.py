import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
import matplotlib.pyplot as plt

def main():
    print("Loading RF-DETR model...")
    checkpoint_path = r"e:\Human Detection Dataset\rfdetr_human_detection\checkpoint_best_ema.pth"
    model = RFDETRMedium.from_checkpoint(checkpoint_path)
    
    print("Loading validation dataset...")
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=r"e:\Human Detection Dataset\rfdetr_dataset\valid\images",
        annotations_directory_path=r"e:\Human Detection Dataset\rfdetr_dataset\valid\labels",
        data_yaml_path=r"e:\Human Detection Dataset\rfdetr_dataset\data.yaml"
    )

    def callback(image: np.ndarray) -> sv.Detections:
        # Run inference
        detections = model.predict(image, threshold=0.3)
        
        # We need to make sure the class_id is consistently 0 for human
        if isinstance(detections, sv.Detections):
            detections.class_id = np.zeros_like(detections.class_id)
            return detections
        else:
            # Fallback if custom detections list
            boxes = []
            confidences = []
            class_ids = []
            for d in detections:
                box = d.bounds if hasattr(d, 'bounds') else getattr(d, 'box', None)
                conf = d.confidence if hasattr(d, 'confidence') else 1.0
                if box:
                    boxes.append(box)
                    confidences.append(conf)
                    class_ids.append(0)
            if len(boxes) == 0:
                return sv.Detections.empty()
            return sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )

    print("Computing Confusion Matrix (this may take a minute or two)...")
    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset,
        callback=callback,
        conf_threshold=0.3,
        iou_threshold=0.5
    )

    print("Generating plot...")
    confusion_matrix.plot()
    
    # Save the figure
    output_path = r"e:\Human Detection Dataset\confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    main()
