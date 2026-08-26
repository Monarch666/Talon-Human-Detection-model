import os
import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium

def main():
    print("Loading RF-DETR model checkpoint...")
    checkpoint_path = r"e:\Human Detection Dataset\rfdetr_human_detection\checkpoint_best_ema.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model = RFDETRMedium.from_checkpoint(checkpoint_path)
    else:
        print("Checkpoint not found, loading standard pretrained RFDETRMedium model...")
        model = RFDETRMedium()
        
    img_path = r"e:\Human Detection Dataset\rfdetr_dataset\valid\images\Drone_005.mp4_t-100.jpg"
    print(f"Loading image: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Could not load sample image.")
        return
        
    print("Running inference...")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = model.predict(rgb_image, threshold=0.15)
    
    print(f"Detections found: {len(detections)}")
    
    # Annotate image
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    labels = []
    for i in range(len(detections)):
        conf = detections.confidence[i]
        cls_id = detections.class_id[i]
        labels.append(f"Human {conf:.2f}")
        
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    
    # Add status header overlay
    h, w, _ = annotated.shape
    cv2.rectangle(annotated, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(annotated, "RF-DETR AERIAL HUMAN DETECTION ENGINE | TARGET LOCK: ACTIVE", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
    output_path = r"e:\Human Detection Dataset\demo_result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Demo image saved successfully to {output_path}")

if __name__ == "__main__":
    main()
