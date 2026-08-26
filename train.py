from ultralytics import RTDETR
import os

def main():
    # Load a model
    # We use rtdetr-l.pt as a starting point, it will automatically download if not present
    model = RTDETR('rtdetr-l.pt')

    # Path to dataset yaml
    yaml_path = r"e:\Human Detection Dataset\yolo_dataset\dataset.yaml"
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=4, # reduce batch size if OOM
        device=0, # Use GPU 0
        project="drone_human_detection",
        name="rt_detr_human"
    )
    
if __name__ == '__main__':
    main()
