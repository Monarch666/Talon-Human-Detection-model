import os
from rfdetr import RFDETRMedium # or RFDETRNano, RFDETRSmall depending on resources

def main():
    # We will use RFDETRMedium for a good balance of speed and accuracy
    model = RFDETRMedium()

    # The dataset directory should be the YOLO dataset directory
    # which contains images/train, images/val etc.
    # RFDETR requires train/valid/test splits.
    dataset_dir = r"e:\Human Detection Dataset\rfdetr_dataset"
    
    # Dataset is already reorganized.
    output_dir = r"e:\Human Detection Dataset\rfdetr_human_detection"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting RF-DETR Training...")
    model.train(
        dataset_dir=dataset_dir,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=output_dir,
        device="cuda"
    )

if __name__ == '__main__':
    main()
