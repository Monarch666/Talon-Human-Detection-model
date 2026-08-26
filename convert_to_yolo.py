import os
import cv2
import pandas as pd
from tqdm import tqdm
import shutil

# Paths
BASE_DIR = r"e:\Human Detection Dataset"
DATASET_DIR = os.path.join(BASE_DIR, "dataset1")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "ntut_drone_train", "ntut_drone_train")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "ntut_drone_test", "ntut_drone_test")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Output YOLO dataset paths
YOLO_DIR = os.path.join(BASE_DIR, "yolo_dataset")
YOLO_TRAIN_IMG = os.path.join(YOLO_DIR, "images", "train")
YOLO_VAL_IMG = os.path.join(YOLO_DIR, "images", "val")
YOLO_TRAIN_LBL = os.path.join(YOLO_DIR, "labels", "train")
YOLO_VAL_LBL = os.path.join(YOLO_DIR, "labels", "val")

os.makedirs(YOLO_TRAIN_IMG, exist_ok=True)
os.makedirs(YOLO_VAL_IMG, exist_ok=True)
os.makedirs(YOLO_TRAIN_LBL, exist_ok=True)
os.makedirs(YOLO_VAL_LBL, exist_ok=True)

# Human labels
HUMAN_LABELS = {"walk", "push", "stand", "watchphone", "block25", "block50", "block75"}

def convert_to_yolo():
    csv_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.csv')]
    for csv_file in csv_files:
        drone_name = csv_file.split('-')[0] # e.g., Drone_005
        csv_path = os.path.join(LABELS_DIR, csv_file)
        
        df = pd.read_csv(csv_path)
        
        # Check if drone_name exists in train or test
        if os.path.exists(os.path.join(TRAIN_IMG_DIR, drone_name)):
            img_src_dir = os.path.join(TRAIN_IMG_DIR, drone_name, "vott-csv-export")
            split = "train"
        elif os.path.exists(os.path.join(TEST_IMG_DIR, drone_name)):
            img_src_dir = os.path.join(TEST_IMG_DIR, drone_name, "vott-csv-export")
            split = "val"
        else:
            print(f"Images for {drone_name} not found.")
            continue
            
        print(f"Processing {csv_file} -> {split}...")
        
        # Group by image
        grouped = df.groupby('image')
        for image_name, group in tqdm(grouped, total=len(grouped)):
            img_file = image_name + ".jpg"
            img_path = os.path.join(img_src_dir, img_file)
            
            if not os.path.exists(img_path):
                # Sometimes extensions are already there or different
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width, _ = img.shape
            
            yolo_labels = []
            for _, row in group.iterrows():
                label = str(row['label']).strip().lower()
                # we are detecting only humans
                if label not in HUMAN_LABELS:
                    continue
                
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                
                # Convert to YOLO
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                
                # constrain to 0-1
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))
                
                # class id is 0 for human
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                
            # Save label file and copy image if we have labels
            if len(yolo_labels) > 0:
                txt_name = image_name + ".txt"
                if split == "train":
                    out_img_path = os.path.join(YOLO_TRAIN_IMG, img_file)
                    out_lbl_path = os.path.join(YOLO_TRAIN_LBL, txt_name)
                else:
                    out_img_path = os.path.join(YOLO_VAL_IMG, img_file)
                    out_lbl_path = os.path.join(YOLO_VAL_LBL, txt_name)
                    
                with open(out_lbl_path, "w") as f:
                    f.write("\n".join(yolo_labels))
                
                if not os.path.exists(out_img_path):
                    shutil.copy(img_path, out_img_path)

if __name__ == "__main__":
    convert_to_yolo()
    
    # Create dataset.yaml
    yaml_content = f"""path: {YOLO_DIR}
train: images/train
val: images/val
nc: 1
names:
  0: human
"""
    with open(os.path.join(YOLO_DIR, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Dataset converted to YOLO format.")
