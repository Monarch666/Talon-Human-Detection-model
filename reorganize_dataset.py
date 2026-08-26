import os
import shutil

src_dir = r"e:\Human Detection Dataset\yolo_dataset"
dst_dir = r"e:\Human Detection Dataset\rfdetr_dataset"

os.makedirs(dst_dir, exist_ok=True)
for split in ["train", "valid"]:
    os.makedirs(os.path.join(dst_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, split, "labels"), exist_ok=True)

# Copy train images and labels
src_train_img = os.path.join(src_dir, "images", "train")
src_train_lbl = os.path.join(src_dir, "labels", "train")

dst_train_img = os.path.join(dst_dir, "train", "images")
dst_train_lbl = os.path.join(dst_dir, "train", "labels")

# In our case we pointed valid to train
dst_valid_img = os.path.join(dst_dir, "valid", "images")
dst_valid_lbl = os.path.join(dst_dir, "valid", "labels")

# Since moving is faster than copying and we don't need yolo_dataset anymore, let's just move everything.
def move_files(src, dst):
    if os.path.exists(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if not os.path.exists(d):
                shutil.move(s, d)

print("Moving train...")
move_files(src_train_img, dst_train_img)
move_files(src_train_lbl, dst_train_lbl)

# Since valid is same as train for now, just copy it
print("Copying to valid...")
def copy_files(src, dst):
    if os.path.exists(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if not os.path.exists(d):
                shutil.copy2(s, d)
                
copy_files(dst_train_img, dst_valid_img)
copy_files(dst_train_lbl, dst_valid_lbl)

# Create data.yaml
yaml_content = f"""train: train/images
val: valid/images
nc: 1
names: ['human']
"""
with open(os.path.join(dst_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("Done reorganizing dataset!")
