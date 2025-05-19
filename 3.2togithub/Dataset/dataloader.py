import os
import shutil
import pickle

# Paths based on your provided structure
dataset_root = "Datasets/raw/RIMES-2011-Lines"  # Source dataset path
target_root = "Datasets/formatted"  # Target root for formatted dataset
imgs_path = os.path.join(dataset_root, "Images")
sets_path = os.path.join(dataset_root, "Sets")
transcriptions_path = os.path.join(dataset_root, "Transcriptions")
formatted_path = os.path.join(target_root, "RIMES_2011_line")

# Create formatted dataset directory structure
splits = ['train', 'valid', 'test']
split_dirs = {split: os.path.join(formatted_path, split) for split in splits}
for dir_path in split_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Set files to splits mapping
set_files = {
    'TrainLines': 'train',
    'ValidationLines': 'valid',
    'TestLines': 'test',
}

# Initialize ground truth dictionary with an additional nesting under "ground_truth"
gt = {split: {} for split in splits}
gt_wrapped = {"ground_truth": gt}
charset = set()
# Process each set file and split
for set_file, split in set_files.items():
    with open(os.path.join(sets_path, f"{set_file}.txt"), 'r') as f:
        for line in f:
            filename = line.strip() + ".jpg"  # Adjust the extension if your images have a different format
            src_img_path = os.path.join(imgs_path, filename)
            dst_img_path = os.path.join(split_dirs[split], filename)
            if os.path.exists(src_img_path):
                # Copy the image to the corresponding split folder
                shutil.copy(src_img_path, dst_img_path)
                
                # Read the transcription and update ground truth
                transcription_file = os.path.splitext(filename)[0] + ".txt"
                transcription_path = os.path.join(transcriptions_path, transcription_file)
                if os.path.exists(transcription_path):
                    with open(transcription_path, 'r') as tf:
                        transcription = tf.read().strip()
                        gt_wrapped["ground_truth"][split][filename] = transcription
                        charset.update(set(transcription))  # Update charset with characters from the transcription

charset.add('J')  # Add 'J' explicitly if needed

# After processing all files, save charset and ground truth data
gt_wrapped["charset"] = sorted(list(charset))  # Include charset in the saved data

# Save the ground truth data into a pickle file with the "ground_truth" key
with open(os.path.join(formatted_path, "labels.pkl"), 'wb') as f:
    pickle.dump(gt_wrapped, f)

print("Dataset formatting complete.")