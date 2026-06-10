import os
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch.multiprocessing as mp

# Global model for workers
mtcnn = None


def init_worker():
    global mtcnn
    mtcnn = MTCNN(select_largest=False, device="cuda", image_size=224, margin=0)


def imread(path):
    # Read image using cv2 and convert to RGB
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imsave(path, img):
    # Save image. img can be numpy array or PIL Image
    if isinstance(img, Image.Image):
        img.save(path)
    else:
        # Assume numpy array in RGB
        # cv2 expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)


def process_imglist(imglist, savepath):
    # Ensure directory exists
    os.makedirs(savepath, exist_ok=True)

    # Sort to ensure deterministic order
    imglist.sort()

    for img_path in imglist:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]

        try:
            frame = imread(img_path)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue

        # Detect face
        try:
            batch_boxes, batch_probs = mtcnn.detect(frame, landmarks=False)
        except RuntimeError as e:
            print(f"Error in handling {img_path}: {e}")
            continue

        if batch_boxes is None:
            continue

        box = batch_boxes[0].astype(int)
        box = np.maximum(box, 0)

        # Crop from original frame using box coordinates
        cropped = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

        # Save crop
        imsave(os.path.join(savepath, f"{base_name}.jpg"), Image.fromarray(cropped))


def process_directory(args):
    src_dir, png_files, root_dir, output_dir = args

    # Construct output path mirroring the structure
    rel_path = os.path.relpath(src_dir, root_dir)
    save_path = os.path.join(output_dir, rel_path)

    # Check if we should skip
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        return

    # Create full paths
    img_paths = [os.path.join(src_dir, f) for f in png_files]

    process_imglist(img_paths, save_path)


def main():
    parser = argparse.ArgumentParser(description="Process Idiap dataset frames")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/datasets/OCIM/Idiap/frames",
        help="Root directory containing extracted frames",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/datasets/OCIM/Idiap/faces_2",
        help="Output directory for cropped faces",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of worker processes",
    )

    args = parser.parse_args()
    for dataset in ["Oulu", "Msu_mfsd", "Casia_fasd"]:
        root_dir = f"/home/ubuntu/datasets/OCIM/{dataset}/frames"
        output_dir = f"/home/ubuntu/datasets/OCIM/{dataset}/faces_2"

        root_dir = args.root_dir
        output_dir = args.output_dir

        print(f"Scanning {root_dir} for PNG images...")

        # Find all directories containing png files
        dirs_with_images = []
        for file in Path(root_dir).rglob("*"):
            if file.suffix.lower() != ".png":
                continue
            dirs_with_images.append(file.as_posix())

        print(f"Found {len(dirs_with_images)} directories with images.")

        # Prepare tasks
        tasks = []
        for src_dir, png_files in dirs_with_images:
            tasks.append((src_dir, png_files, root_dir, output_dir))

        print(f"Processing with {args.num_workers} workers on {len(tasks)} images...")

        # Use spawn for CUDA compatibility
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        with mp.Pool(processes=args.num_workers, initializer=init_worker) as pool:
            list(tqdm(pool.imap_unordered(process_directory, tasks), total=len(tasks)))


if __name__ == "__main__":
    main()
