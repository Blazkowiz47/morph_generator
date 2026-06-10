from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import ffmpeg
import os
import shutil


def separate_out_raw() -> None:
    rdir = Path(
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD"
    )
    odir = Path(
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2d_videos"
    )
    raw_odir = odir / "raw"
    for iphone in ["iPhone11", "iPhone12"]:
        print(f"Processing {iphone} dataset...")
        for file in (rdir / iphone).glob("*"):
            if file.is_dir():
                continue
            if "Copy" in file.name or "Guoqian" in file.name:
                continue
            print(f"         {file.name}")

            if "bonafide" in file.name:
                if "video" in file.name:
                    raw_ofile = raw_odir / "real" / file.name
                    raw_ofile.parent.mkdir(parents=True, exist_ok=True)
                    if raw_ofile.exists():
                        continue
                    shutil.copy2(file, raw_ofile)

            else:
                attack_name = "-".join(file.name.split("-")[1:-1]).lower()
                ofile = raw_odir / "attack" / attack_name / file.name
                ofile.parent.mkdir(parents=True, exist_ok=True)
                if ofile.exists():
                    continue
                shutil.copy2(file, ofile)


def extract_7z_files() -> None:
    rdir = Path(
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2d_videos"
    )
    raw_dir = rdir / "raw"
    video_dir = rdir / "videos"
    real_dir = raw_dir / "real"
    for file in real_dir.glob("*.7z"):
        print(f"Extracting {file.name}...")
        odir = video_dir / "real" / file.name.split("-")[0]
        odir.mkdir(parents=True, exist_ok=True)
        os.system(f"cd {odir} && 7z x {file} ")

    attack_dir = raw_dir / "attack"
    for attack in attack_dir.iterdir():
        for file in attack.glob("*.7z"):
            print(f"Extracting {file.name}...")
            odir = video_dir / "attack" / attack.name / file.name.split("-")[0]
            odir.mkdir(parents=True, exist_ok=True)
            os.system(f"cd {odir} && 7z x {file} ")


def extract_videos() -> None:
    rdir = Path(
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2d_videos"
    )
    video_dir = rdir / "videos"
    frames_dir = rdir / "frames"
    args = []
    for file in tqdm(list(video_dir.rglob("*"))):
        if file.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
            continue
        output_dir = Path(
            str(file).replace(str(video_dir), str(frames_dir)).replace(file.suffix, "")
        )
        args.append((file, output_dir))

    with Pool(8) as p:
        _ = list(tqdm(p.imap(extract_frames_wrapper, args), total=len(args)))


def extract_frames(video_path: Path, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(save_dir / "frame_%05d.png")

    try:
        (
            ffmpeg.input(str(video_path))
            .output(output_pattern, **{"qscale:v": 1})
            .run(quiet=True, overwrite_output=True)
        )
        # print(f"✅ Frames extracted from {video_path}")
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg error on {video_path}: {e}")


def extract_frames_wrapper(args):
    return extract_frames(*args)


"""
Processing iPhone11 dataset...
         iPhone11Pro-Display-Attack.7z
         iPhone11Pro-front-bonafide-mugshots-image.7z
         iPhone11Pro-front-bonafide-mugshots-video.7z
         iphone11Pro-Hard-Plastic-Mask-Mugshots.7z
         iPhone11Pro-Latex-Mask-Mugshots.7z
         iPhone11Pro-Paper-Masks-Mugshots.7z
         iPhone11Pro-Print Attack-Mugshots.7z
         iPhone11Pro-Silicon-Mask-Mugshots.7z
         iPhone11Pro-Soft-Plastic-Mask-Mugshots.7z
         iPhone11Pro-Wrap-Attack-Mugshots.7z
Processing iPhone12 dataset...
         iPhone12ProMax-Display-Attack-Mugshots.7z
         iPhone12ProMax-front-bonafide-mugshots-image.7z
         iPhone12ProMax-front-bonafide-mugshots-video.7z
         iPhone12ProMax-Hard-Plastic-Mask-Mugshots.7z
         iPhone12ProMax-Latex-Mask-Mugshots.7z
         iPhone12ProMax-Paper-Mask-Mugshots.7z
         iPhone12ProMax-Print-Attack-Mugshots.7z
         iPhone12ProMax-Silicon-Mask-Mugshots.7z
         iPhone12ProMax-Soft-Plastic-Mask-Mugshots.7z
         iPhone12ProMax-Wrap-Attack-Mugshots.7z
"""


if __name__ == "__main__":
    # separate_out_raw()
    # extract_7z_files()
    extract_videos()
