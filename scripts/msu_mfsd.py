import os
from tqdm import tqdm
import shutil

# protocol = "attack_clientID_cameraType_resolution_attackType_scenario"
# attack_client001_android_SD_ipad_video_scene01

source_dir = "/mnt/cluster/nbl-datasets/spoof-face/OCIM/MSU_MFSD_full/raw/Attack/"

files = list(os.listdir(source_dir))
for file in tqdm(files):
    if file.endswith(".face"):
        continue
    parts = file.split("_")
    attack_type = "_".join(parts[4:-1])
    scenario = parts[-1].split(".")[0]

    target_dir = os.path.join(source_dir, attack_type, scenario)
    os.makedirs(target_dir, exist_ok=True)

    shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))
