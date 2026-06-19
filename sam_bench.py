#!/usr/bin/env python3
import time
from pathlib import Path
import urllib
import cv2
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
)

# Download from:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth --directory-prefix ~/Downloads/
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH =  Path("~/Downloads/sam_vit_b_01ec64.pth").expanduser()

MODEL_TYPE = "vit_b"
NUM_WARMUP = 2
NUM_RUNS = 10


def download_file(url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total_header = response.getheader("Content-Length")
        total_size = int(total_header) if total_header else None
        block_size = 20 * 1024**2  # 20 MB

        with open(output_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=str(output_path),
        ) as pbar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                pbar.update(len(buffer))




# --------------------------------------------------
# Download weights
# --------------------------------------------------

if not CHECKPOINT_PATH.exists():
    download_file(WEIGHT_URL, CHECKPOINT_PATH)

# --------------------------------------------------
# Download test image
# --------------------------------------------------

img_path = Path("benchmark_image.jpg")

if not img_path.exists():
    download_file(IMAGE_URL, img_path)

image_bgr = cv2.imread(str(img_path))
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print(f"Image shape: {image_rgb.shape}")

# --------------------------------------------------
# Load SAM1
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device:          {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version:    {torch.version.cuda}")
print(f"HIP version:     {torch.version.hip}")



sam = sam_model_registry[MODEL_TYPE](
    checkpoint=CHECKPOINT_PATH
)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
)

# --------------------------------------------------
# Warmup
# --------------------------------------------------

for _ in range(NUM_WARMUP):
    _ = mask_generator.generate(image_rgb)

if device == "cuda":
    torch.cuda.synchronize()

# --------------------------------------------------
# Benchmark
# --------------------------------------------------

times = []
mask_counts = []

for i in range(NUM_RUNS):

    start = time.perf_counter()

    masks = mask_generator.generate(image_rgb)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    times.append(elapsed)
    mask_counts.append(len(masks))

    print(
        f"Run {i+1:02d}: "
        f"{elapsed:.3f}s, "
        f"{len(masks)} masks"
    )

# --------------------------------------------------
# Results
# --------------------------------------------------

avg_latency = sum(times) / len(times)
throughput = 1.0 / avg_latency

print("\n=== SAM1 Benchmark Results ===")
print(f"Model: {MODEL_TYPE}")
print(f"Average latency: {avg_latency:.3f} sec/image")
print(f"Throughput:      {throughput:.2f} images/sec")
print(f"Average masks:   {sum(mask_counts)/len(mask_counts):.1f}")

if device == "cuda":
    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {mem_gb:.2f} GB")
