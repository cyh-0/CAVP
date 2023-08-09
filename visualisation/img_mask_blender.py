from PIL import Image
import os


root = "/mnt/HD/Dataset/audio_visual/avsbench_data_single_yh"

fn = "-1XBP-nZ1bQ_1"
image = os.path.join(root, "frame", fn)
mask = os.path.join(root, "labels", fn)

image = Image.open(image)
mask = Image.open(mask)

a=1