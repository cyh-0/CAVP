import cv2
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
from glob import glob
root = "/home/yuanhong/Documents/audio_visual_project/ICCV2023/yy_audio/wandb/offline-run-20230529_220226-1vmo5tpi/files/media/images/6L38ny23NPU_2000_7000"
image_folder = root
video_name = 'video.mp4'
# image_folder='folder_with_images'
fps=1

# image_files = [os.path.join(image_folder,img)
#                for img in os.listdir(image_folder)
#                if img.endswith(".png")]
image_files = glob(os.path.join(image_folder, "y_tilde*"))
print(os.path.join(image_folder, "./y_tilde*"))
print(image_files)   
clip = ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('my_video.mp4')