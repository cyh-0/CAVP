import os
from PIL import Image
import numpy
import cv2
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas

path_ = "/mnt/HD/Dataset/audio_visual/avsbench_semantic/v2/D9LbDE-1FLQ_172000_182000"
# name = "116048".zfill(12)
# female [255, 182, 193], dog [75, 0, 130], cat [165,42,42], car [255, 191, 0]
table_ = {112: [75, 0, 130], 119: [255, 182, 193], 75: [165,42,42], 113: [255, 191, 0]}

f_num = 5
image = os.path.join(path_, "frames", f"{f_num}.jpg")
label = os.path.join(path_, "labels_rgb", f"{f_num}.png")
audio = os.path.join(path_, "audio.wav")

plt.clf()
plt.figure(figsize=(8.5,0.5))
wav = read(audio)
wav = wav[1]
wav = wav[:, 0]
# wav = wav[int(len(wav) / 2) : int(len(wav) / 2 + 16000 * 1.5)]
plt.xlim(0, len(wav))
plt.plot(wav, color="black")
# plt.show()
plt.axis("off")
plt.savefig("{}_audio.png".format("D9LbDE-1FLQ_172000_182000"), bbox_inches="tight", pad_inches=0.03)




# file_lists = os.path.join(path_, "mask")
# color = 
# csv = pandas.read_csv("/media/yuyuan/Applications/dataset/Union_AV/COCO_AV_MS/coco_multi_source.csv")
v2_pallete = numpy.load("v2_pallete.npy")

image = numpy.array(Image.open(image).convert('RGB'), dtype=numpy.float) / 255.
# label = numpy.array(Image.open(label).convert('L'), dtype=numpy.uint8)

label = os.path.join(path_, "labels_rgb", f"{f_num}.png")
label = numpy.array(Image.open(label).convert('RGB'), dtype=numpy.float)
label[numpy.all(label == (64.,192.,0.), axis=-1)] = (32, 128, 128)
label = label / 255.



# image = numpy.array(Image.open(image).convert("RGB"), dtype=numpy.float) / 255.0
# label = numpy.array(Image.open(label).convert("L"), dtype=numpy.uint8)

canvas = numpy.zeros_like(image)
# contours, hierarchy = cv2.findContours(
#     label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )
# label = numpy.expand_dims(label, axis=2).repeat(3, axis=2)
# canvas[label != 0] = color
canvas /= 255.0
plt.axis("off")
image = image * 1.5
# image = cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
result = image * 0.3 + label * 0.7
plt.imshow(result)
plt.savefig("{}_visual.png".format(f"1FLQ_172000_182000_{f_num}"), bbox_inches="tight", pad_inches=0.02)
plt.clf()



# label = os.path.join(path_, "labels_rgb", f"{f_num}.png")
# label = numpy.array(Image.open(label).convert('RGB'), dtype=numpy.float)
# label[numpy.all(label == (64.,192.,0.), axis=-1)] = (32, 128, 128)
# Image.fromarray(label.astype(numpy.uint8)).show()