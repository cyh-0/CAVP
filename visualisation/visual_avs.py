import os
from PIL import Image
import numpy
import cv2
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas as pd


# path_ = "/media/yuyuan/Applications/dataset/AVSBench/avsbench_single"
path_ = "/mnt/HD/Dataset/audio_visual/avsbench_data_single_yh"
df = pd.read_csv(os.path.join(path_, "data.csv"))
pallete = numpy.load("pallete_avs.npy")
index_table = numpy.load("avs_index_table.npy")
name = "-1q0XHPxqe8_1"

label = (df[df.name == name]["category"]).item().split(',')[0]
idx = list(index_table).index(label)
color = list(pallete[idx])


""" Image """
# image = os.path.join(path_, "frames", name + ".png")
# label = os.path.join(path_, "labels", name + ".png")
audios = os.path.join(path_, "audios", name + ".wav")
# image = numpy.array(Image.open(image).convert("RGB"), dtype=numpy.float) / 255.0
# label = numpy.array(Image.open(label).convert("L"), dtype=numpy.uint8)
# canvas = numpy.zeros_like(image)
# contours, hierarchy = cv2.findContours(
#     label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )
# # label = numpy.expand_dims(label, axis=2).repeat(3, axis=2)
# canvas[label != 0] = color
# canvas /= 255.0
# plt.axis("off")
# image = image * 1.5
# image = cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
# result = image * 0.3 + canvas * 0.7
# plt.imshow(result)
# plt.savefig("{}_visual.png".format(name), bbox_inches="tight", pad_inches=0.03)
# plt.clf()

""" Waveform """
fig, ax = plt.subplots()
fig.set_size_inches(7, 1)

ax.set_facecolor(numpy.array(color) / 255.0)
plt.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    # labelbottom=False
)  # labels along the bottom edge are off

plt.tick_params(
    axis="y",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
    length=0,
)  # labels along the bottom edge are off
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)
wav = read(audios)
wav = wav[1]
plt.xlim(0, len(wav))
plt.plot(wav, color="white")
plt.savefig("{}_audio.png".format(name), bbox_inches="tight", pad_inches=0.03)
