import os
from PIL import Image
import numpy
import cv2
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas

path_ = "/mnt/HD/Dataset/audio_visual/Union_AV/COCO_AV"
class_ = "male"
name = 48126
name = str(name).zfill(12)
# female [255, 182, 193], dog [75, 0, 130], cat [165,42,42], car [255, 191, 0]
# male = [128, 192, 128]
color = [64,128,128]
image = os.path.join(path_, "data", class_, name + ".jpg")
file_lists = os.listdir(
    "/mnt/HD/Dataset/audio_visual/Union_AV/COCO_AV/mask/{}".format(class_)
)
audio_lists = os.listdir()
label = [i for i in file_lists if name in i][0]
label = os.path.join(path_, "mask", class_, label)
csv = pandas.read_csv("/mnt/HD/Dataset/audio_visual/Union_AV/COCO_AV/coco_av_train.csv")

# audios = csv[csv['img_Id'] == int(name.strip())]["vgg_file"].item()+".wav"

print(csv[csv["img_Id"] == int(name.strip())]["vgg_label"])
audios = csv[csv["img_Id"] == int(name.strip())].iloc[0]["vgg_file"] + ".wav"

audios = os.path.join(
    "/mnt/HD/Dataset/audio_visual/vggsound_bench/VGGSound/audios/", audios
)
image = numpy.array(Image.open(image).convert("RGB"), dtype=numpy.float) / 255.0
label = numpy.array(Image.open(label).convert("L"), dtype=numpy.uint8)
canvas = numpy.zeros_like(image)
contours, hierarchy = cv2.findContours(
    label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
# label = numpy.expand_dims(label, axis=2).repeat(3, axis=2)

canvas[label != 0] = color
canvas /= 255.0

plt.axis("off")

""" White boundary"""
# # canvas[label == 0] = [255, 255, 255]
# image = image / 2
# image = cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
# result = image * 0.15+ canvas * 0.9
# # result = result[:, 90:, :]
""" ORG """
image = image * 1.5
image = cv2.drawContours(image, contours, -1, (255,255,255), 2)
result = image * .3 + canvas * .7
result = result[:, 90:, :]


plt.imshow(result)
# plt.show()
plt.savefig("{}_visual.png".format(name), bbox_inches="tight", pad_inches=0.03)
plt.clf()

# plt.style.use('seaborn-poster')
fig, ax = plt.subplots()
fig.set_size_inches(7, 1)

# plt.axis("off")
# ax = plt.axes()
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
wav = wav[int(len(wav) / 2) : int(len(wav) / 2 + 16000 * 1.5)]
plt.xlim(0, len(wav))
plt.plot(wav, color="white")
# plt.show()
plt.savefig("{}_audio.png".format(name), bbox_inches="tight", pad_inches=0.03)
