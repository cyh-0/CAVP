import os
from PIL import Image
import numpy
import cv2
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas

path_ = "/media/yuyuan/Applications/dataset/Union_AV/COCO_AV_MS"
name = "116048".zfill(12)
# female [255, 182, 193], dog [75, 0, 130], cat [165,42,42], car [255, 191, 0]
table_ = {112: [75, 0, 130], 119: [255, 182, 193], 75: [165,42,42], 113: [255, 191, 0]}

image = os.path.join(path_, "data", name+".jpg")
audio_lists = os.listdir()
label = os.path.join(path_, "mask", name+".png")
file_lists = os.path.join(path_, "mask")
# for i in os.listdir(file_lists):
#     temp = numpy.array(Image.open(os.path.join(file_lists, i)))
#     if 199 in numpy.unique(temp) and 75 in numpy.unique(temp):
#         print(i)
# exit(1)
csv = pandas.read_csv("/media/yuyuan/Applications/dataset/Union_AV/COCO_AV_MS/coco_multi_source.csv")

# 307423   374922

# df = csv.groupby("img_Id")["cateName"].apply(set)
# for i, j in enumerate(df):
#     if "car" in j and "female" in j :
#         print(df.iloc[[i]])
# exit()

audios_1 = csv[csv['img_Id'] == int(name.strip())].iloc[0]["vgg_file"]+".wav"
audios_2 = csv[csv['img_Id'] == int(name.strip())].iloc[1]["vgg_file"]+".wav"

audios_1 = os.path.join("/media/yuyuan/Applications/dataset/vggsound_bench/VGGSound/audios/", audios_1)
audios_2 = os.path.join("/media/yuyuan/Applications/dataset/vggsound_bench/VGGSound/audios/", audios_2)


image = numpy.array(Image.open(image).convert('RGB'), dtype=numpy.float) / 255.
label = numpy.array(Image.open(label).convert('L'), dtype=numpy.uint8)

canvas = numpy.zeros_like(image)
label_a = label.copy()
label_b = label.copy()
label_a[label_a!=112] = 0
label_b[label_b!=113] = 0
contours_a, hierarchy = cv2.findContours(label_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_b, hierarchy = cv2.findContours(label_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# label = numpy.expand_dims(label, axis=2).repeat(3, axis=2)
for i in table_.keys():
    canvas[label == i] = table_[i]
canvas /= 255.
plt.axis("off")
image = image * 1.5
image = cv2.drawContours(image, contours_a, -1, (255,255,255), 2)
image = cv2.drawContours(image, contours_b, -1, (255,255,255), 2)
result = image * .3 + canvas * .7
plt.imshow(result[:, 60:, :])
# plt.show()
plt.savefig("{}_visual.png".format(name), bbox_inches='tight', pad_inches=0.03)
plt.clf()

# plt.style.use('seaborn-poster')
fig, ax = plt.subplots()
fig.set_size_inches(7, 1)

# plt.axis("off")
# ax = plt.axes()
# ax.set_facecolor(numpy.array(table_[112])/255.)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    # labelbottom=False
)  # labels along the bottom edge are off

plt.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
    length=0)  # labels along the bottom edge are off
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)
wav1 = read(audios_1)
wav2 = read(audios_2)
wav = wav1 + wav2
wav = wav[1]
wav = wav[int(len(wav)/2):int(len(wav)/2+16000*1.5)]

plt.xlim(0, len(wav))
ax.margins(0)
offset = (numpy.max(wav) + numpy.abs(numpy.min(wav)))/3
ax.axhspan(numpy.max(wav)-offset, numpy.max(wav), facecolor=numpy.array(table_[112])/255.)
ax.axhspan(numpy.max(wav)-offset, numpy.min(wav)+offset, facecolor=numpy.array(table_[113])/255.)
ax.axhspan(numpy.min(wav)+offset, numpy.min(wav), facecolor=numpy.array(table_[119])/255.)
plt.plot(wav, color="white")
# plt.show()
plt.savefig("{}_audio.png".format(name), bbox_inches='tight', pad_inches=0.03)


