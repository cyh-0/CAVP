# A Closer Look at Audio-Visual Semantic Segmentation
Official code for "A Closer Look at Audio-Visual Semantic Segmentation"


> **A Closer Look at Audio-Visual Semantic Segmentation**,<br />
> [Yuanhong Chen*](https://scholar.google.com/citations?user=PiWKAx0AAAAJ&hl=en&oi=ao), [Yuyuan Liu](https://scholar.google.com/citations?user=SibDXFQAAAAJ&hl=zh-CN), [Hu Wang](https://huwang01.github.io/), [Fengbei Liu*](https://fbladl.github.io/), [Chong Wang](https://scholar.google.com/citations?user=IWcTej4AAAAJ&hl=en&oi=ao), [Gustavo Carneiro](https://www.surrey.ac.uk/people/gustavo-carneiro).            
> *([arXiv 2203.01937](https://arxiv.org/abs/2203.01937))*



![motivation](./figs/avs_bench-motivation-1.png)
*The TPAVI AVS model tends to be biased to segment specific objects in a scene due to limitations in the training set and in the training process.*


# Dataset
### Visual Post-production (VPO)
We build AVS datasets based on pairs of audio-visual data are obtained by matching images and audio based on the semantic classes
of the visual objects of the images and audio based on the semantic classes of the visual objects of the images. We leverage labelled image data from **COCO**, and audo source from **VGGSound**.

![vpo](./figs/dataset_final-1.png)
*Visual comparison between datasets. We show four audio-visual classes, including “female”, “cat”, “dog”, and “car”. The AVSBench-Single+ (left column) provides pixel-level multi-class annotations to the images containing a single sounding object. The proposed VPO benchmarks (center and right columns) pair a subset of the segmented objects in an
image with relevant audio files to produce pixel-level multi-class annotations.*




### Results