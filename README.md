# Unraveling Instance Associations: A Closer Look for Audio-Visual Segmentation
Official code for "Unraveling Instance Associations: A Closer Look for Audio-Visual Segmentation"

> **Unraveling Instance Associations: A Closer Look for Audio-Visual Segmentation**,<br />
> [Yuanhong Chen*](https://scholar.google.com/citations?user=PiWKAx0AAAAJ&hl=en&oi=ao), [Yuyuan Liu*](https://scholar.google.com/citations?user=SibDXFQAAAAJ&hl=zh-CN), [Hu Wang](https://huwang01.github.io/), [Fengbei Liu](https://fbladl.github.io/), [Chong Wang](https://scholar.google.com/citations?user=IWcTej4AAAAJ&hl=en&oi=ao), Helen Frazer, [Gustavo Carneiro](https://www.surrey.ac.uk/people/gustavo-carneiro).            
> *CVPR 2024 ([arXiv 2304.02970](https://arxiv.org/abs/2304.02970))*

<!-- This work presents VPO and CAVP -->
## Dataset
*VPO datasets are available [`here`](https://drive.google.com/file/d/12jq7-Ke09ZPoUI1od44q97DNLrThoHc3/view?usp=sharing)*

*VGGSound audio files are available [`here`](https://drive.google.com/file/d/1-OB3E9qbanfvZGbxvmRL05hsxwD0YOPq/view?usp=sharing)*

![vpo](./figs/avs_vpo_dataset.png)
*Visual comparison between datasets. We show four audio-visual classes, including “female”, “cat”, “dog”, and “car”. The AVSBench (SS) (1st frame) provides pixel-level multi-class annotations to the images containing a single sounding object.  The proposed VPO benchmarks (2nd frame to 4th frame) pair a subset of the segmented objects in an image with relevant audio files to produce pixel-level multi-class annotations.
*


## Demon

https://github.com/user-attachments/assets/e113d3a7-cbb4-4696-941b-4e5966870bee


https://github.com/user-attachments/assets/821e3c55-7daf-4445-a0df-a869cba37d59


https://github.com/user-attachments/assets/d80d8a75-c038-4169-b40d-261a40767c31



## Results

Please note that all the tables in the original paper use conventional semantic segmentaiton (per-dataset) mIoU and F-score metrics. We update the following table based on per-image mIoU and per-video F-score based on [TPAVI](https://github.com/OpenNLPLab/AVSBench). Please note that the current repository version uses AVSBench-Semantics to facilitate training and evaluation on the AVSBench-Objects dataset. However, the label noise in AVSBench-Semantics may affect the final results on the AVSBench-Objects dataset. Therefore, it is recommended to use the original AVSBench-Objects dataset instead.

|                                        | AVSBench-Object (SS) |         | AVSBench-Object (MS) |         | AVSBench-Semantics |         |
|----------------------------------------|----------------------|---------|----------------------|---------|--------------------|---------|
|                                        | mIoU                 | F-Score | mIoU                 | F-Score | mIoU               | F-Score |
| TPAVI~\cite{zhou2022audio}             | 72.79                | 84.80   | 47.88                | 57.80   | 20.18              | 25.20   |
| AVSBG~\cite{hao2023improving}          | 74.13                | 85.40   | 44.95                | 56.80   | -                  | -       |
| ECMVAE~\cite{mao2023multimodal}        | 76.33                | 86.50   | 48.69                | 60.70   | -                  | -       |
| DiffusionAVS~\cite{mao2023contrastive} | 75.80                | 86.90   | 49.77                | 62.10   | -                  | -       |
| CAVP - CNN                             | 76.94                | 87.31   | 52.68                | 64.87   | 30.37              | 35.29   |
| CAVP - Transformer                     |                      |         |                      |         |                    |         |


avsbench-object-ss-224\
https://drive.google.com/file/d/1uBteMNAO_dXVgN8yzII901469jfJxcmO/view?usp=drive_link

avsbench-object-ms-224\
https://drive.google.com/file/d/1ZWE8dJV_uLfMrbnvZ8NW6GGwXM8-QhxA/view?usp=drive_link

avss-224\
https://drive.google.com/file/d/1DwVw_NtDv23QacpNvKlabWSnnPy25xfr/view?usp=drive_link






## Checkpoints


## Usage
### Requirements
```
git clone git@github.com:cyh-0/CAVP.git
cd CAVP
pip install -r requirements.txt
```
### Path
```
ln -s /path/to/datasets ../audio_visual
ln -s /path/to/ckpts ./ckpts
```

### Training
Before training, you need to update your own WANDB_KEY in the config file.


Training scripts for **AVSBench-Semantic**. 
```
python main_avss.py --experiment_name "CAVP" --setup avss --gpus 1 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 --epochs 80 --wandb_mode disabled --num_workers 16
```

Training scripts for **VPO-MONO**. 
```
python main_vpo_mono.py --experiment_name "CAVP" --setup "vpo_ss" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_mono.py --experiment_name "CAVP" --setup "vpo_ms" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_mono.py --experiment_name "CAVP" --setup "vpo_msmi" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online
```

Training scripts for **VPO-STEREO**. 
```
python main_vpo_stereo.py --experiment_name "CAVP" --setup "vpo_ss" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_stereo.py --experiment_name "CAVP" --setup "vpo_ms" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_stereo.py --experiment_name "CAVP" --setup "vpo_msmi" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online
```




## Citation
```
@misc{chen2024unraveling,
      title={Unraveling Instance Associations: A Closer Look for Audio-Visual Segmentation}, 
      author={Yuanhong Chen and Yuyuan Liu and Hu Wang and Fengbei Liu and Chong Wang and Helen Frazer and Gustavo Carneiro},
      year={2024},
      eprint={2304.02970},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
