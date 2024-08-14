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

## Results

Please note that all the tables in the original paper use conventional semantic segmentaiton (per-dataset) mIoU and F-score metrics. We update the following table based on per-image mIoU and per-video F-score based on [TPAVI](https://github.com/OpenNLPLab/AVSBench). Please note that the current repository version uses AVSBench-Semantics to facilitate training and evaluation on the AVSBench-Objects dataset. However, the label noise in AVSBench-Semantics may affect the final results on the AVSBench-Objects dataset. Therefore, it is recommended to use the original AVSBench-Objects dataset instead.

<!-- <tr>
<th></th>
<th colspan="3" style="text-align:center;">AVSBench-Object (SS)</th>
<th colspan="3" style="text-align:center;">AVSBench-Object (MS)</th>
<th colspan="3" style="text-align:center;">AVSBench-Semantics</th>
</tr> -->

### Instance-level Evaluation (AVSBench Metrics)
<table>
  <tr>
      <th colspan="10" style="text-align:center;">RESNET-50 (IMGNET PRETRAIN)</th>
  </tr>
  <tr>
    <th></th>
    <th colspan="3" style="text-align:center;">AVSBench-Object (SS)</th>
    <th colspan="3" style="text-align:center;">AVSBench-Object (MS)</th>
    <th colspan="3" style="text-align:center;">AVSBench-Semantics</th>
  </tr>
    <tr>
        <td>Model</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
    </tr>
    <tr>
        <td>CATR</td>
        <td>80.70</td>
        <td>74.80</td>
        <td>86.60</td>
        <td>59.05</td>
        <td>52.80</td>
        <td>65.30</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AuTR</td>
        <td>80.10</td>
        <td>75.00</td>
        <td>85.20</td>
        <td>55.30</td>
        <td>49.40</td>
        <td>61.20</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AVSegFormer</td>
        <td>80.67</td>
        <td>76.54</td>
        <td>84.80</td>
        <td>56.17</td>
        <td>49.53</td>
        <td>62.80</td>
        <td>27.12</td>
        <td>24.93</td>
        <td>29.30</td>
    </tr>
    <tr>
        <td>AVSC~\cite{liu2023audiovisual}</td>
        <td>81.13</td>
        <td>77.02</td>
        <td>85.24</td>
        <td>55.55</td>
        <td>49.58</td>
        <td>61.51</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>BAVS</td>
        <td>81.63</td>
        <td>77.96</td>
        <td>85.29</td>
        <td>56.30</td>
        <td>50.23</td>
        <td>62.37</td>
        <td>27.16</td>
        <td>24.68</td>
        <td>29.63</td>
    </tr>
    <tr>
        <td>TPAVI</td>
        <td>78.80</td>
        <td>72.79</td>
        <td>84.80</td>
        <td>52.84</td>
        <td>47.88</td>
        <td>57.80</td>
        <td>22.69</td>
        <td>20.18</td>
        <td>25.20</td>
    </tr>
    <tr>
        <td>AVSBG</td>
        <td>79.77</td>
        <td>74.13</td>
        <td>85.40</td>
        <td>50.88</td>
        <td>44.95</td>
        <td>56.80</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ECMVAE</td>
        <td>81.42</td>
        <td>76.33</td>
        <td>86.50</td>
        <td>54.70</td>
        <td>48.69</td>
        <td>60.70</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>DiffusionAVS</td>
        <td>81.35</td>
        <td>75.80</td>
        <td>86.90</td>
        <td>55.94</td>
        <td>49.77</td>
        <td>62.10</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>CAVP</td>
        <td>83.84</td>
        <td>78.78</td>
        <td>88.89</td>
        <td>61.48</td>
        <td>55.82</td>
        <td>67.14</td>
        <td>32.83</td>
        <td>30.37</td>
        <td>35.29</td>
    </tr>
</table>

<table>
  <tr>
      <th colspan="7" style="text-align:center;">RESNET-50 (IMGNET PRETRAIN)</th>
  </tr>
  <tr>
    <th></th>
    <th colspan="3" style="text-align:center;">AVSBench-Object (SS)</th>
    <th colspan="3" style="text-align:center;">AVSBench-Object (MS)</th>
  </tr>
    <tr>
        <td>Model</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
    </tr>
    <tr>
        <td>AQFormer</td>
        <td>81.70</td>
        <td>77.00</td>
        <td>86.40</td>
        <td>61.30</td>
        <td>55.70</td>
        <td>66.90</td>
    </tr>
        <tr>
        <td>CAVP</td>
        <td>83.75</td>
        <td>78.72</td>
        <td>88.77</td>
        <td>62.34</td>
        <td>56.42</td>
        <td>68.25</td>
    </tr>    
</table>


### Dataset-level Evaluation (Convention Semantic Segmentation Metrics)
<table>
  <tr>
      <th colspan="7" style="text-align:center;">RESNET-50 (COCO PRETRAIN)</th>
  </tr>
  <tr>
    <th></th>
    <th colspan="3" style="text-align:center;">AVSBench-Object (SS)</th>
    <th colspan="3" style="text-align:center;">AVSBench-Object (MS)</th>
  </tr>
    <tr>
        <td>Model</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
        <td>J&amp;F Mean</td>
        <td>J Mean</td>
        <td>F Mean</td>
    </tr>
    <tr>
        <td>CAVP</td>
        <td>89.43</td>
        <td>94.50</td>
        <td>72.79</td>
        <td>83.05</td>
        <td>44.70</td>
        <td>57.76</td>
    </tr>    
</table>



## Demon

https://github.com/user-attachments/assets/e113d3a7-cbb4-4696-941b-4e5966870bee

https://github.com/user-attachments/assets/821e3c55-7daf-4445-a0df-a869cba37d59

https://github.com/user-attachments/assets/d80d8a75-c038-4169-b40d-261a40767c31


## Checkpoints
Checkpoints are available here:
[avsbench-object-ss-224](https://drive.google.com/file/d/1JDC8jDj4iQT5qeJ_8Xt4zP3oWS-q5Hel/view?usp=drive_link), 
[avsbench-object-ms-224](https://drive.google.com/file/d/1SSMTRDjgkaIgYx8ETpk3sE1dcUe1O5js/view?usp=drive_link), 
[avss-224](https://drive.google.com/file/d/1DwVw_NtDv23QacpNvKlabWSnnPy25xfr/view?usp=drive_link).


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
