# Semantics-aware Contrastive Semi-supervised Learning for Low-light Drone Image Enhancement

*Equal Contributions
+Corresponding Author

Xidian University, McMaster University

## Introduction
This is the official repository for our recent paper, "Semantics-aware Contrastive Semi-supervised Learning for Low-light Drone Image Enhancement .

## Abstract
Despite the impressive advancements made in recent low-light image enhancement techniques, the scarcity of annotated data has emerged as a significant obstacle to further advancements. To address this issue, we propose a mean-teacher-based Semi-supervised low-light enhancement framework to utilize the unlabeled data for model optimization. However, the naive implementation of the mean-teacher method encounters two primary challenges. The utilization of pixel-wise distance in the mean-teacher method may lead to the overfitting of incorrect labels, which results in confirmation bias. To mitigate this issue, we introduce semantics-aware contrastive regularization as a preventive measure against overfitting on incorrect labels.

Experimental results demonstrate that our method achieves remarkable quantitative and qualitative improvements over the existing methods.

<img src='overview.png'>

<p align="center">Figure 1. An overview of our approach.</p>

## Dependencies

- Ubuntu==18.04
- Pytorch==1.8.1
- CUDA==11.1

Other dependencies are listed in `requirements.txt`

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

Install Mobile Segment Anything:

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

or clone the repository locally and install with

```
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python timm transformer fairscale loralib pyiqa
```
-i https://pypi.tuna.tsinghua.edu.cn/simple

## Data Preparation

Run `data_split.py` to randomly split your paired datasets into training, validation and testing set.

Run `estimate_illumination.py` to get illumination map of the corresponding image.

Finally, the structure of  `data`  are aligned as follows:

```
data
├── labeled
│   ├── input
│   └── GT
│   
├── unlabeled
│   ├── input
│
└── val
    ├── input
    └── GT
└── test
    ├── benchmarkA
        ├── input
```

You can download the training set and test sets from benchmarks [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset), [UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html), [Sea-thru](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html), [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark). 

## Test

Put your test benchmark under `data/test` folder, run `estimate_illumination.py` to get its illumination map.

Setup the following three paths in `test.py`

```
model_root = 'model/lol_ckpt_begin_0404/model_e200.pth'
input_root = 'data/LOLv1/val'
save_path = 'result/lol_ckpt_begin_0404/'
```

Run `test.py` and you can find results from folder `result`.

```
python test_withgrad.py
```

## Train

To train the framework, run `create_candiate.py` to initialize reliable bank. Hyper-parameters can be modified in `trainer.py`.

Setup the following optioins in `train.py`: 
```
    parser.add_argument('--data_dir', default='./data/LOLv2_real', type=str, help='data root path')
    parser.add_argument('--save_path', default='./model/ckpt_begin_04014_on_LOLv2_real/', type=str)
```

For continue trainning, please setup:
```
    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', default='./model/ckpt_begin_0408_on_visdrone/model_e160.pth', type=str, help='if resume')
```
Run `train.py` to start training.

```
CUDA_VISIBLE_DEVICES=2 nohup python train_lolv1.py --gpus 1 --train_batchsize 6 > logs/train_on_lolv1_visdrone_0414.txt
```

## Citation
If you use the code in this repo for your work, please cite the following bib entries:

```latex
@inproceedings{huang2023contrastive,
  title={Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank},
  author={Huang, Shirui and Wang, Keyan and Liu, Huan and Chen, Jun and Li, Yunsong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18145--18155},
  year={2023}
}
```

## Acknowledgement
The training code architecture is based on the [PS-MT](https://github.com/yyliu01/PS-MT) and [DMT-Net](https://github.com/liuye123321/DMT-Net) and thanks for their work.
We also thank for the following repositories: [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch), [UWNR](https://github.com/ephemeral182/uwnr), [MIRNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py), [2022-CVPR-AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/net/DGRN.py), [SPSR](https://github.com/Maclory/SPSR), [Non-Local-Sparse-Attention](https://github.com/HarukiYqM/Non-Local-Sparse-Attention/blob/main/src/model/attention.py), [AFF](https://github.com/YimianDai/open-aff/blob/master/model/fusion.py), [AECR-Net](https://github.com/GlassyWu/AECR-Net/blob/main/models/CR.py), [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset), [UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html), [Sea-thru](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html), [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark), [MMLE](https://github.com/Li-Chongyi/MMLE_code), [PWRNet](https://github.com/huofushuo/PRWNet), [Ucolor](https://github.com/Li-Chongyi/Ucolor), [CWR](https://github.com/JunlinHan/CWR), [FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN)

