# Semantics-aware Contrastive Semi-supervised Learning for Low-light Drone Image Enhancement

*Equal Contributions
+Corresponding Author

Xidian University, McMaster University

## Introduction
This is the official repository for our recent paper, "Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank [link](https://arxiv.org/pdf/2303.09101.pdf)", where more implementation details are presented.

## Abstract
Despite the remarkable achievement of recent underwater image restoration techniques, the lack of labeled data has become a major hurdle for further progress. In this work, we propose a mean-teacher based **Semi**-supervised **U**nderwater **I**mage **R**estoration (**Semi-UIR**) framework to incorporate the unlabeled data into network training. However, the naive mean-teacher method suffers from two main problems: (1) The consistency loss used in training might become ineffective when the teacher's prediction is wrong. (2) Using L1 distance may cause the network to overfit wrong labels, resulting in confirmation bias. To address the above problems, we first introduce a reliable bank to store the ``best-ever" outputs as pseudo ground truth. To assess the quality of outputs, we conduct an empirical analysis based on the monotonicity property to select the most trustworthy NR-IQA method. Besides, in view of the confirmation bias problem, we incorporate contrastive regularization to prevent the overfitting on wrong labels. Experimental results on both full-reference and non-reference underwater benchmarks demonstrate that our algorithm has obvious improvement over SOTA methods quantitatively and qualitatively.

<img src='overview.png'>

<p align="center">Figure 1. An overview of our approach.</p>

## Dependencies

- Ubuntu==18.04
- Pytorch==1.8.1
- CUDA==11.1

Other dependencies are listed in `requirements.txt`

## Data Preparation

Run `data_split.py` to randomly split your paired datasets into training, validation and testing set.

Run `estimate_illumination.py` to get illumination map of the corresponding image.

Finally, the structure of  `data`  are aligned as follows:

```
data
├── labeled
│   ├── input
│   └── GT
│   └── LA
├── unlabeled
│   ├── input
│   └── LA
│   └── candidate
└── val
    ├── input
    └── GT
    └── LA
└── test
    ├── benchmarkA
        ├── input
        └── LA
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

