# Semi-LLIE: Semi-supervised Contrastive Learning with Mamba-based Low-light Image Enhancement

## Introduction
This is the official repository for our recent paper, "Semi-LLIE: Semi-supervised Contrastive Learning with Mamba-based Low-light Image Enhancement"

## Abstract
Despite the impressive advancements made in recent low-light image enhancement techniques, the scarcity of paired data has emerged as a significant obstacle to further advancements. This work proposes a mean-teacher-based semi-supervised low-light enhancement (Semi-LLIE) framework that integrates the unpaired data into model training. The mean-teacher technique is a prominent semi-supervised learning method, successfully adopted for addressing high-level and low-level vision tasks. However, two primary issues hinder the naive mean-teacher method from attaining optimal performance in low-light image enhancement. Firstly,  pixel-wise consistency loss is insufficient for transferring realistic illumination distribution from the teacher to the student model, which results in color cast in the enhanced images. Secondly, cutting-edge image enhancement approaches fail to effectively cooperate with the mean-teacher framework to restore detailed information in dark areas due to their tendency to overlook modeling structured information within local regions. To mitigate the above issues, we first introduce a semantic-aware contrastive loss to faithfully transfer the illumination distribution, contributing to enhancing images with natural colors. 
Then, we design a Mamba-based low-light image enhancement backbone to effectively enhance Mamba's local region pixel relationship representation ability with a multi-scale feature learning scheme, facilitating the generation of images with rich textural details. 
Further, we propose novel perceptive loss based on the large-scale vision-language Recognize Anything Model (RAM) to help generate enhanced images with richer textual details.
The experimental results indicate that our Semi-LLIE surpasses existing methods in both quantitative and qualitative metrics.

<img src='overview.png'>

<p align="center">Figure 1. An overview of our approach.</p>

## Dependencies

- Ubuntu==20.04
- Pytorch==2.0.1
- CUDA==12.4

## Mamba Env Installation
```
conda create -n mambaenv python=3.9
conda activate mambaenv
conda install cudatoolkit==11.8 -c nvidia
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install causal-conv1d -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install mamba_ssm==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --verbose
```

## Other Dependencies

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
pip install opencv-python timm transformers fairscale loralib pyiqa -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```
pip install adamp -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple

mim install mmcv
```
```
pip uninstall timm
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```
pip install Pillow==9.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If the following error occured, please update numpy version to 1.26.4
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject

```
pip install numpy==1.26.4
```
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
@article{li2024semi,
  title={Semi-LLIE: Semi-supervised Contrastive Learning with Mamba-based Low-light Image Enhancement},
  author={Li, Guanlin and Zhang, Ke and Wang, Ting and Li, Ming and Zhao, Bin and Li, Xuelong},
  journal={arXiv preprint arXiv:2409.16604},
  year={2024}
}
```

## Acknowledgement
The training code architecture is based on the [PS-MT](https://github.com/yyliu01/PS-MT) and [DMT-Net](https://github.com/liuye123321/DMT-Net) and thanks for their work.
We also thank for the following repositories: [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch), [UWNR](https://github.com/ephemeral182/uwnr), [MIRNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py), [2022-CVPR-AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/net/DGRN.py), [SPSR](https://github.com/Maclory/SPSR), [Non-Local-Sparse-Attention](https://github.com/HarukiYqM/Non-Local-Sparse-Attention/blob/main/src/model/attention.py), [AFF](https://github.com/YimianDai/open-aff/blob/master/model/fusion.py), [AECR-Net](https://github.com/GlassyWu/AECR-Net/blob/main/models/CR.py), [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset), [UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html), [Sea-thru](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html), [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark), [MMLE](https://github.com/Li-Chongyi/MMLE_code), [PWRNet](https://github.com/huofushuo/PRWNet), [Ucolor](https://github.com/Li-Chongyi/Ucolor), [CWR](https://github.com/JunlinHan/CWR), [FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN)

