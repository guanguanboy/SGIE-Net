



# SAM-Based low-light image Enhancement

#### News
- **April 27, 2022:** Codes and pre-trained models are released!

<hr />

> **Abstract:** * Given 
<hr />

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run .


## Training and Evaluation

Training and Testing instructions Image Enhancement are provided in their respective directories. 

training code:

```
nohup ./train_sam.sh Enhancement/Options/Enhancement_SGIENet_gray_illlum_drconve_Lol_w_sam_cs.yml > logs/Enhancement_SGIENet_gray_illlum_drconve_Lol_w_sam_cs.txt &
```

testing code:

```
python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGIENet_gray_illlum_drconve_Lol_w_sam_cs.yml --weights experiments/Enhancement_SGIENet_lol_gray_drconv_illum_sam_1019/models/net_g_latest.pth --dataset Dataset_PairedWithGrayIllumImage
```

## Results
Experiments are performed for different image processing tasks.


## Citation
If you use our code, please consider citing:




## Contact
Should you have any question, please contact liguanlin1229@gmail.com


**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 

## Our Related Works
- Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022. [Paper](https://arxiv.org/abs/2111.09881) | [Code](https://github.com/swz30/Restormer)
- Multi-Stage Progressive Image Restoration, CVPR 2021. [Paper](https://arxiv.org/abs/2102.02808) | [Code](https://github.com/swz30/MPRNet)
- Learning Enriched Features for Real Image Restoration and Enhancement, ECCV 2020. [Paper](https://arxiv.org/abs/2003.06792) | [Code](https://github.com/swz30/MIRNet)
- CycleISP: Real Image Restoration via Improved Data Synthesis, CVPR 2020. [Paper](https://arxiv.org/abs/2003.07761) | [Code](https://github.com/swz30/CycleISP)
