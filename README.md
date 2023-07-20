# A2CL-PT
Adversarial Background-Aware Loss for Weakly-supervised Temporal Activity Localization (ECCV 2020)\
[**paper**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590273.pdf) |
[**poster**](https://drive.google.com/file/d/1rLpQkQ3xz5ZndHoOz6IDJE5U4CUv_V7t/view?usp=sharing) | [**presentation**](https://youtu.be/_fwvtSpeplY)

## Overview
We argue that existing methods for weakly-supervised temporal activity localization are not able to sufficiently distinguish background information from activities of interest for each video even though such an ability is critical to strong temporal activity localization. To this end, we propose a novel method named Adversarial and Angular Center Loss with a Pair of Triplets (A2CL-PT). Our method outperforms all the previous state-of-the-art approaches. Specifically, the average mAP of IoU thresholds from 0.1 to 0.9 on THUMOS14 dataset is significantly improved from 27.9% to 30.0%.

| Method \ mAP(%) | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | @0.8 | @0.9 | AVG |
|:----------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [UntrimmedNet](https://arxiv.org/abs/1703.03329) | 44.4 | 37.7 | 28.2 | 21.1 | 13.7 | - | - | - | - | - |
| [STPN](https://arxiv.org/abs/1712.05080) | 52.0 | 44.7 | 35.5 | 25.8 | 16.9 | 9.9 | 4.3 | 1.2 | 0.1 | 21.2 |
| [W-TALC](https://arxiv.org/abs/1807.10418) | 55.2 | 49.6 | 40.1 | 31.1 | 22.8 | - | 7.6 | - | - | - |
| [AutoLoc](https://arxiv.org/abs/1807.08333) | - | - | 35.8 | 29.0 | 21.2 | 13.4 | 5.8 | - | - | - |
| [CleanNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.html) | - | - | 37.0 | 30.9 | 23.9 | 13.9 | 7.1 | - | - | - |
| [MAAN](https://arxiv.org/abs/1905.08586) | 59.8 | 50.8 | 41.1 | 30.6 | 20.3 | 12.0 | 6.9 | 2.6 | 0.2 | 24.9 |
| [BaS-Net](https://arxiv.org/abs/1911.09963) | 58.2 | 52.3 | 44.6 | 36.0 | 27.0 | 18.6 | 10.4 | 3.9 | 0.5 | 27.9 |
| [**A2CL-PT (Ours)**](https://link.springer.com/chapter/10.1007%2F978-3-030-58568-6_17) | **61.2** | **56.1** | **48.1** | **39.0** | **30.1** | **19.2** | **10.6** | **4.8** | **1.0** | **30.0** |

## Weakly-supervised Temporal Activity Localization
The main goal of temporal activity localization is to find the start and end times of activities from untrimmed videos. A weakly-supervised version has recently taken foot in the community: here, one assumes that only video-level groundtruth activity labels are available. These video-level activity annotations are easy to collect and already exist across many datasets, thus weakly-supervised methods can be applied to a broader range of situations.

## Example
![](examples/LongJump.gif)

Full example video clip is included in `examples` folder. You can reproduce the detection results by using `run_example.py`

## Code Usage
First, clone this repository and download these pre-extracted I3D features of the THUMOS14 dataset: [feature\_train.npy](https://drive.google.com/file/d/1PDJtJch7cVgvX-fgyvwl1LN7w8HJUzjk/view?usp=sharing) and [feature\_val.npy](https://drive.google.com/file/d/1OoKwOa-qQAp7cu-UnKnXSjw5O8eTSMWG/view?usp=sharing).
Then, put these files in the `dataset/THUMOS14` folder and just run

`$ python main.py --mode val`

This will reproduce the results reported in the paper.
You can also train the model from scratch by running

`$ python main.py --mode train`

You can refer to the `main.py` file to play with the hyperparameters (margins, alpha, beta, gamma, omega, etc.).

## Notes
- We performed all the experiments with Python 3.6 and PyTorch 1.3.1 on a single GPU (TITAN Xp).

- We also provide the pre-extracted features of ActivityNet-1.3 dataset: [link](https://drive.google.com/drive/folders/1LyypoyYNnJIuN6VYM6CeeCFHHf0gmL8O?usp=drive_link). As described in our paper, you also need to add a 1D grouped convolutional layer (k=13, p=12, d=2). Please refer to this [discussion](https://github.com/MichiganCOG/A2CL-PT/issues/4).

## Citation
```bibtex
@inproceedings{min2020adversarial,
  title={Adversarial Background-Aware Loss for Weakly-supervised Temporal Activity Localization},
  author={Min, Kyle and Corso, Jason J},
  booktitle={European Conference on Computer Vision},
  pages={283--299},
  year={2020},
  organization={Springer}
}
```
