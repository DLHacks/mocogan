# MoCoGAN: Decomposing Motion and Content for Video Generation

- Powered by [DL HACKS](http://deeplearning.jp/hacks/)

## Description

- A pytorch implmention of MoCoGAN
- authors: Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz
- arxiv: https://arxiv.org/abs/1707.04993
- project page: https://github.com/sergeytulyakov/mocogan


## Implemention

- you need to place datasets([run, walk, skip, ......]) [here](http://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html)  under 'raw_data' dir.
- `python resize.py`: resize your raw data as 96x96 and place them under resized_data dir.
- `python train.py`: train MoCoGAN.


## Results
| | | | |
|:---|:---|:---|:---|
|![img](https://github.com/DLHacks/mocogan/blob/media/gif/fakeVideo_epoch-2.gif) | ![img](https://github.com/DLHacks/mocogan/blob/media/gif/fakeVideo_epoch-3.gif) | ![img](https://github.com/DLHacks/mocogan/blob/media/gif/fakeVideo_epoch-4.gif) | ![img](https://github.com/DLHacks/mocogan/blob/media/gif/fakeVideo_epoch-6.gif) |


## Requirements

- python 3.5
- pytorch 0.2
- numpy
- skvideo
- ffmgeg (for preprocessing and saving videos)


## References

- L.Gorelick,M.Blank,E.Shechtman,M.Irani,andR.Basri.
Actions as space-time shapes. PAMI, 29(12):2247–2253, 2007. http://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html (datasets)
- https://github.com/pytorch/examples/tree/master/dcgan  

