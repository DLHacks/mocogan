# [WIP]MoCoGAN: Decomposing Motion and Content for Video Generation

- A pytorch implrmention of MoCoGAN
- authors: Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz
- arxiv: https://arxiv.org/abs/1707.04993
- project page: https://github.com/sergeytulyakov/mocogan

## Implemention

- you need to make dirs 'raw_data' and 'resized_data', and place datasets([run, walk, skip, ......]) under 'raw_data' dir.
- datasets are available at http://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html
- `python resize.py`: resize your raw data as 96x96 and place them under the resized_data dir. This command requires ffmpeg


## References

- L.Gorelick,M.Blank,E.Shechtman,M.Irani,andR.Basri.
Actions as space-time shapes. PAMI, 29(12):2247–2253,
2007. http://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html (datasets)
- https://github.com/pytorch/examples/tree/master/dcgan (partly using their model architecture)) 

