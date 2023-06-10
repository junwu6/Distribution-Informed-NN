# DINO (Distribution-Informed Neural Network)
An implementation for "Distribution-Informed Neural Networks for Domain Adaptation Regression" (NeurIPS'22).

## Environment Requirements
The code has been tested under Python 3.7. The required packages are as follows:
* numpy==1.18.1
* sklearn==0.22.1
* torch==1.4.0
* jax==0.2.26
* neural-tangents==0.3.9

## Data Sets
We used the following data sets in our experiments:
* [dSprites](https://github.com/thuml/Domain-Adaptation-Regression)
* [MPI3D](https://github.com/rr-learning/disentanglement_dataset)

## Acknowledgement
If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{wu2022distribution,
  title={Distribution-Informed Neural Networks for Domain Adaptation Regression},
  author={Wu, Jun and He, Jingrui and Wang, Sheng and Guan, Kaiyu and Ainsworth, Elizabeth},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
