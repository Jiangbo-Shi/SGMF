# A Structure-aware Hierarchical Graph-based Multiple Instance Learning Framework for pT Staging in Histopathological Image ([TMI 2023](https://ieeexplore.ieee.org/abstract/document/10119190))

## Abstract 
Pathological primary tumor (pT) stage focuses on the infiltration degree of the primary tumor to surrounding tissues, which relates to the prognosis and treatment choices. The pT staging relies on the field-of-views from multiple magnifications in the gigapixel images, which makes pixel-level annotation difficult. Therefore, this task is usually formulated as a weakly supervised whole slide image (WSI) classification task with the slide-level label. Existing weakly-supervised classification methods mainly follow the multiple instance learning paradigm, which takes the patches from single magnification as the instances and extracts their morphological features independently. However, they cannot progressively represent the contextual information from multiple magnifications, which is critical for pT staging. Therefore, we propose a structure-aware hierarchical graph-based multi-instance learning framework (SGMF) inspired by the diagnostic process of pathologists. Specifically, a novel graph-based instance organization method is proposed, namely structure-aware hierarchical graph (SAHG), to represent the WSI. Based on that, we design a novel hierarchical attention-based graph representation (HAGR) network to capture the critical patterns for pT staging by learning cross-scale spatial features. Finally, the top nodes of SAHG are aggregated by a global attention layer for bag-level representation. Extensive studies on three large-scale multi-center pT staging datasets with two different cancer types demonstrate the effectiveness of SGMF, which outperforms state-of-the-art up to 5.6% in the F1 score.

## Framework 
![framework_for_response](https://github.com/Jiangbo-Shi/SGMF/assets/60539295/1e236a27-1444-4938-a57c-ace76531b990)

## Case Study
![attention_result](https://github.com/Jiangbo-Shi/SGMF/assets/60539295/144b125b-93c8-4ac5-99a2-0d9610a2abf7)

## Requirments

## Datasets
We will update the dataset download link soon.

## Usage
### Pre-processing
### Training
Run ````scripts/train_20x_10x_TG_slic.sh````
### Prediction
Run ```scripts/eval_20x_10x_slic.sh```

## Citation
If you find our work is helpful for your research, please consider to cite:
```
@article{shi2023pTstaging,
  title={A Structure-aware Hierarchical Graph-based Multiple Instance Learning Framework for pT Staging in Histopathological Image},
  author={Shi, Jiangbo and Tang, Lufei and Li, Yang and Zhang, Xianli and Gao, Zeyu and Zheng, Yefeng and Wang, Chunbao and Gong, Tieliang and Li, Chen},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```
## Acknowledgement
We have great thanks to [CLAM](https://github.com/mahmoodlab/CLAM) and [Patch_GCN](https://github.com/mahmoodlab/Patch-GCN).
