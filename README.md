# Individual and Structural Graph Information Bottlenecks for Out-of-Distribution Generalization

Official Implementation for our TKDE paper [Individual and Structural Graph Information Bottlenecks for Out-of-Distribution Generalization](https://arxiv.org/abs/2306.15902). This work focuses on distribution shifts on graph data (Graph OOD) for both graph- and node-level prediction tasks, and proposes a new method Individual and Structural Graph Information Bottlenecks (**IS-GIB**) for out-of-distribution generalization.

## Overview of IS-GIB
<img width="567" alt="image" src="https://github.com/YangLing0818/GraphOOD/assets/62683396/d1e57948-eef6-4110-9c01-61ea8a8ad8eb">

## Getting Started
### Installation
Git clone our repository, and install the required packages with the following command
```
git clone https://github.com/YangLing0818/GraphOOD.git
cd GraphOOD
pip install -r requirements.txt
```
### Dataset
We provide a sample dataset Twitch in this repo, for more datasets, please download them through the Google drive:
```
https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing
```

## Training and Evaluation
We provide the sample script for training our IS-GIB. For example, the training script for Twitch dataset is
```
python run_train.py \
  --device 0 --log_steps 100 --epochs 3000 --runs 10 \
  --dataset twitch \
  --batch_runs_base_dir ./batch_runs/twitch_expr/ \
  --train_graph_list "['DE', 'ES', 'FR']" \
  --val_graph_list "['ENGB']" \
  --metric roc_auc \
  --append_best_file best_test.txt \
  --cross_graph_label_rel_IB_loss \
  --first_last_layer_IB_loss \
  --first_last_layer_2nd_IB_loss \
  --lr 1e-4 \
  --save_model_dir ./saved_models/twitch_expr_model
```

If you found the codes and datasets are useful, please cite our paper
```
@article{yang2023individual,
  title={Individual and Structural Graph Information Bottlenecks for Out-of-Distribution Generalization},
  author={Yang, Ling and Zheng, Jiayi and Wang, Heyuan and Liu, Zhongyi and Huang, Zhilin and Hong, Shenda and Zhang, Wentao and Cui, Bin},
  journal={arXiv preprint arXiv:2306.15902},
  year={2023}
}
```
