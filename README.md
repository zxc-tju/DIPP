# DIPP-Differentiable Integrated Prediction and Planning
This repo is the implementation of the following paper:

**Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Jingda Wu](https://wujingda.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Paper]](https://ieeexplore.ieee.org/document/10154577/)**&nbsp;**[[arXiv]](https://arxiv.org/abs/2207.10422)**&nbsp;**[[Project Website]](https://mczhi.github.io/DIPP/)**

## Dataset
Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1; only the files in ```uncompressed/scenario/training_20s``` are needed. Place the downloaded files into training and testing folders separately.

## Installation
### Install dependency
```bash
sudo apt-get install libsuitesparse-dev
```

### Create conda env
zxc: delete - scikit-sparse==0.4.6 in the yml file before run conda create!
```bash
conda env create -f environment.yml
conda activate DIPP
```

### Install Theseus

zxc: Install cuda 11.3 in advance!

Install the [Theseus library](https://github.com/facebookresearch/theseus), follow the guidelines.
zxc: pip install theseus-ai

zxc: pip install functorch
zxc: uninstall scikit-sparse
zxc: pip install scikit-sparse==0.4.11

## Usage
### Data Processing
Run ```data_process.py``` to process the raw data for training. This will convert the original data format into a set of ```.npz``` files, each containing the data of a scene with the AV and surrounding agents. You need to specify the file path to the original data ```--load_path``` and the path to save the processed data ```--save_path``` . You can optionally set ```--use_multiprocessing``` to speed up the processing. 
```shell
python data_process.py \
--load_path /path/to/original/data \
--save_path /output/path/to/processed/data \
--use_multiprocessing
```

### Training
Run ```train.py``` to learn the predictor and planner (if set ```--use_planning```). You need to specify the file paths to training data ```--train_set``` and validation data ```--valid_set```. Leave other arguments vacant to use the default setting.
```shell
python train.py \
--name _5_pre_train_10_percent_step_1 \
--train_set /home/zxc/Documents/data/Waymo_sample/processed_normalized_10percent \
--valid_set /home/zxc/Documents/data/Waymo_sample/processed_normalized_10percent \
--use_planning \
--pretrain_epochs 1 \
--train_epochs 40 \
--batch_size 32 \
--learning_rate 2e-4 \
--future_model SelfAttention \
--device cuda:0
```

# for server
```shell 
python train.py --name _5_pre_train_10_percent_step_1 --train_set /mnt/workspace/data/processed_normalized_10percent --valid_set /mnt/workspace/data/processed_normalized_10percent --use_planning --pretrain_epochs 5 --train_epochs 40 --batch_size 128 --learning_rate 2e-4 --future_model CrossTransformer --device cuda:0
```


### Open-loop testing
Run ```open_loop_test.py``` to test the trained planner in an open-loop manner. You need to specify the path to the original test dataset ```--test_set``` (path to the folder) and also the file path to the trained model ```--model_path```. Set ```--render``` to visualize the results and set ```--save``` to save the rendered images.
```shell
python open_loop_test.py \
--name open_loop \
--test_set /path/to/original/test/data \
--model_path /path/to/saved/model \
--use_planning \
--render \
--save \
--device cpu
```

### Closed-loop testing
Run ```closed_loop_test.py``` to do closed-loop testing. You need to specify the file path to the original test data ```--test_file``` (a single file) and also the file path to the trained model ```--model_path```. Set ```--render``` to visualize the results and set ```--save``` to save the videos.
```shell
python closed_loop_test.py \
--name closed_loop \
--test_file /path/to/original/test/data \
--model_path /path/to/saved/model \
--use_planning \
--render \
--save \
--device cpu
```

## Citation
If you find our repo or our paper useful, please use the following citation:
```
@article{huang2023differentiable,
  title={Differentiable integrated motion prediction and planning with learnable cost function for autonomous driving},
  author={Huang, Zhiyu and Liu, Haochen and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
