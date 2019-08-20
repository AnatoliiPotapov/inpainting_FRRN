# Kolmogorov team 
### solution for Huawei inpainting hackathon https://huawei-hackathon.moscow/ !
UPD: Second place on public set and third place on private set.
Leaderboard: http://bit.ly/huawei_hack_lb

### Details
Core paper: Progressive Image Inpainting with Full-Resolution Residual Network https://arxiv.org/abs/1907.10478

### Tricks and modifications:

Use Full Resolution Places 365 dataset (http://places2.csail.mit.edu) for training, with categories selected to be visually close on the hackathon train set.

Train two models (v2 and v2_discriminator) 
v2: with l1 loss and l1 step loss (for each block)
v2_discriminator: v2 + PatchGAN + style losses

Final solution: blend of 11 models with equal weights and bluring (see predict.py)

#### Training
```bash
python train.py
```

#### Predict
```bash
python predict.py --pred_path ../Datasets/Huawei/DATASET_INPAINTING/new_result/ --config_path experiments/config_v2.yml --checkpoint experiments/checkpoints/ --masks_path experiments/masks/private/  --images_path ../Datasets/Huawei/DATASET_INPAINTING/test_final/ --cuda '1' --batch_size 10 --blured True
```

To reproduce our solution, you need to download solution.zip from https://drive.google.com/file/d/1BLVtkeiB2EF7mE61vALKV9yP3ti-akT_/view?usp=sharing, then unzip it to experiments/

PPS: for Dataset class to work each folder with pictures should contain files.txt with local filenames

### Made by Tinkoff NLP team.
Anatoly Potapov (*) (research and coordination, choice of paper, implementation of FRRB)
Vasily Karmazin (implemented most of the ideas, introduced bluring)
Gleb Ishelev (helped a lot)
