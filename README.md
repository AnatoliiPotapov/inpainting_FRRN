# Kolmogorov team 
### solution for Huawei inpainting hackathon

### Details
Core paper: Progressive Image Inpainting with Full-Resolution Residual Network https://arxiv.org/abs/1907.10478

#### Training
```bash
python train.py
```

#### Predict
```bash
python predict.py --pred_path ../Datasets/Huawei/DATASET_INPAINTING/new_result/ --config_path experiments/config_v2.yml --checkpoint experiments/dummy/InpaintingModel.ckpt-85500 --labels_path ../Datasets/Huawei/DATASET_INPAINTING/test_gt
```

