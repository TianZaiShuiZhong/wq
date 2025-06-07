import torch

class Config:
    # 数据配置
    data_root = "labelimg_data/images/train"
    image_size = 512
    batch_size = 8
    
    # 训练配置
    lr = 0.0002
    beta1 = 0.5
    epochs = 200
    latent_dim = 100
    
    # 条件配置
    num_conditions = 5  # 晴天/阴影/变形/半遮挡/污染
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 输出配置
    sample_interval = 100
    output_dir = "results"
