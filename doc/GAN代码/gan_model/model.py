import torch
import torch.nn as nn
import config

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(config.Config.num_conditions, config.Config.num_conditions)
        
        self.init_size = config.Config.image_size // 4
        self.l1 = nn.Sequential(nn.Linear(config.Config.latent_dim + config.Config.num_conditions, 128 * (self.init_size ** 2)))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        # Generate fake images
        # Embed labels (output will be [batch_size, num_conditions])
        embedded_labels = self.label_emb(labels)
        
        # Reshape noise to [batch_size, latent_dim]
        noise = noise.view(noise.size(0), -1)
        
        # Reshape noise to [batch_size, latent_dim]
        noise = noise.view(noise.size(0), config.Config.latent_dim)
        
        # Ensure embedded_labels has shape [batch_size, num_conditions]
        if embedded_labels.dim() != 2:
            embedded_labels = embedded_labels.view(-1, config.Config.num_conditions)
        
        # Concatenate along feature dimension
        gen_input = torch.cat((noise, embedded_labels), dim=1)
        
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(config.Config.num_conditions, config.Config.num_conditions)
        
        self.model = nn.Sequential(
            nn.Conv2d(3 + config.Config.num_conditions, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(512),
        )
        
        self.adv_layer = nn.Sequential(nn.Linear(512 * (config.Config.image_size//16) ** 2, 1), nn.Sigmoid())
        
    def forward(self, img, labels):
        # 将条件标签转换为图像大小
        embedded_labels = self.label_embedding(labels)
        if embedded_labels.dim() == 1:
            embedded_labels = embedded_labels.unsqueeze(1)
        # 确保embedded_labels形状为[batch, num_cond]
        embedded_labels = embedded_labels.view(embedded_labels.size(0), -1)
        # 调整条件标签维度以匹配图像
        c = embedded_labels.unsqueeze(2).unsqueeze(3)  # [batch, num_cond, 1, 1]
        c = c.expand(-1, -1, img.size(2), img.size(3))  # [batch, num_cond, H, W]
        
        # 确保条件标签通道数与配置一致
        if c.size(1) > config.Config.num_conditions:
            c = c[:, :config.Config.num_conditions, :, :]
        elif c.size(1) < config.Config.num_conditions:
            # 如果不足则补零
            padding = torch.zeros(c.size(0), 
                              config.Config.num_conditions - c.size(1),
                              c.size(2), 
                              c.size(3)).to(img.device)
            c = torch.cat([c, padding], dim=1)
            
        # 将条件与图像连接
        try:
            d_in = torch.cat((img, c), dim=1)
        except RuntimeError as e:
            print(f"Image shape: {img.shape}")
            print(f"Condition shape: {c.shape}")
            raise e
        out = self.model(d_in)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity
