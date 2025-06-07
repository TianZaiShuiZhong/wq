import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TennisBallDataset
from model import Generator, Discriminator
import config
import os
from torchvision.utils import save_image
import numpy as np

# 创建所有必要的输出目录
os.makedirs(config.Config.output_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 初始化模型
generator = Generator().to(config.Config.device)
discriminator = Discriminator().to(config.Config.device)

# 损失函数和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=config.Config.lr, betas=(config.Config.beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config.Config.lr, betas=(config.Config.beta1, 0.999))

# 数据加载
dataset = TennisBallDataset(config.Config.data_root)
dataloader = DataLoader(dataset, batch_size=config.Config.batch_size, shuffle=True)

# 训练循环
for epoch in range(config.Config.epochs):
    for i, (imgs, conditions) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        
        # 准备真实和假标签
        valid = torch.ones(batch_size, 1).to(config.Config.device)
        fake = torch.zeros(batch_size, 1).to(config.Config.device)
        
        # 转换为设备
        real_imgs = imgs.to(config.Config.device)
        conditions = conditions.to(config.Config.device)
        
        # 训练生成器
        optimizer_G.zero_grad()
        
        # 生成噪声和条件 - 确保batch_size一致
        z = torch.randn(batch_size, config.Config.latent_dim).to(config.Config.device)
        gen_labels = torch.multinomial(torch.ones(batch_size, config.Config.num_conditions), 1).to(config.Config.device)
        
        # 生成图像 - 确保batch_size匹配
        if z.size(0) != gen_labels.size(0):
            gen_labels = gen_labels[:z.size(0)]
        gen_imgs = generator(z, gen_labels)
        
        # 计算生成器损失
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # 训练判别器
        optimizer_D.zero_grad()
        
        # 真实图像损失
        real_loss = adversarial_loss(discriminator(real_imgs, conditions.argmax(dim=1)), valid)
        # 假图像损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        # 打印训练状态
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{config.Config.epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # 保存生成的图像样本
        batches_done = epoch * len(dataloader) + i
        if batches_done % config.Config.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"{config.Config.output_dir}/{batches_done}.png", nrow=5, normalize=True)
    
    # 保存模型检查点
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"models/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"models/discriminator_{epoch}.pth")
