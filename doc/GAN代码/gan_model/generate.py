import torch
from model import Generator
import config
import argparse
from torchvision.utils import save_image
import os

def generate_images(conditions, num_images, output_dir, model_path="models/generator_final.pth"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化生成器
    generator = Generator().to(config.Config.device)
    generator.load_state_dict(torch.load(model_path, map_location=config.Config.device))
    generator.eval()
    
    # 分批生成图像(每次5张)
    batch_size = 5
    num_batches = (num_images + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch in range(num_batches):
            # 计算当前批次大小
            current_batch_size = min(batch_size, num_images - batch * batch_size)
            
            # 准备噪声
            z = torch.randn(current_batch_size, config.Config.latent_dim).to(config.Config.device)
            
            # 准备条件标签
            labels = torch.tensor(conditions).repeat(current_batch_size, 1).argmax(dim=1).to(config.Config.device)
            
            # 生成图像
            gen_imgs = generator(z, labels)
            
            # 保存图像
            for i in range(current_batch_size):
                img_num = batch * batch_size + i
                save_image(gen_imgs[i], f"{output_dir}/generated_{img_num}.png", normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=int, nargs="+", required=True,
                       help="Condition vector (e.g. 1 0 0 0 0 for sunny)")
    parser.add_argument("--num", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="generated_samples", help="Output directory")
    parser.add_argument("--model", type=str, default="models/generator_final.pth", help="Generator model path")
    
    args = parser.parse_args()
    
    # 验证条件向量长度
    if len(args.conditions) != config.Config.num_conditions:
        raise ValueError(f"Condition vector must have length {config.Config.num_conditions}")
    
    generate_images(args.conditions, args.num, args.output, args.model)
