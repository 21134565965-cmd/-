import torch
import torch.nn as nn
import torch.autograd as autograd
from models.blur_generator import BlurGenerator
from models.discriminator import Discriminator

class BlurGAN(nn.Module):
    """用于生成遥感图像模糊效果的GAN模型"""
    def __init__(self):
        super(BlurGAN, self).__init__()
        # 生成器 - 将清晰图像转换为模糊图像
        self.generator = BlurGenerator()
        # 判别器 - 区分真实模糊图像和生成的模糊图像
        self.discriminator = Discriminator()
        # 损失函数
        self.content_loss = nn.L1Loss()
        # 感知损失 - 使用预训练的VGG16
        self.perceptual_loss = nn.MSELoss()
        # 加载预训练的VGG16
        self.vgg = self._load_vgg()
        
    def _load_vgg(self):
        """加载预训练的VGG16模型用于感知损失"""
        vgg = nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features.children())[:17])
        # 冻结VGG参数
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def forward(self, x):
        # 生成器生成模糊图像
        generated_blur = self.generator(x)
        return generated_blur
    
    def compute_gradient_penalty(self, real_images, fake_images):
        """计算WGAN-GP的梯度惩罚"""
        # 确保在正确的设备上
        device = real_images.device
        
        # 创建随机系数
        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
        
        # 创建插值样本
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates = interpolates.detach().requires_grad_(True)
        
        # 通过判别器传递
        d_interpolates = self.discriminator(interpolates)
        
        # 创建匹配形状的张量作为梯度输出
        ones = torch.ones_like(d_interpolates, device=device)
        
        # 计算梯度
        try:
            gradients = autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
        except RuntimeError:
            # 如果梯度计算失败，返回一个小的惩罚值
            return torch.tensor(1.0, device=device)
        
        # 检查梯度是否存在
        if gradients is None or gradients.numel() == 0:
            return torch.tensor(1.0, device=device)
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    # 修改后
    def train_step(self, sharp_images, real_blur_images, optimizer_g, optimizer_d, lambda_content=100, lambda_perceptual=20, lambda_gp=10):
        # 确保所有张量都在正确的设备上
        device = next(self.parameters()).device
        sharp_images = sharp_images.to(device)
        real_blur_images = real_blur_images.to(device)
        
        # 确保VGG模型在正确的设备上
        self.vgg = self.vgg.to(device)
        
        # ----------------- 训练判别器 ----------------- #
        optimizer_d.zero_grad()
        
        # 生成假模糊图像
        with torch.set_grad_enabled(True):
            fake_blur_images = self.generator(sharp_images)
            
            # 计算真实模糊图像和生成模糊图像的判别器输出
            real_output = self.discriminator(real_blur_images)
            fake_output = self.discriminator(fake_blur_images)
            
            # 计算WGAN损失
            d_loss_real = -torch.mean(real_output)
            d_loss_fake = torch.mean(fake_output)
            d_loss = d_loss_real + d_loss_fake
            
            # 计算梯度惩罚
            try:
                real_images_gp = real_blur_images.clone().detach().requires_grad_(True)
                fake_images_gp = fake_blur_images.clone().detach().requires_grad_(True)
                gradient_penalty = self.compute_gradient_penalty(real_images_gp, fake_images_gp)
                d_loss += lambda_gp * gradient_penalty
            except RuntimeError as e:
                print(f"Warning: Gradient penalty computation failed: {e}")
            
            # 反向传播
            d_loss.backward()
        optimizer_d.step()
        
        # ----------------- 训练生成器 -----------------
        optimizer_g.zero_grad()
        
        # 重新计算生成模糊图像和判别器输出，建立新的计算图
        with torch.set_grad_enabled(True):
            fake_blur_images = self.generator(sharp_images)
            fake_output = self.discriminator(fake_blur_images)
            
            # 对抗损失
            g_loss_adversarial = -torch.mean(fake_output)
            
            # 内容损失 - 确保生成的模糊图像与真实模糊图像相似
            g_loss_content = self.content_loss(fake_blur_images, real_blur_images) * lambda_content
            
            # 感知损失 - 使用较小的批量处理VGG特征
            # 将数据分成更小的批次进行VGG处理
            batch_size = real_blur_images.size(0)
            chunk_size = max(1, batch_size // 2)  # 分成最多2个子批次
            
            vgg_real_chunks = []
            vgg_fake_chunks = []
            
            # 分块处理以减少内存使用
            for i in range(0, batch_size, chunk_size):
                real_chunk = real_blur_images[i:i+chunk_size]
                fake_chunk = fake_blur_images[i:i+chunk_size]
                
                with torch.no_grad():
                    vgg_real_chunk = self.vgg(real_chunk)
                    vgg_fake_chunk = self.vgg(fake_chunk)
                
                vgg_real_chunks.append(vgg_real_chunk)
                vgg_fake_chunks.append(vgg_fake_chunk)
                
                # 清除中间变量
                del real_chunk, fake_chunk, vgg_real_chunk, vgg_fake_chunk
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 重新组合结果
            vgg_real = torch.cat(vgg_real_chunks)
            vgg_fake = torch.cat(vgg_fake_chunks)
            
            # 计算感知损失
            g_loss_perceptual = self.perceptual_loss(vgg_fake, vgg_real) * lambda_perceptual
            
            # 总生成器损失
            g_loss = g_loss_adversarial + g_loss_content + g_loss_perceptual
            
            # 反向传播
            g_loss.backward()
        optimizer_g.step()
        
        # 清理不需要的张量
        del fake_blur_images, fake_output, vgg_real, vgg_fake
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return d_loss.item(), g_loss.item()

# 测试GAN模型
if __name__ == '__main__':
    model = BlurGAN()
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = model(input_tensor)
    print(f"BlurGAN输入形状: {input_tensor.shape}")
    print(f"BlurGAN输出形状: {output_tensor.shape}")