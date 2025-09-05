import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.blur_gan import BlurGAN
from data.dataset import get_dataloader

# 设置随机种子，保证 reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据集配置
data_root = 'D:/pythonFileData/absolute/path/to/data/train'  # 指向左右分屏数据集
image_size = (256, 256)  # 图像大小
# 修改后
batch_size = 2  # 降低批次大小以减少内存消耗
num_epochs = 30 # 训练轮数
lr = 0.0001  # 学习率

if __name__ == '__main__':
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = 'results/blur_generation'
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    model = BlurGAN().to(device)

    # 优化器
    optimizer_g = optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 学习率调度器
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)

    # 数据加载器 - 使用左右分屏数据集
    train_dataloader = get_dataloader(data_root, 'train', image_size, batch_size, use_split_screen=True)
    # 注意：我们将test集用作验证集
    val_dataloader = get_dataloader(data_root, 'test', image_size, batch_size, shuffle=False, use_split_screen=True)

    # TensorBoard 日志
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_d_loss = 0.0
        train_g_loss = 0.0

        for i, (blur_images, sharp_images) in enumerate(train_dataloader):
            # 注意这里的输入顺序：生成器接收清晰图像，目标是生成模糊图像
            # 所以我们交换了blur_images和sharp_images的位置
            sharp_images, real_blur_images = blur_images, sharp_images
            
            # 移到设备上
            sharp_images = sharp_images.to(device)
            real_blur_images = real_blur_images.to(device)

            # 训练一步
            d_loss, g_loss = model.train_step(
                sharp_images, real_blur_images, optimizer_g, optimizer_d,
                lambda_content=100, lambda_perceptual=20
            )

            # 累加损失
            train_d_loss += d_loss
            train_g_loss += g_loss

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}")

            # 记录到TensorBoard
            global_step = epoch * len(train_dataloader) + i
            writer.add_scalar('train/d_loss', d_loss, global_step)
            writer.add_scalar('train/g_loss', g_loss, global_step)

            # 每200步可视化一次结果
            if global_step % 200 == 0:
                with torch.no_grad():
                    generated_blur = model(sharp_images[:4])
                writer.add_images('train/sharp_images', sharp_images[:4], global_step)
                writer.add_images('train/real_blur_images', real_blur_images[:4], global_step)
                writer.add_images('train/generated_blur_images', generated_blur[:4], global_step)

        # 计算平均训练损失
        train_d_loss /= len(train_dataloader)
        train_g_loss /= len(train_dataloader)

        # 学习率调度
        scheduler_g.step()
        scheduler_d.step()

        # 验证
        model.eval()
        val_d_loss = 0.0
        val_g_loss = 0.0
        with torch.no_grad():
            for blur_images, sharp_images in val_dataloader:
                sharp_images, real_blur_images = blur_images, sharp_images
                sharp_images = sharp_images.to(device)
                real_blur_images = real_blur_images.to(device)
                
                # 生成模糊图像
                generated_blur = model(sharp_images)
                
                # 计算验证损失（简化版）
                val_g_loss += torch.mean(torch.abs(generated_blur - real_blur_images)).item()

            # 计算平均验证损失
            val_g_loss /= len(val_dataloader)
            writer.add_scalar('val/g_loss', val_g_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train d_loss: {train_d_loss:.4f}, Train g_loss: {train_g_loss:.4f}, Val g_loss: {val_g_loss:.4f}")

            # 保存最佳模型
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

        # 每个epoch保存一次模型
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))

    print("训练完成!")
    writer.close()