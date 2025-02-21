import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {device}")
MODEL_PATH = './compressed_model.pth'

class FourierFeatures(nn.Module):

    def __init__(self, in_features=1, mapping_size=64, scale=10.0):
        super(FourierFeatures, self).__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B  # (N, mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
    def forward(self, input, target):
        if input.size(2) < 224 or input.size(3) < 224:
            input = F.interpolate(input, size=(224,224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224,224), mode='bilinear', align_corners=False)
        input_3 = input.repeat(1,3,1,1)
        target_3 = target.repeat(1,3,1,1)
        input_3 = (input_3 - self.mean) / self.std
        target_3 = (target_3 - self.mean) / self.std
        feat_input = self.vgg(input_3)
        feat_target = self.vgg(target_3)
        return F.mse_loss(feat_input, feat_target)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)

class PsiNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
        super(PsiNetwork, self).__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(in_features=input_dim, mapping_size=fourier_mapping_size, scale=fourier_scale)
            in_dim = 2 * fourier_mapping_size
        else:
            in_dim = input_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.fc2(x)

class PhiNetwork(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=1):
        super(PhiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.res = ResidualBlock(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res(x)
        return self.fc2(x)

class KolmogorovBlock(nn.Module):
    def __init__(self, hidden_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
        super(KolmogorovBlock, self).__init__()
        self.psi_x = PsiNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim, 
                                  use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
        self.psi_y = PsiNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim,
                                  use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
        self.phi = PhiNetwork(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
    def forward(self, x, y):
        psi_x_out = self.psi_x(x)
        psi_y_out = self.psi_y(y)
        z = psi_x_out + psi_y_out
        return self.phi(z)

class KolmogorovImageCompressor(nn.Module):
    def __init__(self, num_blocks=20, hidden_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
        super(KolmogorovImageCompressor, self).__init__()
        self.blocks = nn.ModuleList([
            KolmogorovBlock(hidden_dim, use_fourier, fourier_mapping_size, fourier_scale)
            for _ in range(num_blocks)
        ])
    def forward(self, coords):
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        return sum(block(x, y) for block in self.blocks)

class ImageDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(ImageDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(features*8*4*4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_image(path, size=(64, 64)):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img)  # (1, H, W)

def create_coordinate_grid(H, W):
    xs = torch.linspace(-1, 1, steps=W)
    ys = torch.linspace(-1, 1, steps=H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)
    return coords.view(-1, 2)

def save_compressed_model(model, path="compressed_model.pth"):
    torch.save(model.state_dict(), path)

def load_compressed_model(path="compressed_model.pth", num_blocks=20, hidden_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
    model = KolmogorovImageCompressor(num_blocks=num_blocks, hidden_dim=hidden_dim,
                                      use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def compress_image(image_path,
                   model_save_path="compressed_model.pth",
                   image_size=(64, 64),
                   num_blocks=20,
                   hidden_dim=256,
                   use_fourier=True,
                   fourier_mapping_size=64,
                   fourier_scale=10.0,
                   epochs=15000,
                   lr=1e-3,
                   mse_weight=1.0,
                   l1_weight=0.5,
                   perceptual_weight=0.8,
                   adv_weight=0.005,
                   use_adversarial=True):
    """
    Обучает генератор на изображении с комбинированной функцией потерь.
    Если модель уже существует, можно загрузить её вместо обучения с нуля.
    """
    if os.path.exists(model_save_path):
        print(f"Модель {model_save_path} уже существует. Загрузка существующей модели.")
        return load_compressed_model(model_save_path, num_blocks=num_blocks, hidden_dim=hidden_dim,
                                     use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
    
    print("Загрузка изображения...")
    img_tensor = load_image(image_path, size=image_size)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    coords = create_coordinate_grid(H, W).to(device)
    target = img_tensor.view(-1, 1).to(device)
    
    generator = KolmogorovImageCompressor(num_blocks=num_blocks, hidden_dim=hidden_dim,
                                          use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale).to(device)
    optimizer_G = optim.AdamW(generator.parameters(), lr=lr)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
    
    mse_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = PerceptualLoss(device)
    
    if use_adversarial:
        discriminator = ImageDiscriminator(in_channels=1, features=64).to(device)
        optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr)
        bce_loss_fn = nn.BCELoss()
    
    print("Начало обучения модели...")
    for epoch in range(epochs):
        generator.train()
        optimizer_G.zero_grad()
        
        output = generator(coords)
        fake_img = output.view(1, 1, H, W)
        real_img = target.view(1, 1, H, W)
        
        loss_mse = mse_loss_fn(output, target)
        loss_l1 = l1_loss_fn(output, target)
        loss_perc = perceptual_loss_fn(fake_img, real_img)
        
        loss_adv = 0.0
        if use_adversarial:
            pred_fake = discriminator(fake_img)
            valid = torch.ones_like(pred_fake, device=device)
            loss_adv = bce_loss_fn(pred_fake, valid)
        
        loss_G = mse_weight * loss_mse + l1_weight * loss_l1 + perceptual_weight * loss_perc + adv_weight * loss_adv
        loss_G.backward()
        optimizer_G.step()
        scheduler_G.step()
        
        if use_adversarial:
            optimizer_D.zero_grad()
            pred_real = discriminator(real_img)
            valid = torch.ones_like(pred_real, device=device)
            loss_real = bce_loss_fn(pred_real, valid)
            pred_fake = discriminator(fake_img.detach())
            fake = torch.zeros_like(pred_fake, device=device)
            loss_fake = bce_loss_fn(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
        
        if epoch % 1000 == 0:
            if use_adversarial:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f}, Adv={loss_adv.item():.6f}), Loss_D = {loss_D.item():.6f}")
            else:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f})")
    print("Обучение завершено. Сохранение модели...")
    save_compressed_model(generator, model_save_path)
    print(f"Модель сохранена в {model_save_path}")
    return generator

def decompress_image(model_path,
                     image_size=(64, 64),
                     num_blocks=20,
                     hidden_dim=256,
                     use_fourier=True,
                     fourier_mapping_size=64,
                     fourier_scale=10.0):
    H, W = image_size
    coords = create_coordinate_grid(H, W).to(device)
    generator = load_compressed_model(model_path, num_blocks=num_blocks, hidden_dim=hidden_dim,
                                      use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
    generator.eval()
    with torch.no_grad():
        reconstructed = generator(coords).view(H, W).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.title("Реконструированное изображение")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")
    plt.show()
    return reconstructed

def compress_to_jpeg(image_path, quality=10, output_path="compressed.jpg"):
    img = cv2.imread(image_path)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Изображение сжато в JPEG и сохранено как {output_path}")

def evaluate_compression(original_path, decompressed_path):
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    decompressed = cv2.imread(decompressed_path, cv2.IMREAD_GRAYSCALE)
    psnr_value = cv2.PSNR(orig, decompressed)
    ssim_value = ssim(orig, decompressed)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

def postprocess_image(reconstructed, upscale_factor=2):
    img_uint8 = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
    upscaled = cv2.resize(img_uint8, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    equalized = cv2.equalizeHist(denoised)
    return equalized

def fine_tune_image(image_path,
                    model_path="compressed_model.pth",
                    fine_tune_epochs=5000,
                    lr=1e-4,
                    mse_weight=1.0,
                    l1_weight=0.5,
                    perceptual_weight=0.5,
                    adv_weight=0.005,
                    use_adversarial=True,
                    num_blocks=20,
                    hidden_dim=256,
                    use_fourier=True,
                    fourier_mapping_size=64,
                    fourier_scale=10.0,
                    image_size=(64, 64)):

    print("Загрузка изображения для дообучения...")
    img_tensor = load_image(image_path, size=image_size)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    coords = create_coordinate_grid(H, W).to(device)
    target = img_tensor.view(-1, 1).to(device)

    generator = load_compressed_model(model_path, num_blocks=num_blocks, hidden_dim=hidden_dim,
                                      use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
    generator.train()

    optimizer_G = optim.AdamW(generator.parameters(), lr=lr)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=fine_tune_epochs)
    
    mse_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = PerceptualLoss(device)

    if use_adversarial:
        discriminator = ImageDiscriminator(in_channels=1, features=64).to(device)
        optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr)
        bce_loss_fn = nn.BCELoss()
    
    print("Начало дообучения модели...")
    for epoch in range(fine_tune_epochs):
        generator.train()
        optimizer_G.zero_grad()
        
        output = generator(coords)
        fake_img = output.view(1, 1, H, W)
        real_img = target.view(1, 1, H, W)
        
        loss_mse = mse_loss_fn(output, target)
        loss_l1 = l1_loss_fn(output, target)
        loss_perc = perceptual_loss_fn(fake_img, real_img)
        
        loss_adv = 0.0
        if use_adversarial:
            pred_fake = discriminator(fake_img)
            valid = torch.ones_like(pred_fake, device=device)
            loss_adv = bce_loss_fn(pred_fake, valid)
        
        loss_G = mse_weight * loss_mse + l1_weight * loss_l1 + perceptual_weight * loss_perc + adv_weight * loss_adv
        loss_G.backward()
        optimizer_G.step()
        scheduler_G.step()
        
        if use_adversarial:
            optimizer_D.zero_grad()
            pred_real = discriminator(real_img)
            valid = torch.ones_like(pred_real, device=device)
            loss_real = bce_loss_fn(pred_real, valid)
            pred_fake = discriminator(fake_img.detach())
            fake = torch.zeros_like(pred_fake, device=device)
            loss_fake = bce_loss_fn(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
        
        if epoch % 200 == 0:
            if use_adversarial:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f}, Adv={loss_adv.item():.6f}), Loss_D = {loss_D.item():.6f}")
            else:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f})")
    
    print("Дообучение завершено. Сохранение обновлённой модели...")
    save_compressed_model(generator, model_path)
    print(f"Обновлённая модель сохранена в {model_path}")


if __name__ == "__main__":
    image_path = input('Введите путь к изображению: ')
    model_save_path = "compressed_model.pth"
    jpeg_output_path = "compressed.jpg"
    image_size = (64, 64)
    
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден. Поместите изображение в рабочую папку и переименуйте его в '{image_path}'.")
        exit(1)
    
    compress_image(image_path,
                   model_save_path=model_save_path,
                   image_size=image_size,
                   num_blocks=20,
                   hidden_dim=256,
                   use_fourier=True,
                   fourier_mapping_size=64,
                   fourier_scale=10.0,
                   epochs=15000,
                   lr=1e-3,
                   mse_weight=1.0,
                   l1_weight=0.5,
                   perceptual_weight=0.8,
                   adv_weight=0.005,
                   use_adversarial=True)
    
    fine_tune = input("Выполнить дообучение модели? (y/n): ").strip().lower()
    if fine_tune == 'y':
        fine_tune_image(image_path,
                        model_path=model_save_path,
                        fine_tune_epochs=5000,
                        lr=1e-4,
                        mse_weight=1.0,
                        l1_weight=0.5,
                        perceptual_weight=0.5,
                        adv_weight=0.005,
                        use_adversarial=True,
                        num_blocks=20,
                        hidden_dim=256,
                        use_fourier=True,
                        fourier_mapping_size=64,
                        fourier_scale=10.0,
                        image_size=image_size)
    
    reconstructed = decompress_image(model_save_path,
                     image_size=image_size,
                     num_blocks=20,
                     hidden_dim=256,
                     use_fourier=True,
                     fourier_mapping_size=64,
                     fourier_scale=10.0)
    
    postprocessed = postprocess_image(reconstructed, upscale_factor=2)
    plt.figure(figsize=(6,6))
    plt.title("Постобработанное изображение")
    plt.imshow(postprocessed, cmap="gray")
    plt.axis("off")
    plt.show()
    
    compress_to_jpeg(image_path, quality=10, output_path=jpeg_output_path)
    evaluate_compression(image_path, jpeg_output_path)
