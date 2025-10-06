import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64, kernel_size=2, stride=2):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        
        self.gen = nn.Sequential(
            
            self.get_generator_block(z_dim, 
                                     hidden_dim * 4,
                                     kernel_size=kernel_size, 
                                     stride=stride),
            
            self.get_generator_block(hidden_dim * 4, 
                                     hidden_dim * 2,
                                     kernel_size=kernel_size,
                                     stride=stride),
            self.get_generator_block(hidden_dim * 2, 
                                     hidden_dim * 2,
                                     kernel_size=kernel_size,
                                     stride=stride),
            self.get_generator_block(hidden_dim * 2,
                                     hidden_dim ,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                    ),
            
            self.get_generator_final_block(hidden_dim,
                                           im_chan,
                                           kernel_size=2,
                                           stride=stride)
            

        )
        
        
    def get_generator_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
                nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True),
        )
    
    
    def get_generator_final_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return  nn.Sequential(
                nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.Tanh()
            )
    
    
    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)    
    

class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16, kernel_size=4, stride=2):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            self.get_critic_block(im_chan,
                                         hidden_dim * 4,
                                         kernel_size=kernel_size,
                                         stride=stride),
            
            self.get_critic_block(hidden_dim * 4,
                                         hidden_dim * 8,
                                         kernel_size=kernel_size,
                                         stride=stride,),
            
            self.get_critic_final_block(hidden_dim * 8,
                                               1,
                                               kernel_size=kernel_size,
                                               stride=stride,),

        )

        
    def get_critic_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.LeakyReLU(0.2, inplace=True)
        )
    
    
    def get_critic_final_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return  nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            )
    
    def forward(self, image):
        return self.disc(image)

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples,z_dim, z_dim, device=device)

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty
    
def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss

def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
        
    )[0]
    return gradient

def plot_image(image_no, epoch_no):
    gen_image = gen(x_train[image_no:image_no+1, :, :, :].to(device)).cpu().detach().numpy()
    fig, ax = plt.subplots(3, 1, figsize=(5,10))
    coarse = x_train[image_no, 0, :, :]
    coarse = np.where(coarse > 0, coarse, np.nan)*wind_max
    c = ax[0].pcolormesh(coarse,
                         vmin=0, vmax=30, cmap='Spectral_r')
    plt.colorbar(c)
    ax[0].set_title('4 km data')
    fine = y_train[image_no, 0, :, :]
    fine = np.where(fine > 0, fine, np.nan)*wind_max
    c = ax[1].pcolormesh(fine,
                         vmin=0, vmax=30, cmap='Spectral_r')
    plt.colorbar(c)
    ax[1].set_title('1 km data')
    gen_image = np.where(gen_image[0, 0, :, :] > 0, gen_image[0, 0, :, :], np.nan) * wind_max
    c = ax[2].pcolormesh(gen_image,
                         vmin=0, vmax=30, cmap='Spectral_r')
    ax[2].set_title(f'Generator output epoch # {epoch_no}')
    fig.savefig(os.path.join(out_png_path, f'comparison{epoch_no}.png'), bbox_inches='tight')
    plt.close(fig)
    del fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Script Description")

    # Positional argument
    parser.add_argument("model_name", type=str, help="Name of experiment")

    # Optional arguments
    parser.add_argument("-lr", "--lr", type=float, default=1e-5,
                        help="Starting learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=500,
                        help="Number of training epochs (default: 500)")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("-kg", "--kernel_size_generator", type=int, default=2,
                        help="Kernel size for generator (default: 2)")
    parser.add_argument("-sg", "--stride_size_generator", type=int, default=2,
                        help="Stride size (default: 2)")
    parser.add_argument("-kc", "--kernel_size_critic", type=int, default=2,
                        help="Kernel size for critic (default: 4)")
    parser.add_argument("-sc", "--stride_size_critic", type=int, default=4,
                        help="Stride size (default: 4)")
    parser.add_argument("-z", "--zdim", type=int, default=128,
            help="Latent space dimension")
    parser.add_argument("-hg", "--hidden_size_generator", type=int, default=64,
                        help="Hidden layer size (default: 64)")
    parser.add_argument("-hc", "--hidden_size_critic", type=int, default=16,
                        help="Hidden layer size (default: 16)")
    parser.add_argument("-c", "--critic_rounds", type=int, default=5,
            help="Critic rounds")
    parser.add_argument("-b1", "--beta_1", type=float, default=0.5,
            help="Adam optimizer parameter")
    parser.add_argument("-b2", "--beta_2", type=float, default=0.999,
            help="Adam optimizer parameter")
    parser.add_argument("-et", "--epoch_threshold", type=int, default=150,
            help="Divide learning rate by 10 after this threshold if decay rate is 0 (default: 150)")
    parser.add_argument("-d", "--decay_rate", type=float, default=0.01,
            help="Decay rate for learning rate.")
    args = parser.parse_args()
    model_name = args.model_name
    out_png_path = f'/lcrc/group/earthscience/rjackson/Earnest/WGAN_tests/{model_name}/'
    out_model_path = f'/lcrc/group/earthscience/rjackson/Earnest/WGANs/{model_name}/'
    if not os.path.exists(out_png_path):
        os.makedirs(out_png_path)
    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)

    post_processed = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/KDVN*.nc',
        concat_dim="time", combine="nested")
    post_processed_KDMX = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/KDMX*.nc',
        concat_dim="time", combine="nested")
    post_processed_KARX = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/KARX*.nc',
        concat_dim="time", combine="nested")
    post_processed_KOAX = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/KOAX*.nc',
        concat_dim="time", combine="nested")
    post_processed_KFSD = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/KFSD*.nc',
        concat_dim="time", combine="nested")

    spd_fine = np.concatenate([post_processed['spd1'].values, 
            post_processed['spd2'].values,
            post_processed['spd3'].values,
            post_processed['spd4'].values,
            post_processed_KDMX['spd1'].values, 
            post_processed_KDMX['spd2'].values,
            post_processed_KDMX['spd3'].values,
            post_processed_KDMX['spd4'].values,
            post_processed_KOAX['spd1'].values,
            post_processed_KOAX['spd2'].values,
            post_processed_KOAX['spd3'].values,
            post_processed_KOAX['spd4'].values,
            post_processed_KFSD['spd1'].values,
            post_processed_KFSD['spd2'].values,
            post_processed_KFSD['spd3'].values,
            post_processed_KFSD['spd4'].values,
            post_processed_KARX['spd1'].values,
            post_processed_KARX['spd2'].values,
            post_processed_KARX['spd3'].values,
            post_processed_KARX['spd4'].values], axis=0)
    spd_fine.shape
    spd_fine_mask = np.where(~np.isfinite(spd_fine), 0, 1) 
    spd_fine_gen = np.where(~np.isfinite(spd_fine), 0, spd_fine)
    #spd_fine_gen = np.stack([spd_fine_gen, spd_fine_mask], axis=1)
    spd_coarse = np.concatenate([post_processed['spd1_c'].values, 
                post_processed['spd2_c'].values,
                post_processed['spd3_c'].values,
                post_processed['spd4_c'].values,
                post_processed_KDMX['spd1_c'].values, 
                post_processed_KDMX['spd2_c'].values,
                post_processed_KDMX['spd3_c'].values,
                post_processed_KDMX['spd4_c'].values,
                post_processed_KOAX['spd1_c'].values,
                post_processed_KOAX['spd2_c'].values,
                post_processed_KOAX['spd3_c'].values,
                post_processed_KOAX['spd4_c'].values,
                post_processed_KFSD['spd1_c'].values,
                post_processed_KFSD['spd2_c'].values,
                post_processed_KFSD['spd3_c'].values,
                post_processed_KFSD['spd4_c'].values,
                post_processed_KARX['spd1_c'].values,
                post_processed_KARX['spd2_c'].values,
                post_processed_KARX['spd3_c'].values,
                post_processed_KARX['spd4_c'].values], axis=0)
    spd_coarse_mask = np.where(~np.isfinite(spd_coarse), 0, 1)
    spd_coarse_gen = np.where(~np.isfinite(spd_coarse), 0, spd_coarse)
    spd_coarse_gen = np.stack([spd_coarse_gen, spd_coarse_mask], axis=1)
    #spd_coarse = spd_coarse.reshape((spd_coarse.shape[0], spd_coarse.shape[1]*spd_coarse.shape[2]))
    print(spd_coarse_gen.shape)

    x_train, x_test, y_train, y_test = train_test_split(
            spd_coarse_gen, spd_fine_gen, random_state=666)
    # Normalize data
    wind_max = spd_fine_gen.max()
    x_test = x_test/wind_max
    y_test = y_test/wind_max
    x_train = x_train/wind_max
    y_train = y_train/wind_max
    
    # One channel 
    y_test = y_test[:, np.newaxis, :, :]
    y_train = y_train[:, np.newaxis, :, :]
    # Convert to Tensor
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    print(x_test.shape)

    z_dim = args.zdim            # Input space dimension
    batch_size = args.batch_size
    
    fixed_noise = get_noise(batch_size, z_dim, device=device)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataloader_train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)
    
    lr = args.lr
    beta_1 = args.beta_1 
    beta_2 = args.beta_2
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    
            
    gen = Generator(
            z_dim, im_chan=1,
            hidden_dim=args.hidden_size_generator,
            stride=args.stride_size_generator,
            kernel_size=args.kernel_size_generator,).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    crit = Critic(
            hidden_dim=args.hidden_size_critic,
            stride=args.stride_size_critic,
            kernel_size=args.kernel_size_critic).to(device) 
    crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)    
    n_epochs = args.epochs
    cur_step = 0
    total_steps = 0
    start_time = time.time()
    cur_step = 0
    
    generator_losses = []
    critic_losses = []
    gen_losses_val = []
    crit_losses_val = []
    mean_psnr_test = []
    mean_ssim_test = []
    C_mean_losses = []
    G_mean_losses = []
    
    c_lambda = 10
    crit_repeats = args.critic_rounds
    display_step = 50
    
    for epoch in range(n_epochs):
        cur_step = 0
        start = time.time()
        gen.train()
        if epoch == args.epoch_threshold and args.decay_rate == 0:
            for param_group in gen_opt.param_groups:
                param_group['lr'] = param_group['lr']/10
            for param_group in crit_opt.param_groups:
                param_group['lr'] = param_group['lr']/10
        if args.decay_rate > 0:
            for param_group in gen_opt.param_groups:
                param_group['lr'] = param_group['lr']/(1 + epoch*args.decay_rate)
            for param_group in crit_opt.param_groups:
                param_group['lr'] = param_group['lr']/(1 + epoch*args.decay_rate)

        for coarse, fine in dataloader_train:
            cur_batch_size = len(coarse)
            coarse = coarse.to(device)
            fine = fine.to(device)
    
            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                
                ### Put in 4 channels of noise, one channel of real values
                fake = gen(coarse)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = fine
    
                epsilon = torch.rand(
                        len(fine), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, fine, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(
                        crit_fake_pred, crit_real_pred, gp, c_lambda)
    
                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward()
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]
    
            ### Update generator ###
            gen_opt.zero_grad()
            fake_2 = gen(coarse)
            crit_fake_pred = crit(fake_2)
            
            gen_loss = get_gen_loss(crit_fake_pred)
            gr_loss = 100*torch.nn.functional.l1_loss(fake_2, fine)
            gen_loss = gen_loss + gr_loss
            gen_loss.backward()
    
            # Update the weights
            gen_opt.step()
    
            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]
            
            cur_step += 1
            total_steps += 1
            
            print_val = f"Epoch: {epoch}/{n_epochs} Steps:{cur_step}/{len(dataloader_train)}\t"
            print_val += f"Epoch_Run_Time: {(time.time()-start):.6f}\t"
            print_val += f"Loss_C : {mean_iteration_critic_loss:.6f}\t"
            print_val += f"Loss_G : {gen_loss:.6f}\t"
        with torch.no_grad():
            all_psnr, all_ssim = [], []
            for coarse, fine in dataloader_test:
                coarse, fine = coarse.to(device), fine.to(device)
                fake_high_res = gen(coarse)
                all_psnr.append(psnr(fake_high_res, fine).item())
                all_ssim.append(ssim(fake_high_res, fine).item())
        mean_psnr_test.append(sum(all_psnr) / len(all_psnr))
        mean_ssim_test.append(sum(all_ssim) / len(all_ssim))
        np.savez(os.path.join(out_model_path, "metrics.npz"), mean_psnr_test=np.array(mean_psnr_test), 
                mean_ssim_test=np.array(mean_ssim_test),
                generator_losses=np.array(generator_losses),
                critic_losses=np.array(critic_losses))
        print(print_val, end='\r',flush = True)
        plot_image(1580, epoch)
        if epoch % 10 == 0: 
            torch.save(gen, os.path.join(out_model_path, f'generator{epoch}.pth'))
            torch.save(crit, os.path.join(out_model_path, f'critic{epoch}.pth'))
    
        
    
