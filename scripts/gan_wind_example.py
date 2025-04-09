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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

out_png_path = '/lcrc/group/earthscience/rjackson/Earnest/WGAN_tests/'

class Generator(nn.Module):

    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        
        self.gen = nn.Sequential(
            
            self.get_generator_block(z_dim, 
                                     hidden_dim * 2,
                                     kernel_size=4, 
                                     stride=2),
            self.get_generator_block(hidden_dim * 2,
                                     hidden_dim * 4,
                                     kernel_size=4,
                                     stride=2),            
            self.get_generator_block(hidden_dim * 4,
                                     hidden_dim * 4,
                                     kernel_size=2,
                                     stride=2),                        
            
            self.get_generator_final_block(hidden_dim * 4,
                                           im_chan,
                                           kernel_size=2,
                                           stride=2)
            

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

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            self.get_critic_block(im_chan,
                                         hidden_dim * 4,
                                         kernel_size=4,
                                         stride=2),
            
            self.get_critic_block(hidden_dim * 4,
                                         hidden_dim * 8,
                                         kernel_size=4,
                                         stride=2,),
            
            self.get_critic_final_block(hidden_dim * 8,
                                               1,
                                               kernel_size=4,
                                               stride=2,),

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
    gen_image = gen(x_train[image_no].to(device)).cpu().detach().numpy()
    fig, ax = plt.subplots(3, 1, figsize=(5,10))
    c = ax[0].pcolormesh(x_train[image_no, 0, :, :]*wind_max, vmin=0, vmax=50, cmap='Spectral_r')
    plt.colorbar(c)
    ax[0].set_title('4 km data')
    c = ax[1].pcolormesh(y_train[image_no, 0, :, :]*wind_max, vmin=0, vmax=50, cmap='Spectral_r')
    plt.colorbar(c)
    ax[1].set_title('1 km data')
    c = ax[2].pcolormesh(gen_image[0, 0, :, :]*wind_max, vmin=0, vmax=50, cmap='Spectral_r')
    plt.colorbar(c)
    ax[2].set_title(f'Generator output epoch # {epoch_no}')
    fig.savefig(os.path.join(out_png_path, f'comparison{epoch_no}.png'), bbox_inches='tight')
    
if __name__ == "__main__":
    post_processed = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/KDVN*.nc',
        concat_dim="time", combine="nested")
    post_processed
    spd_fine = np.concatenate([post_processed['spd1'].values, 
            post_processed['spd2'].values,
            post_processed['spd3'].values,
            post_processed['spd4'].values], axis=0)
    spd_fine.shape

    spd_coarse = np.concatenate([post_processed['spd1_c'].values, 
                post_processed['spd2_c'].values,
                post_processed['spd3_c'].values,
                post_processed['spd4_c'].values], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(spd_coarse, spd_fine, random_state=666)
    # Normalize data
    wind_max = spd_fine.max()
    x_test = x_test/wind_max
    y_test = y_test/wind_max
    x_train = x_train/wind_max
    y_train = y_train/wind_max
    
    # One channel 
    x_test = x_test[:, np.newaxis, :]
    y_test = y_test[:, np.newaxis, :, :]
    x_train = x_train[:, np.newaxis, :]
    y_train = y_train[:, np.newaxis, :, :]
    # Convert to Tensor
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    z_dim = 8            # Input space dimension
    batch_size = 128       
    
    fixed_noise = get_noise(batch_size, z_dim, device=device)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataloader_test = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    lr = 0.0002
    beta_1 = 0.5 
    beta_2 = 0.999
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    
            
    gen = Generator(64).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    summary(gen, (8,8))    
    crit  = Critic().to(device) 
    crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)    
    n_epochs = 1000
    cur_step = 0
    total_steps = 0
    start_time = time.time()
    cur_step = 0
    
    generator_losses = []
    critic_losses = []
    
    C_mean_losses = []
    G_mean_losses = []
    
    c_lambda = 10
    crit_repeats = 5
    display_step = 50
    
    for epoch in range(n_epochs):
        cur_step = 0
        start = time.time()
        gen.train()
        for coarse, fine in dataloader_test:
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
    
                epsilon = torch.rand(len(fine), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, fine, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
    
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
            gr_loss = 30*torch.nn.functional.l1_loss(fake_2, fine)
            gen_loss = gen_loss + gr_loss
            gen_loss.backward()
    
            # Update the weights
            gen_opt.step()
    
            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]
            
            cur_step += 1
            total_steps += 1
            
            print_val = f"Epoch: {epoch}/{n_epochs} Steps:{cur_step}/{len(dataloader_test)}\t"
            print_val += f"Epoch_Run_Time: {(time.time()-start):.6f}\t"
            print_val += f"Loss_C : {mean_iteration_critic_loss:.6f}\t"
            print_val += f"Loss_G : {gen_loss:.6f}\t"  
        print(print_val, end='\r',flush = True)
        plot_image(4000, epoch)

    torch.save(gen, 'generator.pickle')
    torch.save(critic, 'critic.pickle')
    
        
    
