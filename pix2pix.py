import torch
import os
import shutil
from tqdm import tqdm
import torch.nn as nn
from Models.unet import UNet, Logs as G_Logs
from Models.patchgan import Logs as D_Logs
from Models.patchgan import PathGan
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

PRINTLOG = False


class Logs:

    def __init__(self, printlogs=False):
        global PRINTLOG
        PRINTLOG = printlogs

    def __str__(self):
        return f"Printing Pix2Pix logs : {PRINTLOG}"


def check_dir(name):
    if os.path.exists(name) is False:
        os.makedirs(name)


def init_weights(net, init_type='normal', scaling=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)

        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_avg(List):
    return sum(List)/len(List)


class Pix2Pix(nn.Module):

    def __init__(self, in_channels, out_channels, device, learning_rate=0.0002, save_after=5, LOAD_MODEL=None):

        super().__init__()

        self.init_params(in_channels, out_channels,
                         learning_rate, device, LOAD_MODEL)
        self.init_logs(save_after)

    def init_params(self, in_channels, out_channels, learning_rate, device, LOAD_MODEL):

        self.generator = UNet(in_channels).float().to(device)
        self.discriminator = PathGan(
            in_channels + out_channels).float().to(device)
        self.gen_opt = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate)
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate)

        if LOAD_MODEL:
            self.load_model(LOAD_MODEL)
        else:
            init_weights(self.generator)
            init_weights(self.discriminator)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.device = device
        self.step = 0

    def init_logs(self, save_after):

        self.VAL_LOSS = []
        self.GAN_LOSS = []
        self.DIS_LOSS = []
        self.GAN_LOSS_VAL = []
        self.DIS_LOSS_VAL = []

        self.writer = SummaryWriter(
            '/content/logs') if os.path.isdir('/content/logs') else SummaryWriter('logs')

        self.create_checkpoint_after = save_after
        self.max_checkpoints_capacity = 3
        self.max_img_log_capacity = 20
        self.save_path = 'Checkpoint/'
        self.img_dir = 'Img_Dir/'
        check_dir(self.img_dir)

    def train_discriminator(self, img, output, real_target, fake_target):
        if PRINTLOG:
            print(
                f"Pix2Pix ==> Traning Discriminator : Img {img.shape}, Output {output.shape}")

        if PRINTLOG:
            print(f"Pix2Pix ==> Generating Fake Images : Img {img.shape}")
        fake_images = self.generator(img)
        if PRINTLOG:
            print(
                f"Pix2Pix ==> Traning Discriminator on Fake Images : Fake img {fake_images.shape}, Output {output.shape}")
        pred_descriminator = self.discriminator(fake_images, output)
        fake_loss = self.discriminator_loss(pred_descriminator, fake_target)

        if PRINTLOG:
            print(
                f"Pix2Pix ==> Traning Discriminator on Real Images : Real img {img.shape}, Output {output.shape}")
        pred_descriminator = self.discriminator(img, output)
        real_loss = self.discriminator_loss(pred_descriminator, real_target)
        loss = (fake_loss + real_loss)/2
        self.DIS_LOSS.append(float(loss))

        self.writer.add_scalar('Model/D Loss', loss, self.step)
        self.writer.add_scalar('Discriminator/Real Loss', real_loss, self.step)
        self.writer.add_scalar('Discriminator/Fake Loss', fake_loss, self.step)
        return loss

    def train_generator(self, img, output, real_target):
        if PRINTLOG:
            print(
                f"Pix2Pix ==> Traning Generator : Img {img.shape}, Output {output.shape}")
        fake_img = self.generator(img)
        if PRINTLOG:
            print(
                f"Pix2Pix ==> Checking Generated img on Discriminator : Fake Img {fake_img.shape}, Output {output.shape}")
        G = self.discriminator(fake_img, output)
        gan_loss = self.generator_loss(fake_img, output, G, real_target)
        self.GAN_LOSS.append(float(gan_loss))

        self.writer.add_scalar('Model/G Loss', gan_loss, self.step)

        return gan_loss

    def discriminator_loss(self, output, label):
        return self.adversarial_criterion(output, label)

    def generator_loss(self, generated_output, target, discriminator_output, real_target):
        if PRINTLOG:
            print(
                f"Pix2Pix ==> Generator loss : Fake Img {generated_output.shape}, Target {target.shape}, Discrim Output {discriminator_output.shape}, Real Output {real_target.shape}")
        gen_loss = self.adversarial_criterion(
            discriminator_output, real_target)
        if PRINTLOG:
            print(f"Pix2Pix ==> : Generator loss Recreation loss {gen_loss}")
        recreation_loss = self.recon_criterion(generated_output, target)
        if PRINTLOG:
            print(
                f"Pix2Pix ==> : Recreation Loss {recreation_loss}, Adverse loss {gen_loss.shape}")
        self.writer.add_scalar(
            'Generator/Adversarial Loss', gen_loss, self.step)
        self.writer.add_scalar('Generator/Recreation Loss',
                               recreation_loss, self.step)

        return gen_loss + (100 * recreation_loss)

    def validation(self, Input, target, real_target, fake_target):
        generated_img = self.generator(Input)
        validation_by_discriminator = self.discriminator(generated_img, target)
        generator_loss = self.generator_loss(
            generated_img, target, validation_by_discriminator, real_target)
        discriminator_loss = self.discriminator_loss(
            validation_by_discriminator, fake_target)
        return (generator_loss, discriminator_loss)

    def create_checkpoint(self, epochs, Method, loss_gen, loss_dicrim):
        check_dir(self.save_path)
        if len(next(os.walk(self.save_path))[2]) >= 3:
            dirs = sorted(list(next(os.walk(self.save_path))
                          [2]), key=lambda x: int(x[5:-4]))
            os.remove(self.save_path + dirs[0])
        path = self.save_path + Method

        torch.save({
            'generator_model': self.generator.state_dict(),
            'disciminator_model': self.discriminator.state_dict(),
            'generator_loss': loss_gen,
            'disciminator_loss': loss_dicrim,
            'generator_optim': self.gen_opt.state_dict(),
            'disciminator_optim': self.disc_opt.state_dict(),
            'epoch': str(epochs)
        }, path + '.pt')

        print(f"\nPix2Pix {Method} Saved!")

    def load_model(self, LOAD_MODEL):
        if os.path.exists(LOAD_MODEL):
            dirs = sorted(list(next(os.walk(self.save_path))
                          [2]), key=lambda x: int(x[5:-4]))
            checkpoint = torch.load(self.save_path + dirs[-1])
            self.generator.load_state_dict(checkpoint['generator_model'])
            self.discriminator.load_state_dict(
                checkpoint['disciminator_model'])
            self.gen_opt.load_state_dict(checkpoint['generator_optim'])
            self.disc_opt.load_state_dict(checkpoint['disciminator_optim'])
        else:
            raise Exception("Model not found!!!")

    def save_images(self):
        dirs = sorted(list(next(os.walk(self.img_dir))
                      [2]), key=lambda x: int(x[:-4]))
        if len(dirs) >= self.max_img_log_capacity:
            os.remove(self.img_dir + dirs[0])

    def train(self, train_loader, val_loader, epochs, patch_gan_output=16):

        for epoch in (range(epochs)):

            try:
                for data in tqdm(train_loader):
                    MR, CT = data['MR'].to(self.device).float(
                    ), data['CT'].to(self.device).float()

                    fake_target = torch.autograd.Variable(torch.zeros(
                        MR.size(0), 1, patch_gan_output, patch_gan_output).to(self.device))
                    real_target = torch.autograd.Variable(torch.ones(
                        MR.size(0), 1, patch_gan_output, patch_gan_output).to(self.device))
                    if PRINTLOG:
                        print(
                            f"Pix2Pix ==> MR {MR.shape}, CT {CT.shape}, Realtarget {real_target.shape}, Faketarget {fake_target.shape}")

                    self.disc_opt.zero_grad()
                    D_loss = self.train_discriminator(
                        MR, CT, real_target, fake_target)
                    D_loss.backward()
                    self.disc_opt.step()

                    self.gen_opt.zero_grad()
                    gen_loss = self.train_generator(MR, CT, real_target)

                    gen_loss.backward()
                    self.gen_opt.step()
                    self.step += 1

                del fake_target
                del real_target
                del D_loss
                del gen_loss

                device = self.device
                for data in tqdm(val_loader):
                    MR, CT = data['MR'].to(device).float(
                    ), data['CT'].to(device).float()
                    fake_target = torch.autograd.Variable(torch.zeros(
                        MR.size(0), 1, patch_gan_output, patch_gan_output).to(device))
                    real_target = torch.autograd.Variable(torch.ones(
                        MR.size(0), 1, patch_gan_output, patch_gan_output).to(device))
                    g_loss, d_loss = self.validation(
                        MR, CT, real_target, fake_target)
                    g_loss, d_loss = float(g_loss), float(d_loss)
                    self.GAN_LOSS_VAL.append(g_loss)
                    self.DIS_LOSS_VAL.append(d_loss)

                del fake_target
                del real_target
                del g_loss
                del d_loss

                img = self.generator(MR)

                fig, ax = plt.subplots(1, 3, figsize=(30, 10))
                ax[0].imshow(MR[0, 0, :, :].cpu().numpy(), cmap='gray')
                ax[1].imshow(CT[0, 0, :, :].cpu().numpy(), cmap='gray')
                ax[2].imshow(img[0, 0, :, :].cpu(
                ).detach().numpy(), cmap='gray')
                del img
                plt.show()
                self.save_images()
                fig.savefig(self.img_dir + str(epoch) + '.png')

                self.writer.add_scalars('Episode/Generator', {'Gen_Train_Loss': get_avg(self.GAN_LOSS),
                                        'Gen_Valid_Loss': get_avg(self.GAN_LOSS_VAL)}, epoch)
                self.writer.add_scalars('Episode/Discriminator', {'Dis_Train_Loss': get_avg(self.DIS_LOSS),
                                        'DIS_Valid_Loss': get_avg(self.DIS_LOSS_VAL)}, epoch)

                print(
                    f'Epoch {epoch}: Generator Loss {get_avg(self.GAN_LOSS)}, Discriminator Loss {get_avg(self.DIS_LOSS)}')
                if (epoch+1) % self.create_checkpoint_after == 0:
                    METHOD = f"epoch{epoch}_"
                    self.create_checkpoint(epoch, METHOD, get_avg(
                        self.GAN_LOSS), get_avg(self.DIS_LOSS))
            except Exception as e:
                print("Error : ", e)
                METHOD = f"epoch{epoch}_"
                self.create_checkpoint(epoch, METHOD, get_avg(
                    self.GAN_LOSS), get_avg(self.DIS_LOSS))
                raise Exception("Error ! in train loop")
