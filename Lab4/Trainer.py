import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.kl_anneal_type = args.kl_anneal_type
        self.current_epoch = current_epoch
        self.n_cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.n_epoch = args.num_epoch

        self.L =  np.ones(self.n_epoch) * 1.0
        if self.kl_anneal_type == "Cyclical":
            self.frange_cycle_linear(n_epoch=self.n_epoch, n_cycle=self.n_cycle,ratio=self.ratio)
        elif self.kl_anneal_type == "Monotonic":
            self.frange_cycle_linear(n_epoch=self.n_epoch, n_cycle=1,ratio=self.ratio)
        # for i in range(0,self.n_epoch):
        #     print(self.L[i])
        # self.plot_L()
    def update(self):
        # TODO
        self.current_epoch+=1
    
    def get_beta(self):
        # TODO
        return self.L[self.current_epoch]

    def frange_cycle_linear(self, start=0.0, stop=1.0, n_epoch=0, n_cycle=4, ratio=0.4):
        # TODO
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule
        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                self.L[int(i+c*period)] = v
                v += step
                i += 1
        return self.L 

    def plot_L(self):
        epochs = list(range(0, self.n_epoch))

        # 绘制曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.L, label="KL weight")
        plt.xlabel("Epochs")
        plt.ylabel("KL weight")
        plt.title("KL weight curve")
        plt.legend()
        plt.grid()
        plt.show()
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        #Train loss curve and PSNR curve
        self.train_loss = []
        self.train_kld = []
        self.val_PSNR = []
    def forward(self, img, label):
        pass
    
    def training_stage(self):       
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()     
            total_loss = 0
            total_kld = 0
            loss = 0
 
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):   
                adapt_TeacherForcing = True if random.random() < self.tfr else False                          
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse, kld = self.training_one_step(img, label, adapt_TeacherForcing)
                total_loss += mse.item() * img.size(0)
                total_kld += kld.item() * img.size(0)
                #total_kld 
                beta = self.kl_annealing.get_beta()

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.2f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss, lr='{:.0e}'.format(self.scheduler.get_last_lr()[0]))
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.2f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss, lr='{:.0e}'.format(self.scheduler.get_last_lr()[0]))
            
            epoch_loss = total_loss/len(train_loader.dataset)
            epoch_kld = total_kld/len(train_loader.dataset)
            self.train_loss.append(epoch_loss)
            self.train_kld.append(epoch_kld)
            self.eval()
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            print(f"Epoch {self.current_epoch} AVG MSE loss is {epoch_loss} and AVG KLD loss is {epoch_kld}")
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            self.current_epoch += 1
                
     
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss, lr='{:.0e}'.format(self.scheduler.get_last_lr()[0]))
        self.val_PSNR.append(loss)
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        decoded_frame_list = [img[0].cpu()]
        mse = 0
        kld = 0
        self.label_transformation.zero_grad()
        self.frame_transformation.zero_grad()
        self.Gaussian_Predictor.zero_grad()
        self.Decoder_Fusion.zero_grad()
        self.Generator.zero_grad()  
        human_feat_hat_list = []
        for i in range(0,self.train_vi_len):
            human_feat_hat = self.frame_transformation(img[i])
            human_feat_hat_list.append(human_feat_hat)
        for i in range(1, self.train_vi_len):
            #如果採用teacherforcing,encoder input為i-1張原圖,反之,使用預測出的圖
            label_feat = self.label_transformation(label[i])      
            z, mu, logvar = self.Gaussian_Predictor(human_feat_hat_list[i],label_feat)
            if adapt_TeacherForcing:
                human_feat_hat = human_feat_hat_list[i-1]
            else:
                human_feat_hat = self.frame_transformation(decoded_frame_list[i-1].to(self.args.device))

            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)   
            out = self.Generator(parm)
            mse += self.mse_criterion(out, img[i])
            kld += kl_criterion(mu, logvar, self.batch_size)
            decoded_frame_list.append(out.cpu())

        beta = self.kl_annealing.get_beta()
        loss = mse + kld * beta
        loss.backward()
        self.optimizer_step()
        return loss.detach().cpu().numpy() / self.train_vi_len, mse.detach().cpu().numpy() / self.train_vi_len, kld.detach().cpu().numpy() / self.train_vi_len
    
    def val_one_step(self, img, label):
        # TODO
        PSNR_LIST = []
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        assert img.shape[0] == 630, "Testing video seqence should be 630"

        decoded_frame_list = [img[0].cpu()]
        out = img[0]
        for i in range(1,self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)      
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)     
            decoded_frame_list.append(out.cpu())

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        ground_truth = img.permute(1, 0, 2, 3, 4).to("cpu")
        #print(type(ground_truth),type(generated_frame))
        
        for i in range(1, 630):
            PSNR = Generate_PSNR(ground_truth[0][i], generated_frame[0][i])
            PSNR_LIST.append(PSNR.item())
        # frame_indices = range(len(PSNR_LIST))  # 用來表示幀的索引
        # average_psnr = np.mean(PSNR_LIST)
        # plt.figure(figsize=(10, 6))  # 設定圖的大小
        # plt.plot(frame_indices, PSNR_LIST, label=f'Avg PSNR: {average_psnr:.2f}')
        # plt.title('Per Frame Quality(PSNR)')  # 設定圖的標題
        # plt.xlabel('Frame Index')  # x 軸標籤
        # plt.ylabel('PSNR')  # y 軸標籤
        # plt.legend()
        # plt.grid(True)  # 顯示網格
        # plt.show()  # 顯示圖形

        return sum(PSNR_LIST)/(len(PSNR_LIST)-1)

                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch < self.args.tfr_sde:
            return 1.0

        steps_since_tfr_sde = self.current_epoch - self.args.tfr_sde
        if steps_since_tfr_sde >= 0 and steps_since_tfr_sde % 2 == 0:
            self.tfr -= self.args.tfr_d_step

        return self.tfr
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       : self.tfr,
            "last_epoch": self.current_epoch,
            "train_loss_curve": self.train_loss,
            "train_kld_curve": self.train_kld,
            "val_PSNR": self.val_PSNR
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            #self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']
            self.train_loss = checkpoint['train_loss_curve']
            self.train_kld = checkpoint["train_kld_curve"]
            self.val_PSNR = checkpoint['val_PSNR']

            #self.plot_loss(checkpoint['train_loss_curve'],"train_loss")
            #self.plot_loss(checkpoint['val_PSNR'],"PSNR")
    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()
    def plot_loss(self,datapoint,title):
        epochs = list(range(1, len(datapoint) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, datapoint, marker='o')
        plt.title(title+' Curve')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.grid(True)
        plt.show()


def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=4)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            default="/kaggle/input/lab4-dataset/LAB4_Dataset",type=str,   help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, default="/kaggle/working/",  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=8,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    parser.add_argument('--beta1',         type=float, default=0.95,     help="optim momentum beta1")
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=19,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.05,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,   default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Monotonic',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=2,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    

    args = parser.parse_args()
    main(args)
