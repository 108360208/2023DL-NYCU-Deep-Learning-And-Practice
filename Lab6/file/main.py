from dataloader import iclevrDataSet
from evaluator import evaluation_model
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from diffusers import DDIMScheduler, DDPMPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPFeatureExtractor, CLIPTextModel
from diffusers import UNet2DConditionModel, UNet2DModel, DDPMScheduler
import argparse

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=3):
        super().__init__()
        
        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=64,           # the target image resolution
            in_channels=3, # Additional input channels for class cond.
            out_channels=3,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels = (128, 128, 256, 256, 512, 512), 
            down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, cond):
        # class conditioning in right shape to add as additional input channels
        cond = cond.squeeze() # Map to embedding dinemsion

        return self.model(sample = x, timestep = t, class_labels  = cond).sample # (bs, 1, 28, 28)
def tqdm_bar(pbar, loss, lr,epoch):
    pbar.set_description(f"(Epoch {epoch}, lr:{lr}", refresh=False)
    pbar.set_postfix(loss=float(loss), refresh=False)
    pbar.refresh()
def sample(net,noise_scheduler,dataloader):
    # Sampling loop
    for  img, cond in tqdm(dataloader, ncols=120):
        img = img.to(args.device)
        cond = cond.to(args.device)
        for t in(noise_scheduler.timesteps):
            # Get model pred
            with torch.no_grad():
                residual = net(img, t.to(args.device), cond)  # Again, note that we pass in our labels y
            # Update sample with step
            img = noise_scheduler.step(residual, t, img).prev_sample
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--mode',    type=int,    default=4)
    args = parser.parse_args()
    #noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    noise_scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    noise_scheduler.set_timesteps(num_inference_steps=40)
    dataset = iclevrDataSet("iclevr","train")
    train_loader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=4,
                            shuffle=True)  
    dataset = iclevrDataSet("iclevr","test")
    test_loader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=4,
                            shuffle=False)  
    # How many runs through the data should we do?
    n_epochs = 10
    lr = 0.001
    # Our network 
    net = ClassConditionedUnet().to(args.device)
    eval_model = evaluation_model()
    # Our loss finction
    loss_fn = nn.MSELoss()
    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr ) 
    # Keeping a record of the losses for later viewing

    # The training loop
    for epoch in range(n_epochs):
        total_loss = 0
        for  img, cond in (pbar := tqdm(train_loader, ncols=120)):
            opt.zero_grad()
            # Get some data and prepare the corrupted version
            img = img.to(args.device) 
            cond = cond.to(args.device)
            noise = torch.randn_like(img)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (img.shape[0],),
                device=args.device,
            ).long()
            noisy_x = noise_scheduler.add_noise(img, noise, timesteps)
            # Get the model prediction
            pred = net(noisy_x, timesteps, cond) # Note that we pass in the labels y
            # Calculate the loss
            loss = loss_fn(pred, noise) # How close is the output to the noise
            # Backprop and update the params:
            loss.backward()
            opt.step()
            # Store the loss for later
            total_loss += loss.item() * img.size(0)
            tqdm_bar(pbar, loss, '{:.0e}'.format(lr),epoch)

        epoch_loss = total_loss/len(train_loader.dataset)
        decode_result = sample(net,noise_scheduler,test_loader)
        data_iterator = iter(test_loader)
        batch = next(data_iterator)
        _, labels = batch
        acc = eval_model.eval(decode_result,labels)
        print(f'Finished epoch {epoch}. Average of loss values: {epoch_loss:05f}')
        print(f'Test acc is {acc}')
        grid_of_images = make_grid(decode_result, nrow=8) 
        save_image(grid_of_images, str(epoch)+'image.png')
        