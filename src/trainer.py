import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def build_image_dataloaders(dataset_name, data_dir, batch_size, shuffle_train=True):
    """Construct train/validation dataloaders for image datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        val_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    elif dataset_name == "fashion-mnist":
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
        val_dataset = datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class Trainer:
    """Trainer class for SpCauchyVAE"""
     
    def __init__(
        self,
        model,
        dataset="mnist",
        data_dir="./data",
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_epochs=100,
        checkpoint_dir="./checkpoints",
        device=None
    ):
        self.model = model
        self.dataset_name = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device is not None else model.config.device
        
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # dataloaders
        self.train_loader, self.val_loader = self._get_dataloaders()
        
        # checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def _get_dataloaders(self):
        """Get train and validation data loaders."""
        return build_image_dataloaders(
            dataset_name=self.dataset_name,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            shuffle_train=True,
        )
    
    def train(self):
        """Train the model"""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, second_param = self.model(data)
                loss, recon_loss, kl_loss = self.model.loss_function(data, recon_batch, mu, second_param)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon_loss': f'{recon_loss.item():.4f}',
                    'kl_loss': f'{kl_loss.item():.6f}'
                })
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_recon_loss = train_recon_loss / len(self.train_loader)
            avg_train_kl_loss = train_kl_loss / len(self.train_loader)
            self.model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
                for batch_idx, (data, _) in enumerate(pbar):
                    data = data.to(self.device)
                    recon_batch, mu, second_param = self.model(data)
                    loss, recon_loss, kl_loss = self.model.loss_function(data, recon_batch, mu, second_param)
                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'recon_loss': f'{recon_loss.item():.4f}',
                        'kl_loss': f'{kl_loss.item():.6f}'
                    })
            avg_val_loss = val_loss / len(self.val_loader)
            avg_val_recon_loss = val_recon_loss / len(self.val_loader)
            avg_val_kl_loss = val_kl_loss / len(self.val_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon_loss:.4f}, KL: {avg_train_kl_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(f"best_model.pt", epoch, avg_val_loss)
                print(f"  New best model saved with validation loss: {avg_val_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, avg_val_loss)
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.model.config
        }, checkpoint_path)
