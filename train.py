import argparse
import wandb

import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torch.nn import functional as F 

from src.dataset import get_mnist
from src.model import NeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cpu')
parser.add_argument("--batch", default='64')
parser.add_argument("--epoch", default="20")
parser.add_argument("--lr", default="1e-3")
parser.add_argument('--path', default='./best.pth')
args = parser.parse_args()


def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
    
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    
    _id = 0
    for i,l,p,s in zip(log_images,log_labels,log_preds,log_scores):
        img_id = str(_id) + '_' + str(log_counter)
        test_table.add_data(img_id, wandb.Image(i), p,l, *s)    
        _id += 1
        

def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer, epoch:int) -> None:
    
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (images, targets) in enumerate(dataloader):
        
        images = images.to(args.device) 
        targets = targets.to(args.device)   
        targets = torch.flatten(targets)
        
        preds = model(images)
        loss = loss_fn(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 == 0:
            wandb.log({"train_loss": loss, "epoch": epoch})
            loss = loss.item()
            current = batch*len(images)
            print(f'loss: {loss:>7f} [{current:5d}/{size:>5d}]')
            
            
def vaild_one_epoch(dataloader: DataLoader, model:nn.Module, device:str, loss_fn: nn.Module, epoch:int, test_table: wandb.Table) -> None:
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    
    
    
    with torch.inference_mode():
        for batch, (images,targets) in enumerate(dataloader):
            
            images = images.to(args.device)
            targets = targets.to(args.device)
            targets = torch.flatten(targets)
            
            preds = model(images)
            
            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()
            
            if batch == 0:
                log_test_predictions(images, targets, preds, preds.argmax(1), test_table, args.epoch)
    
    test_loss /= num_batches
    correct /= size
    

    
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Best Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    wandb.log({"test_loss": test_loss, "test_accuracy": correct, "epoch": epoch})
    
    return correct        

def train(device: str):
    num_classes = 10
    batch_size = int(args.batch)
    epochs = int(args.epoch)
    lr = float(args.lr)
    best_correct = 0
    
    data_dir = 'data'
    train_data, test_data = get_mnist(data_dir)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)   
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)    
    
    model = NeuralNetwork(num_classes=num_classes).to(args.device)  
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    test_table = wandb.Table(columns=["id", "iamge", "predicted", "true", *[f"class_{i}_score" for i in range(10)]])
    
    best_correct = 0 
    
    for t in range(epochs):
        print(f'Epoch {t+1}\n------------------')
        train_one_epoch(train_loader, args.device, model, loss_fn, optimizer, t+1)
        correct = vaild_one_epoch(test_loader, model, args.device, loss_fn, t+1, test_table)
        if correct > best_correct:
            torch.save(model.state_dict(), args.path)
    print("Done!")    

    
    
    
    wandb.log({"predictions": test_table})
    torch.save(model.state_dict(),'mnist-1.pth')
    
    print('Saved Pytorch Model State to mnist.pth')
    
if __name__ == "__main__":
    wandb.init(
        
        project="mnist",
        
        config = {
            "learning_rate" : args.lr,
            "architecture" : NeuralNetwork,
            "dataset" : "MNIST",
            "epochs" : 40,
        }
    )
    
    train(device = args.device)
    wandb.finish()
    
"""
train
1. argparser
2. pth 경로, epoch, lr, batch, device
3. Accuracy 젤 높은거 저장

test
1. pth 경로
"""