from model import CustomResNet as TheModel
from train import train as the_trainer
from predict import predict as the_predictor
from dataset import FlowerDataset as TheDataset
from dataset import train_loader as the_dataloader
from config import batch_size as the_batch_size
from config import num_epochs as total_epochs
