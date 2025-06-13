# lstm_single_task_avalanche.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import re
from collections import Counter
import csv
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.supervised import Naive, LwF, EWC, CWRStar, SynapticIntelligence, AR1
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics

from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils import as_classification_dataset
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP

# ==== LSTM Model ====
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=1)  # Optional: for inference only

    def forward(self, x):
        # x shape: (batch, input_size); add time dimension => (batch, 1, input_size)
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout(x)
        x = self.bn(x)
        logits = self.classifier(x)
        # output = self.softmax(logits)
        return logits


def generate_multiclass_dataset(
    n_samples=1000,
    n_features=20,
    n_classes=5,
    test_size=0.2,
    random_state=42
):
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_targets = (y_train_tensor.tolist())
    test_targets = (y_test_tensor.tolist())
    
    train_tensor_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_tensor_dataset.targets = train_targets
    
    # Wrap in TensorDataset
    test_tensor_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_tensor_dataset.targets = test_targets

    return train_tensor_dataset, test_tensor_dataset

# ==== Create benchmark ====
def create_benchmark():
    # Transformation: convert image to tensor and flatten it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Load MNIST datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #trim down the dataset to 10000 samples for training and 2000 for testing
    train_dataset_trim = torch.utils.data.Subset(train_dataset, range(10000))
    train_dataset_trim.targets = torch.utils.data.Subset(train_dataset.targets, range(10000))
    train_dataset_trim.classes = train_dataset.classes
    test_dataset_trim = torch.utils.data.Subset(test_dataset, range(2000))
    test_dataset_trim.targets = torch.utils.data.Subset(test_dataset.targets, range(2000))
    test_dataset_trim.classes = test_dataset.classes
    print("Train dataset size:", len(train_dataset_trim))
    print("Test dataset size:", len(test_dataset_trim))
    print(train_dataset_trim.targets)
    print(test_dataset_trim.targets)
    num_classes = len(train_dataset_trim.classes)
    
    train_datasets = []
    test_datasets = []
    

    for cls in range(num_classes):
        # For each experience, filter only samples with the current class label
        train_indices = [idx for idx, (_, label) in enumerate(train_dataset_trim) if label == cls]
        test_indices = [idx for idx, (_, label) in enumerate(test_dataset_trim) if label == cls]

        train_subset = torch.utils.data.Subset(train_dataset_trim, train_indices)
        train_subset.targets = [cls for _ in train_indices]
        print(len(train_subset.targets), "train samples for class", cls)
        test_subset = torch.utils.data.Subset(test_dataset_trim, test_indices)
        test_subset.targets = [cls for _ in test_indices]

        train_datasets.append(as_classification_dataset(train_subset))
        test_datasets.append(as_classification_dataset(test_subset))
        
    merged_train_datasets = []
    merged_test_datasets = []
    classes_per_experience = 1 # Number of classes per experience
    for i in range(0, len(train_datasets), classes_per_experience):
        merged_train = ConcatDataset(train_datasets[i:i+classes_per_experience])
        merged_test = ConcatDataset(test_datasets[i:i+classes_per_experience])
        merged_train_datasets.append(merged_train)
        merged_test_datasets.append(merged_test)

    benchmark = dataset_benchmark(train_datasets=merged_train_datasets, test_datasets=merged_test_datasets)

    # benchmark = dataset_benchmark(train_datasets=train_dataset,
    #                               test_datasets=test_dataset)
    return benchmark, num_classes

def flatten_transform(x):
    # If x is already a tensor, just flatten it
    if isinstance(x, torch.Tensor):
        return x.view(-1)
    # Otherwise convert to tensor and flatten it
    x = transforms.functional.to_tensor(x)
    return x.view(-1)

# ==== Main training ====
def main():
    # Hyperparameters
    hidden_size = 32
    num_epochs = 10
    lr = 0.00001
    input_size = 784  # number of features  28*28

    # Create benchmark 
    # Define a transformation to flatten the MNIST images
    transform = transforms.Lambda(flatten_transform)

    # Create benchmark with transformation for both training and evaluation
    benchmark = SplitMNIST(n_experiences=5, return_task_id=False, seed=42,
                           train_transform=transform,
                           eval_transform=transform)
    # benchmark = SplitMNIST(n_experiences=10, return_task_id=False, seed=42)
    
    # benchmark, num_classes = create_benchmark()
    
    num_classes = benchmark.n_classes
    
    print("Number of classes in the benchmark:", num_classes)
    
    #? Instead of incremental experiences, concatenate all training and test experiences:
    from torch.utils.data import ConcatDataset
    full_train_dataset = ConcatDataset([exp.dataset for exp in benchmark.train_stream])
    full_test_dataset  = ConcatDataset([exp.dataset for exp in benchmark.test_stream])
    print("Full train dataset size:", len(full_train_dataset))
    print("Full test dataset size:", len(full_test_dataset))
    
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(full_test_dataset, batch_size=32, shuffle=False)
    #?
    # Model: using the numeric features as input with input_size instead of vocab_size/embedding_dim
    model = LSTMClassifier(input_size=input_size,
                           hidden_size=hidden_size,
                           num_classes=num_classes)
    # model = SimpleMLP(num_classes=num_classes,hidden_size=64,hidden_layers=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    
    # Optimizer, Loss, logger, evaluator, and strategy remain unchanged
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger = TextLogger(open('avalanche_log.txt', 'w'))
    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[logger]
    )

    
    
    # strategy = CWRStar(model=model, optimizer=optimizer, criterion=criterion, cwr_layer_name="classifier", train_mb_size=100, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    strategy = LwF(model=model, optimizer=optimizer, criterion=criterion, alpha=0.2, temperature=0.2, train_mb_size=100, train_epochs=num_epochs, eval_mb_size=50,
        evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # strategy = SynapticIntelligence(model=model, optimizer=optimizer, criterion=criterion, si_lambda=0.5, train_mb_size=64, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # strategy = EWC(model=model, optimizer=optimizer, criterion=criterion, ewc_lambda=0.5, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
    # evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")

    #? cannot use custom model, hardcoded mobileNet
    # strategy = AR1(model=model, lr=lr, criterion=criterion, freeze_below_layer="classifier", ewc_lambda=0.2, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    #?-------------------------
    # print('Starting classical training...')
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for batch_idx, (data, target, _) in enumerate(train_loader):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    # # Evaluation loop
    # model.eval()
    # test_loss = 0.0
    # correct = 0
    # with torch.no_grad():
    #     for data, target, _ in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += criterion(output, target).item() * data.size(0)
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    # test_loss /= len(full_test_dataset)
    # accuracy = correct / len(full_test_dataset)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    #?-------------------------
    # Training loop
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        res = strategy.train(experience)
        print('Training completed')
        results.append(strategy.eval(benchmark.test_stream, num_workers=4))
        
    #res format
    #{'Top1_Acc_MB/train_phase/train_stream/Task000': 1.0, 'Loss_MB/train_phase/train_stream/Task000': 0.008055219426751137, 'Top1_Acc_Epoch/train_phase/train_stream/Task000': 1.0, 'Loss_Epoch/train_phase/train_stream/Task000': 0.01061710013117093, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp000': 7.869167728813327, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp001': 7.377173664076213, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp002': 7.209547952164051, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp003': 7.20576016265567, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp004': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp004': 7.805358150583178, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp005': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp005': 8.26806780041066, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp006': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp006': 6.513399675644017, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp007': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp007': 4.884914323977459, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp008': 1.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp008': 0.007657568545120881, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp009': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp009': 9.296576205046609, 'Top1_Acc_Stream/eval_phase/test_stream/Task000': 0.0974, 'Loss_Stream/eval_phase/test_stream/Task000': 6.649215859268885, 'StreamForgetting/eval_phase/test_stream': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp000': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp001': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp002': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp003': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp004': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp005': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp006': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp007': 1.0}
    #considering res format, let's print the accuracy for each experience
    for i, res in enumerate(results):
        print(f"Experience {i} results:")
        for key, value in res.items():
            if 'Acc' in key:
                print(f"{key}: {value:.4f}")
    
    #lets print loss for each experience
    for i, res in enumerate(results):
        print(f"Experience {i} results:")
        for key, value in res.items():
            if 'Loss' in key:
                print(f"{key}: {value:.4f}")
if __name__ == "__main__":
    main()
