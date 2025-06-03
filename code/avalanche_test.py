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

# ==== LSTM Model ====
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Removed the embedding layer since our data is numeric
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x is of shape (batch, input_size); add a time dimension => (batch, 1, input_size)
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        logits = self.classifier(h_n[-1])
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

    num_classes = len(train_dataset.classes)
    
    train_datasets = []
    test_datasets = []
    for cls in range(num_classes):
        # For each experience, filter only samples with the current class label
        train_indices = [idx for idx, (_, label) in enumerate(train_dataset) if label == cls]
        test_indices = [idx for idx, (_, label) in enumerate(test_dataset) if label == cls]

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_subset.targets = [cls for _ in train_indices]
        print(len(train_subset.targets), "train samples for class", cls)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_subset.targets = [cls for _ in test_indices]

        train_datasets.append(as_classification_dataset(train_subset))
        test_datasets.append(as_classification_dataset(test_subset))

    benchmark = dataset_benchmark(train_datasets=train_datasets,
                                  test_datasets=test_datasets)
    return benchmark, num_classes

# ==== Main training ====
def main():
    # Hyperparameters
    hidden_size = 32
    num_epochs = 3
    lr = 0.001
    input_size = 784  # number of features from the synthetic dataset

    # Create benchmark 
    benchmark, num_classes = create_benchmark()
    
    
    # Determine number of classes from the training set

    
    # Model: using the numeric features as input with input_size instead of vocab_size/embedding_dim
    model = LSTMClassifier(input_size=input_size,
                           hidden_size=hidden_size,
                           num_classes=num_classes)
    
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

    # strategy = CWRStar(model=model, optimizer=optimizer, criterion=criterion, cwr_layer_name="classifier", train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    strategy = LwF(model=model, optimizer=optimizer, criterion=criterion, alpha=0.5, temperature=0.2, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
        evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # strategy = SynapticIntelligence(model=model, optimizer=optimizer, criterion=criterion, si_lambda=0.5, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    #? cannot use custom model, hardcoded mobileNet
    # strategy = AR1(model=model, lr=lr, criterion=criterion, freeze_below_layer="classifier", ewc_lambda=0.2, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    

    # Training loop
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        res = strategy.train(experience, num_workers=4)
        print('Training completed')
        results.append(strategy.eval(benchmark.test_stream, num_workers=4))
        
    #res format
    #{'Top1_Acc_MB/train_phase/train_stream/Task000': 1.0, 'Loss_MB/train_phase/train_stream/Task000': 0.008055219426751137, 'Top1_Acc_Epoch/train_phase/train_stream/Task000': 1.0, 'Loss_Epoch/train_phase/train_stream/Task000': 0.01061710013117093, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp000': 7.869167728813327, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp001': 7.377173664076213, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp002': 7.209547952164051, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp003': 7.20576016265567, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp004': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp004': 7.805358150583178, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp005': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp005': 8.26806780041066, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp006': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp006': 6.513399675644017, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp007': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp007': 4.884914323977459, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp008': 1.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp008': 0.007657568545120881, 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp009': 0.0, 'Loss_Exp/eval_phase/test_stream/Task000/Exp009': 9.296576205046609, 'Top1_Acc_Stream/eval_phase/test_stream/Task000': 0.0974, 'Loss_Stream/eval_phase/test_stream/Task000': 6.649215859268885, 'StreamForgetting/eval_phase/test_stream': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp000': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp001': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp002': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp003': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp004': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp005': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp006': 1.0, 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp007': 1.0}
    #considering res format, let's print the accuracy for each experience
    for i, res in enumerate(results):
        print(f"Experience {i} results:")
        for key, value in res.items():
            if 'Top1_Acc' in key:
                print(f"{key}: {value:.4f}")
    
    #lets print loss for each experience
    for i, res in enumerate(results):
        print(f"Experience {i} results:")
        for key, value in res.items():
            if 'Loss' in key:
                print(f"{key}: {value:.4f}")
    
    metric_dict = evaluator.get_all_metrics()
    print(metric_dict)
if __name__ == "__main__":
    main()
