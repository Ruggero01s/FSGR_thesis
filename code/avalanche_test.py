# lstm_single_task_avalanche.py
#todo find true dataset to test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.logging import InteractiveLogger
from avalanche.training.supervised import Naive, LwF, EWC, CWRStar, SynapticIntelligence
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils import as_classification_dataset


# ==== Synthetic sequential dataset ====
class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_sequences=100, seq_length=10, input_size=8, num_classes=2):
        self.data = []
        self.labels = []
        for _ in range(num_sequences):
            x = np.random.randn(seq_length, input_size)
            y = int(np.mean(x) > 0)
            self.data.append(torch.tensor(x, dtype=torch.float32))
            self.labels.append(y)
        self.targets = self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==== LSTM Model ====
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        logits = self.classifier(h_n[-1])
        return logits

# ==== Create synthetic benchmark ====
def create_benchmark():
    task_label = 0  # Single task label for all experiences
    task1 = SyntheticSequenceDataset()
    task2 = SyntheticSequenceDataset()
    t1_dataset = AvalancheDataset(task1)
    t2_dataset = AvalancheDataset(task2)
    t1_dataset = as_classification_dataset(task1)
    t2_dataset = as_classification_dataset(task2)
    benchmark = dataset_benchmark(train_datasets=[t1_dataset, t2_dataset],
                                  test_datasets=[t1_dataset, t2_dataset])
    return benchmark

# ==== Main training ====
def main():
    # Hyperparameters
    input_size = 8
    hidden_size = 32
    num_classes = 2
    num_epochs = 3
    lr = 0.001

    # Create benchmark
    benchmark = create_benchmark()

    # Model
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Logger and evaluator
    logger = InteractiveLogger()
    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[logger]
    )

    # Strategy
    strategy = CWRStar(model, optimizer, criterion, cwr_layer_name="classifier", train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
        evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    

    # Training loop
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience)
        print(f"End training on experience {experience.current_experience}")
        print("Evaluating on test stream...")
        strategy.eval(benchmark.test_stream)

if __name__ == "__main__":
    main()
