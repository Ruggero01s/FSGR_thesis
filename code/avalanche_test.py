# lstm_single_task_avalanche.py
#todo find true dataset to test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.logging import InteractiveLogger
from avalanche.training.supervised import Naive, LwF, EWC, CWRStar, SynapticIntelligence, AR1
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils import as_classification_dataset

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

# ==== Create benchmark ====
def create_benchmark(num_experiences=1):
    #? prepare data
    data = pd.read_csv('DailyDelhiClimateTrain.csv')
    data.head()
    data = data.drop([1461])
    data = data.drop(columns=['date'])
    median = data.loc[data['wind_speed']<15, 'wind_speed'].median()
    data.loc[data['wind_speed'] < 1, 'wind_speed'] = np.nan
    data.loc[data['wind_speed'] > 15, 'wind_speed'] = np.nan
    data['wind_speed'].fillna(median,inplace=True)
    median1 = data.loc[data['meanpressure']<1050, 'meanpressure'].median()
    data.loc[data['meanpressure'] > 1050 , 'meanpressure'] = np.nan 
    data.loc[data['meanpressure'] < 760 , 'meanpressure'] = np.nan
    data['meanpressure'].fillna(median1,inplace=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    scaler2=MinMaxScaler()
    scaler3=MinMaxScaler()
    scaler4=MinMaxScaler()

    meantemp_scaled = scaler.fit_transform(data[['meantemp']])
    humidity_scaled = scaler2.fit_transform(data[['humidity']])
    windspeed_scaled = scaler3.fit_transform(data[['wind_speed']])
    meanpressure_scaled = scaler4.fit_transform(data[['meanpressure']])
    #?
    X_scaled = np.concatenate((meantemp_scaled, humidity_scaled, windspeed_scaled, meanpressure_scaled), axis=1)
    print(X_scaled.shape)
    x=[]
    for i in range(X_scaled.shape[0]-30):
        row = X_scaled[i:i+31]
        x.append(row)

    x = np.array(x)
    X_train = x[:, :-1]  # 1st 30 values as x
    Y_train = x[:, -1, 0]  # last value as y
    X_train = X_train.reshape(-1, 30, 4)  # [batch, seq_len, features]
    Y_train = Y_train.astype(np.int64).flatten()    # Ensure labels are int for classification and 1D

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

    # Wrap in TensorDataset and add targets attribute
    train_tensor_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_tensor_dataset.targets = Y_train_tensor
    
    # Split the dataset into num_experiences parts for continual learning
    experience_size = len(train_tensor_dataset) // num_experiences
    train_datasets = []
    for i in range(num_experiences):
        start_idx = i * experience_size
        end_idx = (i + 1) * experience_size if i < num_experiences - 1 else len(train_tensor_dataset)
        subset = torch.utils.data.Subset(train_tensor_dataset, list(range(start_idx, end_idx)))
        subset.targets = train_tensor_dataset.targets[start_idx:end_idx]
        train_datasets.append(as_classification_dataset(subset))
    
    # same process for Test data ---
    data_test = pd.read_csv('DailyDelhiClimateTest.csv')
    new_data = pd.DataFrame()
    new_data = pd.concat([data.tail(30), data_test], ignore_index=True)
    new_data = new_data.drop(columns=['date'])
    meantemp_scaled1 = scaler.fit_transform(new_data[['meantemp']])
    humidity_scaled1 = scaler2.fit_transform(new_data[['humidity']])
    windspeed_scaled1 = scaler3.fit_transform(new_data[['wind_speed']])
    meanpressure_scaled1 = scaler4.fit_transform(new_data[['meanpressure']])
    new_data_scaled = np.concatenate((meantemp_scaled1, humidity_scaled1, windspeed_scaled1, meanpressure_scaled1), axis=1)

    x1 = []
    for i in range(new_data_scaled.shape[0] - 30):
        row = new_data_scaled[i:i+31]
        x1.append(row)
    x1 = np.array(x1)
    X_test = x1[:, :-1]
    Y_test = x1[:, -1, 0]
    X_test = X_test.reshape(-1, 30, 4)  # [batch, seq_len, features]
    Y_test = Y_test.astype(np.int64).flatten()  # Ensure 1D

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    test_tensor_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    test_tensor_dataset.targets = Y_test_tensor  # <-- add this line
    test_data = as_classification_dataset(test_tensor_dataset)

    benchmark = dataset_benchmark(train_datasets=train_datasets,
                                  test_datasets=[test_data])
    return benchmark

# ==== Main training ====
def main():
    # Hyperparameters
    input_size = 4
    hidden_size = 32
    num_classes = 2
    num_epochs = 3
    lr = 0.001
    num_experiences = 4
    
    
    # Create benchmark
    benchmark = create_benchmark(num_experiences)

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
        forgetting_metrics(experience=True, stream=True),
        loggers=[logger]
    )

    # Strategy
    strategy = CWRStar(model, optimizer, criterion, cwr_layer_name="classifier", train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
        evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # strategy = LwF(model, optimizer, criterion, alpha=0.5, temperature=0.2, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
    #     evaluator=evaluator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # strategy = SynapticIntelligence(model, optimizer, criterion, si_lambda=0.5, train_mb_size=16, train_epochs=num_epochs, eval_mb_size=32,
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

        # train returns a dictionary which contains all the metric values
        res = strategy.train(experience, num_workers=4)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # eval also returns a dictionary which contains all the metric values
        results.append(strategy.eval(benchmark.test_stream, num_workers=4))
        
    for res in results:
        print(res)
            
    

if __name__ == "__main__":
    main()
