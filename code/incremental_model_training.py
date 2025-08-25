from cmath import e
from gc import callbacks
from keras.layers import Dense, LSTM, Embedding, Input, Softmax
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from plan import Plan
from utils import AttentionWeights, ContextVector
from utils import load_from_folder
from plan_generator import PlanGeneratorMultiPerc, PlanGeneratorMultiPercAugmented
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from keras.models import load_model
from keras import metrics
from keras.optimizers import Adam
import numpy as np
from os import path
from keras import backend as K
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import datetime
import os
import warnings
import wandb  # Import wandb for logging

# Modify SaveBestModelCallback to track actual epochs trained
class SaveBestModelCallback(Callback):

    def __init__(self, temp_dir: str, patience: int, iteration: int):
        self.temp_dir = temp_dir
        self.max_prec = -1
        self.min_val_loss = 1000
        self.iteration = iteration
        self.best_weights = None
        self.patience = patience
        self.epochs_trained = 0  # Add counter for epochs trained

    def on_train_begin(self, logs=None):
        self.max_prec = -1
        self.min_val_loss = 1000
        self.best_weights = self.model.get_weights()
        self.increased_epochs = 0
        self.epochs_trained = 0  # Reset counter
    
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_trained = epoch + 1  # Update epochs trained counter
        
        if self.iteration > 5:
            new_best, values = self.prioritize_precision(logs['val_precision'], logs['loss'], logs['val_loss'], self.max_prec, self.min_val_loss, self.model, self.best_weights)
            if new_best:
                [self.max_prec, self.min_val_loss, self.best_weights] = values
                self.increased_epochs = 0
                message = f"New best model found with precision {logs['val_precision']:.4f} and loss {logs['val_loss']:.4f}"
                print(message)
                # wandb.log({"early_stopping_event": message})
        else:
            new_best, values = self.prioritize_loss(logs['loss'], logs['val_loss'], self.min_val_loss, self.model, self.best_weights)
            if new_best:
                [self.min_val_loss, self.best_weights] = values
                self.increased_epochs = 0
                message = f"New best model found with loss {logs['val_loss']:.4f}"
                print(message)
                # wandb.log({"early_stopping_event": message})
        
        if not new_best:
            self.increased_epochs += 1
            if self.increased_epochs > self.patience:
                self.model.stop_training = True
                # wandb.log({"early_stopping_epoch": epoch + 1})
                print(f"Early stopping triggered at epoch {epoch+1}")

    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.model.save(path.join(self.temp_dir, 'model_end.keras'))
            print('Best model weights restored')


    @staticmethod
    def prioritize_precision(val_prec: float, loss: float, val_loss: float, max_prec: float, min_val_loss: float, model: Model, weights: list):
        new_best = False
        if val_prec > max_prec and val_loss < 1 and loss < 1:
            print(f'New best model found with precision {val_prec} and loss {val_loss}')
            max_prec = val_prec
            min_val_loss = val_loss
            weights = model.get_weights()
            new_best = True
        elif np.abs(val_prec - max_prec) < 1e-5 and val_loss < min_val_loss:
            print(f'New best model found with precision {val_prec} and loss {val_loss}')
            min_val_loss = val_loss
            weights = model.get_weights()
            new_best = True

        return new_best, [max_prec, min_val_loss, weights]

    @staticmethod
    def prioritize_loss(loss: float, val_loss: float, min_val_loss: float, model: Model, weights: list):
        new_best = False
        if val_loss < min_val_loss and loss < 1 and val_loss < 1:
            print(f'New best model found with loss {val_loss}')
            min_val_loss = val_loss
            weights = model.get_weights()
            new_best = True
        return new_best, [min_val_loss, weights]


# Custom Keras callback to log metrics to wandb
class WandbCallback(Callback):
    
    def __init__(self, iteration=None, epochs_trained=0):
        self.iteration = iteration
        self.epochs_trained = epochs_trained  # Track epochs trained
        super().__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_trained += 1
        log_dict = {
            "train_epochs": self.epochs_trained,
            "epoch": epoch + 1,
            "train_loss": logs.get('loss'),
            "train_accuracy": logs.get('accuracy'),
            "train_hamming": logs.get('Custom_Hamming_Loss1'),
            "train_precision": logs.get('precision'),
            "val_loss": logs.get('val_loss'),
            "val_accuracy": logs.get('val_accuracy'),
            "val_hamming": logs.get('val_Custom_Hamming_Loss1'),
            "val_precision": logs.get('val_precision'),
        }
        
        if self.iteration is not None:
            log_dict["iteration"] = self.iteration
            
        wandb.log(log_dict)


def Custom_Hamming_Loss1(y_true, y_pred):
    #! translated from the following code:
    #? tmp = K.abs(y_true-y_pred)
    #? return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))
    # round‐style Hamming loss: count how many preds differ >0.5 from true
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    diff = tf.cast(tf.abs(y_true - y_pred), tf.float64)
    mismatches = tf.cast(tf.greater(diff, 0.5), tf.float64)
    return tf.reduce_mean(mismatches)

def create_model(generator: PlanGeneratorMultiPerc, lr: float):
    input_layer = Input(shape=(generator.max_dim,))
    embedding_layer = Embedding(input_dim=len(generator.dizionario)+1, # vocab size
                                output_dim=85, # embedding size
                                mask_zero=True,
                                name='embedding')(input_layer)
    # using hyperparameters from optuna optimization
    lstm_layer = LSTM(units=446, return_sequences=True, dropout=0.12, recurrent_dropout=0.01, activation='linear', name='lstm')(embedding_layer)
    attention_weights = AttentionWeights(generator.max_dim, name='attention_weights')(lstm_layer)
    context_vector = ContextVector()([lstm_layer, attention_weights])
    output_layer = Dense(len(generator.dizionario_goal), activation='sigmoid', name='dense')(context_vector)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])

    return model

def get_model_predictions(model: Model, test_generator: PlanGeneratorMultiPerc) -> list:
    y_pred = list()
    y_true = list()
    for i in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(i)
        y_pred.extend(model.predict(x))
        y_true.extend(y)
    return y_pred, y_true

def print_metrics(y_true: list, y_pred: list, dizionario_goal: dict, save_dir: str = None,
                  filename: str = 'metrics') -> list:
    # Convert predictions to binary
    for i, y in enumerate(y_pred):
        y_pred[i] = [0 if pred < 0.5 else 1 for pred in y]
    
    labels = list(dizionario_goal.keys())
    to_print = []
    accuracy = accuracy_score(y_true, y_pred)
    hamming_loss_score = hamming_loss(y_true, y_pred)
    to_print.append(f'Accuracy: {accuracy}\n')
    to_print.append(f'Hamming Loss: {hamming_loss_score}\n')
    to_print.append(classification_report(y_true, y_pred, target_names=labels))
    if save_dir is None:
        for line in to_print:
            print(line)
    else:
        with open(path.join(save_dir, f'{filename}.txt'), 'w') as file:
            for line in to_print:
                file.write(line)
            file.close()
    return [accuracy, hamming_loss_score]

def run_tests(model: Model, test_plans: list, dizionario: dict, dizionario_goal: dict, batch_size: int,
              max_plan_dim: int,
              min_plan_perc: float, plan_percentage: float, save_dir: str, filename='metrics') -> None:
    if test_plans is not None:
        test_generator = PlanGeneratorMultiPerc(test_plans, dizionario, dizionario_goal, batch_size,
                                                max_plan_dim, min_plan_perc, plan_percentage, shuffle=False)
        y_pred, y_true = get_model_predictions(model, test_generator)
        scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=dizionario_goal, save_dir=save_dir,
                               filename=filename)

if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU') # Uncomment this line to disable GPU usage
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )
    np.random.seed(420)
    warnings.filterwarnings("ignore")

    plans_dir = './datasets/gr_logistics/pickles'
    #plans_dir = 'datasets/logistics/optimal_plans/plans_max-plan-dim=30_train_percentage=0.8' #old plans
    dict_dir = './datasets/gr_logistics/pickles'
    target_dir = path.join('./datasets/gr_logistics/results/incremental_tensorflow', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logs_dir = path.join(target_dir, 'logs')
    temp_dir = path.join(target_dir, 'temp')
    # [action_dict, goals_dict] = load_from_folder(dict_dir, ['dizionario', 'dizionario_goal'])
    [action_dict, goals_dict] = load_from_folder(
        dict_dir, ["action_dict.pkl", "goal_dict.pkl"]
    )

    # Control flow flags:
    test = False          # Whether to run a quick test of training logic
    train = True          # Enable the training loop
    results = False       # Whether to collect final results over saved iterations
    live_test = True      # Perform evaluation on test set after each iteration
    random_model = False   #? If True non traina e fa un random per baseline?
    incremental = True
    
    
    # Incremental training parameters:
    start_index = 0               # Starting index in train_plans for current iteration
    increment = 16                # Number of batches (of size batch_size) per iteration #? è quante batches è un "experience"?
    batch_size = 64               # Number of samples per batch
    old_plans_percentage = 2      # Fraction of previous plans to include in new increment
    min_perc = 0.3                # Minimum fraction of plan to use in generator
    max_perc = 1                # Maximum fraction of plan to use in generator
    max_dim = 32                  # Maximum plan length in dataset #todo check
    epochs = 30                   # Epochs per iteration
    patience = 5                  # Early stopping patience in epochs

    # Data augmentation:
    augmentation_plans = 4        # Number of plans to augment per batch
    use_full_plan = True          # Whether to include the complete plan in augmentation

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="fsgr-plan-training",
        name=f"incremental_tensorflow_old_factor_2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "increment": increment,
            "batch_size": batch_size,
            "old_plans_percentage": old_plans_percentage,
            "min_perc": min_perc,
            "max_perc": max_perc,
            "max_dim": max_dim,
            "epochs": epochs,
            "patience": patience,
            "augmentation_plans": augmentation_plans,
            "use_full_plan": use_full_plan,
            "incremental": incremental,
            "test_mode": test,
            "framework": "tensorflow",
            "random_model": random_model,
        }
    )

    if live_test or results:
        [test_plans] = load_from_folder(plans_dir, ['test_plans'])
        test_generator = PlanGeneratorMultiPerc(test_plans, action_dict, goals_dict, batch_size,
                                        max_dim, min_perc, max_perc, shuffle=False)
        # Log dataset info
        wandb.log({
            "test_plans_count": len(test_plans)
        })

    if train:
        [train_plans, val_plans] = load_from_folder(plans_dir, ['train_plans', 'val_plans'])
        os.makedirs(logs_dir, exist_ok=True)
        print(f'Training on {len(train_plans)} plans, validation on {len(val_plans)} plans')
        
        # Log dataset info
        wandb.log({
            "vocab_size": len(action_dict),
            "goal_size": len(goals_dict),
            "train_plans_count": len(train_plans),
            "val_plans_count": len(val_plans),
        })
        
        model = None
        if test:
            iterations = 2
        else:
            iterations = len(train_plans)//(increment*batch_size)

        print(f'Iterations: {iterations}')
        wandb.log({"total_iterations": iterations})
        
        # Initialize total epochs counter
        epochs_trained = 0
        
        val_generator = PlanGeneratorMultiPerc(val_plans, action_dict, goals_dict, batch_size=batch_size,
                                               max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, shuffle=False)
        if incremental == True:
            for iteration in range(0, iterations):
                # Create early stopping callback and keep reference to it
                save_best_callback = SaveBestModelCallback(temp_dir=temp_dir, iteration=iteration, patience=patience)
                wandb_callback = WandbCallback(iteration=iteration, epochs_trained=epochs_trained)  # Create WandbCallback
                callbacks = [
                    TensorBoard(log_dir=logs_dir, histogram_freq=1, write_graph=True, write_images=True),
                    ModelCheckpoint(filepath=path.join(temp_dir, 'model.h5.keras'), monitor='val_loss', save_best_only=True),
                    save_best_callback,  # Use the reference
                    wandb_callback  # Add WandbCallback
                ]

                # selects starting index for current iteration so that we look at new plans
                start_index = iteration*increment*batch_size
                print('Iteration: {}'.format(iteration))
                end_index = start_index + increment*batch_size
                train_plans_subset = train_plans[start_index:end_index]

                lr = np.linspace(0.001, 0.00001, iterations)[iteration]
                
                # Log iteration info
                # wandb.log({
                #     "iteration": iteration,
                #     "current_learning_rate": lr,
                #     "train_plans_subset_size": len(train_plans_subset),
                # })

                if iteration > 1:
                    # train_plans_subset.extend(train_plans[0:start_index])
                    train_plans_subset.extend(np.random.choice(train_plans[0:start_index], int(old_plans_percentage*increment*batch_size), replace=False))
                start_index = end_index
                np.random.shuffle(train_plans_subset)
                # * we can use the augmented generator
                train_generator = PlanGeneratorMultiPercAugmented(train_plans_subset, action_dict, goals_dict,
                                                                  num_plans=augmentation_plans, batch_size=batch_size,
                                                                  max_dim=max_dim, min_perc=min_perc, max_perc=max_perc,
                                                                  add_complete=use_full_plan, shuffle=True)
                # train_generator = PlanGeneratorMultiPerc(train_plans_subset, action_dict, goals_dict,
                #                                         batch_size=batch_size, max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, shuffle=True)

                print(f"Iteration {iteration} - Training on {len(train_generator.plans)} plans, in {train_generator.num_batches} batches")
                
                # Log training info
                # wandb.log({
                #     f"iteration_{iteration:03d}/total_plans": len(train_generator.plans),
                #     f"iteration_{iteration:03d}/num_batches": train_generator.num_batches,
                # })
                
                if iteration == 0:
                    model = create_model(train_generator, lr=lr)
                    print(model.summary())
                    
                    # Log model architecture
                    wandb.log({
                        "model_architecture": str(model.summary()),
                        "model_parameters": model.count_params(),
                    })
                else:
                    model = load_model(path.join(target_dir, f'model_{iteration-1}.keras'), custom_objects={'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
                    'AttentionWeights': AttentionWeights, 
                    'ContextVector': ContextVector
                    })
                    optimizer = Adam(learning_rate=lr)
                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])

                if random_model:
                    model = create_model(train_generator, lr=lr)
                else:
                    history = model.fit(train_generator, epochs=epochs, verbose=2, validation_data=val_generator, callbacks=callbacks)
                    
                    # Update total epochs counter with actual epochs trained
                    epochs_trained = wandb_callback.epochs_trained                    
                    # Log epochs data
                    # wandb.log({
                    #     # f"iteration_{iteration:03d}/epochs_trained": epochs_trained,
                    #     "total_epochs": total_epochs
                    # })
                    
                model.save(path.join(target_dir, 'model_{0}.keras').format(iteration))
                
                if live_test:
                    model = load_model(path.join(target_dir, f'model_{iteration}.keras'), 
                                    custom_objects={'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
                                        'AttentionWeights': AttentionWeights, 
                                        'ContextVector': ContextVector
                                        })
                    optimizer = Adam(learning_rate=lr)
                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
                    y_pred, y_true = get_model_predictions(model, test_generator)
                    scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=goals_dict, save_dir=target_dir, filename='metrics_{0}'.format(iteration))
                    
                    # Log test metrics
                    # wandb.log({
                    #     f"iteration_{iteration:03d}/test_accuracy": scores[0],
                    #     f"iteration_{iteration:03d}/test_hamming_loss": scores[1],
                    # })
        else:
            # For non-incremental training, similar approach
            save_best_callback = SaveBestModelCallback(temp_dir=temp_dir, iteration=0, patience=patience)
            
            callbacks = [
                TensorBoard(log_dir=logs_dir, histogram_freq=1, write_graph=True, write_images=True),
                ModelCheckpoint(filepath=path.join(temp_dir, 'model.h5.keras'), monitor='val_loss', save_best_only=True),
                save_best_callback,  # Use the reference
                WandbCallback()  # Add WandbCallback
            ]

            lr = 0.0001

            np.random.shuffle(train_plans)
            # * we can use the augmented generator
            train_generator = PlanGeneratorMultiPercAugmented(train_plans, action_dict, goals_dict,
                                                               num_plans=augmentation_plans, batch_size=batch_size,
                                                               max_dim=max_dim, min_perc=min_perc, max_perc=max_perc,
                                                               add_complete=use_full_plan, shuffle=True)
            #train_generator = PlanGeneratorMultiPerc(train_plans, action_dict, goals_dict,
            #                                        batch_size=batch_size, max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, shuffle=True)

            model = create_model(train_generator, lr=lr)
            print(model.summary())
            
            # Log model architecture
            wandb.log({
                "model_architecture": str(model.summary()),
                "model_parameters": model.count_params(),
                "total_plans": len(train_generator.plans),
                "num_batches": train_generator.num_batches,
                "current_learning_rate": lr,
            })
            
            optimizer = Adam(learning_rate=lr)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])

            if random_model:
                model = create_model(train_generator, lr=lr)
            else:
                history = model.fit(train_generator, epochs=epochs, verbose=2, validation_data=val_generator, callbacks=callbacks)
                
                # Update total epochs counter with actual epochs trained
                epochs_trained += save_best_callback.epochs_trained
                
            model.save(path.join(target_dir, 'model_FULL.keras'))
            if live_test:
                model = load_model(path.join(target_dir, f'model_FULL.keras'), 
                                custom_objects={'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
                                    'AttentionWeights': AttentionWeights, 
                                    'ContextVector': ContextVector
                                    })
                optimizer = Adam(learning_rate=lr)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
                y_pred, y_true = get_model_predictions(model, test_generator)
                scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=goals_dict, save_dir=target_dir, filename='metrics_FULL')
                
                # Log test metrics
                wandb.log({
                    "test_accuracy": scores[0],
                    "test_hamming_loss": scores[1],
                })
            
        # Add total epochs to final report
        print(f"\n=== Training Complete ===")
        
        # Log final total epochs
        # wandb.log({"final_total_epochs": total_epochs})
    
    if results:
        start_iteration = 0
        end_iteration = 18
        model_path = path.join(target_dir, 'model_{}.keras')

        for iteration in range(start_iteration, end_iteration+1):
            model = load_model(model_path.format(iteration), custom_objects={
            'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
            'AttentionWeights': AttentionWeights,
            'ContextVector': ContextVector
            })
            y_pred, y_true = get_model_predictions(model, test_generator)
            scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=goals_dict, save_dir=path.dirname(model_path), filename='metrics_{0}'.format(iteration))
            
            # Log results
            # wandb.log({
            #     f"results_iteration_{iteration}/test_accuracy": scores[0],
            #     f"results_iteration_{iteration}/test_hamming_loss": scores[1],
            # })
    
    # Close wandb run
    wandb.finish()
