from cmath import e
from gc import callbacks
from incremental_model_training import SaveBestModelCallback, Custom_Hamming_Loss1
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input
from tensorflow.keras.models import Model
from goal_rec_utils.plan import Plan
from goal_rec_utils.attention_layers import AttentionWeights, ContextVector
from utils_unibs.files import load_from_folder
from plan_generator import PlanGeneratorMultiPerc, PlanGeneratorMultiPercAugmented
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Regularizer, L1L2
from sklearn.metrics import hamming_loss
import numpy as np
from os import path
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import datetime
import os
import optuna
import oneHot_deep
from tensorflow.keras.losses import BinaryCrossentropy
import copy


def create_dictionary(plans: list, oneHot: bool = True):
    dictionary = oneHot_deep.create_dictionary(plans)
    dictionary = oneHot_deep.shuffle_dictionary(dictionary)
    if oneHot:
        oneHot_deep.completa_dizionario(dictionary)
    return dictionary


def create_dictionary_goals_not_fixed(plans):
    goals = []
    for p in plans:
        for fact in p.goals:
            if fact not in goals:
                goals.append(fact)
    dizionario_goal = oneHot_deep.create_dictionary_goals(goals)
    dizionario_goal = oneHot_deep.shuffle_dictionary(dizionario_goal)
    oneHot_deep.completa_dizionario(dizionario_goal)
    return dizionario_goal


def create_model(generator: PlanGeneratorMultiPerc, lr: float, embedding_dim: int, lstm_dim: int, dropout: float, rec_dropout: float, lstm_kernel_regularizer: Regularizer):
    input_layer = Input(shape=(generator.max_dim,))
    embedding_layer = Embedding(input_dim=len(generator.dizionario)+1,
                                input_length=generator.max_dim, 
                                output_dim=embedding_dim,
                                mask_zero=True,
                                name='embedding')(input_layer)
    lstm_layer = LSTM(lstm_dim, return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout, activation='linear', kernel_regularizer=lstm_kernel_regularizer, name='lstm')(embedding_layer)
    attention_weights = AttentionWeights(generator.max_dim, name='attention_weights')(lstm_layer)
    context_vector = ContextVector()([lstm_layer, attention_weights])
    output_layer = Dense(len(generator.dizionario_goal), activation='sigmoid', name='dense')(context_vector)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=lr)
    model.compile(loss=BinaryCrossentropy(from_logits=False), optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
    
    return model


def get_params(trial: optuna.Trial) -> dict:
    use_dropout = trial.suggest_categorical('use_dropout', [True, False])
    if use_dropout:
        dropout = trial.suggest_uniform('dropout', 0, 0.5)
    else:
        dropout = 0
    use_recurrent_dropout = trial.suggest_categorical('use_recurrent_dropout', [True, False])

    if use_recurrent_dropout:
        recurrent_dropout = trial.suggest_uniform('rec_dropout', 0, 0.5)
    else:
        recurrent_dropout = 0

    l1 = trial.suggest_categorical('l1', [0, 0.01])
    l2 = trial.suggest_categorical('l2', [0, 0.01])
    
    units=trial.suggest_int('lstm_dim', 150, 512)
    output_dim=trial.suggest_int('embedding_dim', 50, 200)

    return {'lstm_dim' : units,
            'embedding_dim' : output_dim,
            'dropout' : dropout,
            'rec_dropout' : recurrent_dropout,
            'lstm_kernel_regularizer' : L1L2(l1=l1, l2=l2)}


def objective(trial: optuna.Trial,
              train_generator: list,
              val_generator: list,
              epochs: int):
    
    params = get_params(trial)
    model = create_model(generator=train_generator, lr=0.001, **params)
    print(model.summary())
    optuna_callbacks = [
                SaveBestModelCallback(temp_dir=temp_dir, iteration=0, patience=5)
                ]
    model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=2, callbacks=optuna_callbacks)
    y_pred, y_true = get_model_predictions(model, val_generator)
    bce = BinaryCrossentropy()
    y_pred_bce = copy.deepcopy(y_pred)
    y_true_bce = copy.deepcopy(y_true)
    y_pred_bce = np.array(y_pred_bce).reshape(-1, batch_size, len(val_generator.dizionario_goal), 1)
    y_true_bce = np.array(y_true_bce).reshape(-1, batch_size, len(val_generator.dizionario_goal), 1)
    s = 0 
    for i, batch in enumerate(y_pred_bce):
        s += bce(y_true_bce[i], batch)
    s /= len(y_pred_bce)
    loss = bce(y_true_bce, y_pred_bce)
    for i, y in enumerate(y_pred):
        y_pred[i] = [0 if pred < 0.5 else 1 for pred in y]
    result = -hamming_loss(y_true, y_pred) - float(loss)*1e-3
    return result
    
    

def create_study(study_name: str, db_dir: str) -> optuna.Study:
    study = optuna.create_study(
                storage=f'sqlite:///{path.join(db_dir, f"{study_name}.db")}',
                sampler=optuna.samplers.TPESampler(seed=43),
                direction='maximize',
                load_if_exists=True,
                study_name=study_name
            )
    return study

    

    

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

    np.random.seed(420)
    
    plans_dir = '/data/users/mchiari/WMCA/datasets/blocksworld/optimal_plans/plans_max-plan-dim=42_train_percentage=0.8'
    dict_dir = '/data/users/mchiari/WMCA/datasets/blocksworld/optimal_plans/dictionaries_and_plans'
    target_dir = path.join('/data/users/mchiari/WMCA/blocksworld/iterative_results/', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #target_dir = '/data/users/mchiari/WMCA/blocksworld/incremental_results/20230511-091354'
    

    
    test = False
    train = True
    results = False
    live_test = True

    start_index =7
    #iterations = len(train_plans)//(increment*batch_size)
    end_index = 18
    increment = 15
    batch_size = 64
    min_perc = 0.3
    max_perc = 1.0
    max_dim = 42
    optuna_epochs = 20
    epochs = 50
    augmentation_plans =  3
    use_full_plan = True
    patience = 10
    num_trials = 25



    if live_test or results:
        [test_plans] = load_from_folder(plans_dir, ['test_plans'])


    if train:
        [train_plans, val_plans] = load_from_folder(plans_dir, ['train_plans', 'val_plans'])
        

        model = None
        if test:
            start_index = 0
            end_index = 2
            num_trials = 2
            epochs = 2
            optuna_epochs = 1
            target_dir = '/data/users/mchiari/TESTTODELETE/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        logs_dir = path.join(target_dir, 'logs')
        temp_dir = path.join(target_dir, 'temp')
        os.makedirs(logs_dir, exist_ok=True)

        for  iteration in range(start_index, end_index):

            callbacks = [#TensorBoard(log_dir=logs_dir, histogram_freq=1, write_graph=True, write_images=True),
                SaveBestModelCallback(temp_dir=temp_dir, iteration=0, patience=patience)
                ]
            start_index = iteration*increment*batch_size
            print('Iteration: {}'.format(iteration))
            end_index = start_index + increment*batch_size
            train_plans_subset = train_plans[:end_index]
            
            action_dict = create_dictionary(train_plans_subset, False)
            goals_dict = create_dictionary_goals_not_fixed(train_plans_subset)
            val_generator = PlanGeneratorMultiPerc(val_plans, action_dict, goals_dict, batch_size=batch_size, max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, shuffle=False)

            #lr = np.linspace(0.001, 0.00001, iterations)[iteration]
            lr = 0.001

            np.random.shuffle(train_plans_subset)
            train_generator = PlanGeneratorMultiPercAugmented(train_plans_subset, action_dict, goals_dict, num_plans=augmentation_plans, batch_size=batch_size, max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, add_complete=use_full_plan, shuffle=True)
            study_dir = path.join(target_dir, f'studies')
            os.makedirs(study_dir, exist_ok=True)
            study = create_study(f'model_{iteration}', study_dir)
            study.optimize(
                lambda trial: objective(trial=trial, 
                                        train_generator=train_generator,
                                        val_generator=val_generator,
                                        epochs=optuna_epochs),
                n_trials=num_trials,
                gc_after_trial=True
            )

            best_params = study.best_params
            to_pop = []
            for k in best_params.keys():
                if k.startswith('use_'):
                    to_pop.append(k)
            for k in to_pop:
                best_params.pop(k)
            l1 = best_params.pop('l1')
            l2 = best_params.pop('l2')
            best_params['lstm_kernel_regularizer'] = L1L2(l1=l1, l2=l2)
            if 'dropout' not in best_params.keys():
                best_params['dropout'] = 0
            if 'rec_dropout' not in best_params.keys():
                best_params['rec_dropout'] = 0
            
            model = create_model(train_generator, lr=lr, **best_params)
            print(model.summary())
            model.fit(train_generator, epochs=epochs, verbose=2, validation_data=val_generator, callbacks=callbacks)
            model.save(path.join(target_dir, 'model_{0}').format(iteration))
            if live_test:
                test_generator = PlanGeneratorMultiPerc(test_plans, action_dict, goals_dict, batch_size,
                                        max_dim, min_perc, max_perc, shuffle=False)
                y_pred, y_true = get_model_predictions(model, test_generator)
                scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=goals_dict, save_dir=target_dir, filename='metrics_{0}'.format(iteration))