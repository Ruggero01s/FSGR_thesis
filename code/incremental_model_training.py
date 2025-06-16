from cmath import e
from gc import callbacks
from keras.layers import Dense, LSTM, Embedding, Input
from keras.models import Model
from plan import Plan
# from goal_rec_utils.attention_layers import AttentionWeights, ContextVector
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

class SaveBestModelCallback(Callback):

    def __init__(self, temp_dir: str, patience: int, iteration: int):
        self.temp_dir = temp_dir
        self.max_prec = -1
        self.min_val_loss = 1000
        self.iteration = iteration
        self.best_weights = None
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.max_prec = -1
        self.min_val_loss = 1000
        self.best_weights = self.model.get_weights()
        self.increased_epochs = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if self.iteration > 5:
            new_best, values = self.prioritize_precision(logs['val_precision'], logs['loss'], logs['val_loss'], self.max_prec, self.min_val_loss, self.model, self.best_weights)
            if new_best:
                [self.max_prec, self.min_val_loss, self.best_weights] = values
                self.increased_epochs = 0
        else:
            new_best, values = self.prioritize_loss(logs['loss'], logs['val_loss'], self.min_val_loss, self.model, self.best_weights)
            if new_best:
                [self.min_val_loss, self.best_weights] = values
                self.increased_epochs = 0
        
        if not new_best:
            self.increased_epochs += 1
            if self.increased_epochs > self.patience:
                self.model.stop_training = True

    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.model.save(path.join(self.temp_dir, 'model.h5'))
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




def Custom_Hamming_Loss1(y_true, y_pred):
  tmp = K.abs(y_true-y_pred)
  return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))


def create_model(generator: PlanGeneratorMultiPerc, lr: float):
    
    

    input_layer = Input(shape=(generator.max_dim,))
    embedding_layer = Embedding(input_dim=len(generator.dizionario)+1,
                                input_length=generator.max_dim, 
                                output_dim=83,
                                mask_zero=True,
                                name='embedding')(input_layer)
    #? why return sequence was True?
    lstm_layer = LSTM(350, return_sequences=False, dropout=0, recurrent_dropout=0, activation='linear', name='lstm')(embedding_layer)
    # attention_weights = AttentionWeights(generator.max_dim, name='attention_weights')(lstm_layer)
    # context_vector = ContextVector()([lstm_layer, attention_weights])
    output_layer = Dense(len(generator.dizionario_goal), activation='sigmoid', name='dense')(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
    
    return model

def get_model_predictions(model: Model, test_generator: PlanGeneratorMultiPerc) -> list:
    y_pred = list()
    y_true = list()
    for i in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(i)
        y_pred.extend(model.predict(x))
        y_true.extend(y)
        # print(np.shape(y_pred)) #todo capire perchè questo coso ha 3D || era perchè lstm layer return_sequences = True
    return y_pred, y_true

def print_metrics(y_true: list, y_pred: list, dizionario_goal: dict, save_dir: str = None,
                  filename: str = 'metrics') -> list:
    #todo capire vettore pred come voleva essere fatto
    for i, y in enumerate(y_pred):
        y_pred[i] = [0 if pred < 0.5 else 1 for pred in y]
    # for i in range(len(y_pred)):
    #     y_rounded = y_pred[i]
    #     # print(y_rounded)
    #     for k, pred in enumerate(y_rounded):
    #         # print(pred)
    #         if pred < 0.5:
    #             y_rounded[k] = 0
    #         else:
    #             y_rounded[k] = 1
    #     y_pred[i] = y_rounded

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
    
    plans_dir = './datasets/gr_logistics/pickles'
    dict_dir = './datasets/gr_logistics/pickles'
    target_dir = path.join('./datasets/gr_logistics/results/random/', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #target_dir = '/data/users/mchiari/WMCA/blocksworld/incremental_results/20230511-091354'
    logs_dir = path.join(target_dir, 'logs')
    temp_dir = path.join(target_dir, 'temp')
    [action_dict, goals_dict] = load_from_folder(dict_dir, ['action_dict.pkl', 'goal_dict.pkl'])

    
    test = True
    train = True
    results = False
    live_test = True
    random_model = True

    start_index = 0
    increment = 15
    batch_size = 64
    old_plans_percentage = 1
    min_perc = 0.3
    max_perc = 1.0
    max_dim = 26
    epochs = 50

    augmentation_plans =  4
    use_full_plan = True
    patience = 10

    os.makedirs(target_dir, exist_ok=True)
    
    if live_test or results:
        [test_plans] = load_from_folder(plans_dir, ['test_plans.pkl'])
        test_generator = PlanGeneratorMultiPerc(test_plans, action_dict, goals_dict, batch_size,
                                        max_dim, min_perc, max_perc, shuffle=False)
        #print(f"------------------------------------------\n{test_generator.plans}\n-----------------------------------------")


    if train:
        [train_plans, val_plans] = load_from_folder(plans_dir, ['train_plans.pkl', 'val_plans.pkl'])
        os.makedirs(logs_dir, exist_ok=True)

        model = None
        if test:
            iterations = 2
        else:
            iterations = len(train_plans)//(increment*batch_size)
        
        val_generator = PlanGeneratorMultiPerc(val_plans, action_dict, goals_dict, batch_size=batch_size, max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, shuffle=False)
        
        for  iteration in range(0, iterations):

            callbacks = [TensorBoard(log_dir=logs_dir, histogram_freq=1, write_graph=True, write_images=True),
                #ModelCheckpoint(filepath=path.join(temp_dir, 'model.h5'), monitor='val_loss', save_best_only=True)
                SaveBestModelCallback(temp_dir=temp_dir, iteration=iteration, patience=patience)
                ]
            start_index = iteration*increment*batch_size
            print('Iteration: {}'.format(iteration))
            end_index = start_index + increment*batch_size
            train_plans_subset = train_plans[start_index:end_index]

            lr = np.linspace(0.001, 0.00001, iterations)[iteration]

            if start_index > 0:
                # train_plans_subset.extend(train_plans[0:start_index])
                train_plans_subset.extend(np.random.choice(train_plans[0:start_index], int(old_plans_percentage*increment*batch_size), replace=False))
            start_index = end_index
            np.random.shuffle(train_plans_subset)
            train_generator = PlanGeneratorMultiPercAugmented(train_plans_subset, action_dict, goals_dict, num_plans=augmentation_plans, batch_size=batch_size, max_dim=max_dim, min_perc=min_perc, max_perc=max_perc, add_complete=use_full_plan, shuffle=True)
            if iteration == 0:
                model = create_model(train_generator, lr=lr)
                print(model.summary())
            else:
                model = load_model(path.join(target_dir, f'model_{iteration-1}.keras'), custom_objects={'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
                                                                                    # 'AttentionWeights': AttentionWeights, 
                                                                                    # 'ContextVector': ContextVector
                                                                                    })
                optimizer = Adam(learning_rate=lr)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
            
            if random_model:
                model = create_model(train_generator, lr=lr)
            else:
                model.fit(train_generator, epochs=epochs, verbose=2, validation_data=val_generator, callbacks=callbacks)
            model.save(path.join(target_dir, 'model_{0}.keras').format(iteration))
            if live_test:
                model = load_model(path.join(target_dir, f'model_{iteration}.keras'), custom_objects={'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
                                                                                    # 'AttentionWeights': AttentionWeights, 
                                                                                    # 'ContextVector': ContextVector
                                                                                    })
                optimizer = Adam(learning_rate=lr)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
                y_pred, y_true = get_model_predictions(model, test_generator)
                scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=goals_dict, save_dir=target_dir, filename='metrics_{0}'.format(iteration))


    if results:
        start_iteration = 0
        end_iteration = 18
        #!
        model_path = '' #todo 
        #!

        
        for iteration in range(start_iteration, end_iteration+1):

            model = load_model(model_path.format(iteration), custom_objects={'Custom_Hamming_Loss1': Custom_Hamming_Loss1, 
                                                                            #  'AttentionWeights': AttentionWeights,
                                                                            #  'ContextVector': ContextVector
                                                                             })
            y_pred, y_true = get_model_predictions(model, test_generator)
            scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=goals_dict, save_dir=path.dirname(model_path), filename='metrics_{0}'.format(iteration))
