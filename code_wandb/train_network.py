import wandb
import yaml
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from goal_rec_utils.attention_layers import AttentionWeights, ContextVector
from plan_generator import PlanGeneratorMultiPerc, PlanGeneratorMultiPercAugmented
from keras import metrics
from keras.regularizers import Regularizer, L1L2
from keras import backend as K
import oneHot_deep
from utils_unibs.files import load_from_folder
import copy
from sklearn.metrics import hamming_loss
from keras.losses import BinaryCrossentropy
import datetime
from keras.callbacks import EarlyStopping
import numpy as np
from goal_rec_utils.plan import Plan


config = {
    'lr': 0.001,
    'embedding_dim': 128,
    'lstm_dim': 128,
    'dropout': 0.2,
    'rec_dropout': 0.2,
    'lstm_kernel_regularizer': None
}

def get_model_predictions(model: Model, test_generator: PlanGeneratorMultiPerc) -> list:
    y_pred = list()
    y_true = list()
    for i in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(i)
        y_pred.extend(model.predict(x))
        y_true.extend(y)
    return y_pred, y_true


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

def Custom_Hamming_Loss1(y_true, y_pred):
  tmp = K.abs(y_true-y_pred)
  return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))

def get_source_dir(domain: str) -> str:
    if domain == 'blocksworld':
        return '/data/users/mchiari/WMCA/data_wandb/bw_plans_max-plan-dim=75_train_percentage=0.7'
    if domain =='depots':
        return '/data/users/mchiari/WMCA/data_wandb/dp_plans_max-plan-dim=50_train_percentage=0.7'
    else:
        return None

def create_model(generator: PlanGeneratorMultiPercAugmented, lr: float, embedding_dim: int, lstm_dim: int, dropout: float, rec_dropout: float, lstm_kernel_regularizer: Regularizer):
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
    model.compile(loss='bce', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss1, metrics.Precision(name='precision')])
    return model

def main():

  wandb.login()

  with open('/data/users/mchiari/WMCA/code_wandb/test_config.yml') as file:
     config = yaml.load(file, Loader=yaml.FullLoader)
  
  project_name = config.pop('project_name')

  increment = 15*64

  run = wandb.init(project=project_name, config=config)

  
  domain = run.config.domain
  iteration = run.config.iteration
  min_plan_perc = run.config.min_plan_perc
  max_plan_perc = run.config.max_plan_perc
  max_plan_dim = run.config.max_plan_dim
  use_full_plan = run.config.use_full_plan
  lr = run.config.learning_rate
  lstm_units = run.config.lstm_units
  embedding_dim = run.config.embedding_dim
  epochs = run.config.epochs
  dropout = run.config.dropout
  rec_dropout = run.config.rec_dropout
  batch_size = run.config.batch_size
  l1 = run.config.l1
  l2 = run.config.l2
  lstm_kernel_regularizer = L1L2(l1=l1, l2=l2)
  batch_size = run.config.batch_size
  epochs = run.config.epochs
  augmentation_plans = run.config.augmentation_plans
  
  source_dir = get_source_dir(domain)
  [train_plans, val_plans] = load_from_folder(source_dir, ['train_plans', 'val_plans'])
  
  
  callbacks = [
      EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
      wandb.keras.WandbCallback()
  ]
  start_index = iteration*increment
  print('Iteration: {}'.format(iteration))
  end_index = start_index + increment
  train_plans_subset = train_plans[:end_index]
  
  action_dict = create_dictionary(train_plans_subset, False)
  goals_dict = create_dictionary_goals_not_fixed(train_plans_subset)

  wandb.log({'actions' : len(action_dict), 'goals' : len(goals_dict)})

  val_generator = PlanGeneratorMultiPerc(val_plans,
                                         action_dict, 
                                         goals_dict, 
                                         batch_size=batch_size, 
                                         max_dim=max_plan_dim, 
                                         min_perc=min_plan_perc,
                                         max_perc=max_plan_perc, 
                                         shuffle=False)
  params = {'lstm_dim' : lstm_units,
            'embedding_dim' : embedding_dim,
            'dropout' : dropout,
            'rec_dropout' : rec_dropout,
            'lstm_kernel_regularizer' : lstm_kernel_regularizer}
  
  np.random.seed(420)
  np.random.shuffle(train_plans_subset)
  
  train_generator = PlanGeneratorMultiPercAugmented(train_plans_subset, 
                                                    action_dict, 
                                                    goals_dict, 
                                                    num_plans=augmentation_plans, 
                                                    batch_size=batch_size, 
                                                    max_dim=max_plan_dim, 
                                                    min_perc=min_plan_perc, 
                                                    max_perc=max_plan_perc, 
                                                    add_complete=use_full_plan, 
                                                    shuffle=True)    

  model = create_model(generator=train_generator, lr=lr, **params)
  model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=2, callbacks=callbacks)

if __name__ == '__main__':
  main()





  