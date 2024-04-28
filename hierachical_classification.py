# necessary imports
import torch, numpy as np, pandas as pd
from tqdm.auto import tqdm
import wandb, os, random, json

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoModel, BertModel, EvalPrediction
from evaluate import evaluator
import evaluate

from torch import nn
from torch.utils.data import  DataLoader, Subset

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score





model_names = {
    "bert-base": "bert-base-uncased",
    "legal-base": "nlpaueb/legal-bert-base-uncased",
    "legal-echr": "nlpaueb/bert-base-uncased-echr",
    "longformer": "allenai/longformer-base-4096",
    "hat": "kiddothe2b/hierarchical-transformer-base-4096",
}

labels = ['dependency', 'statecontrol', 'victimisation', 'migration', 'discrimination', 'reproductivehealth', 'unpopularviews'] #'intersectionality'


#### fix random seed
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

### creat datset
def create_dataset( tokenizer, path, binary=0, sample_size=None, retokenize=False):
  def to_one_hot(A):  # A is Input array
    a = np.unique(A, return_inverse=1)[1]
    return (a.ravel()[:,None] == np.arange(a.max()+1)).astype(int)
  

  def preprocess_function(doc, binary_mode):
    '''
    binary_mode: 
      1: prepare dataset for binary classification
      0: multi-label classification
    '''
    ### binary classification 
    if binary_mode == 1: 
      print("+++++++++ binary classification ++++++++++++")
      encoding =  tokenizer(doc["fact"], truncation=True, padding=True,return_token_type_ids=False, return_attention_mask=False)
      # one hot binary vulnerabel label 
      v_labels = doc['vulnerable']
      encoding['labels'] = to_one_hot(v_labels)  # binary label to one hot
      

    ### multi-label classification 
    else: 
      text = doc["fact"]
      encoding = tokenizer(text, truncation=True, padding=True,return_token_type_ids=False, return_attention_mask=False)

      labels_batch = {k: doc[k] for k in doc.keys() if k in labels}
      # create numpy array of shape (batch_size, num_labels)
      labels_matrix = np.zeros((len(text), len(labels)))
      # fill numpy array
      for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

      encoding["labels"] = labels_matrix.tolist()
      
      
    return encoding

  ds_dict =  DatasetDict()

  for split in ['train','eval','test','challenge']: #'challenge_71','challenge_58'
    df_split = pd.read_json(f'{path}/{split}.json')
    df_split['fact'] = df_split['fact'].apply(lambda x: ' '.join(x))
    df_split.vulnerable = pd.to_numeric(df_split.vulnerable)
    # sample_size = len(df_split) if sample_size is None else sample_size

    if binary == 1: 
      ds = Dataset.from_pandas(df_split[["vulnerable","fact"]])  
      # tokenize
      print("Tokenizing dataset")
      tok_ds = ds.map(preprocess_function, fn_kwargs={"binary_mode":binary}, batched=True,remove_columns=["fact", '__index_level_0__','vulnerable'])
      num_labels = 2
      # rename vulnerable to the label we will predict
      # tok_ds = tok_ds.rename_columns({"vulnerable": "label"})
      # n_labels = 2
    
    else: 
      ds = Dataset.from_pandas(df_split[["fact", *labels]])
      # tokenize
      tok_ds = ds.map(preprocess_function, fn_kwargs={"binary_mode":binary},batched=True, remove_columns=["fact",'__index_level_0__', *labels])
      num_labels = len(labels)

    ds_dict[split] = tok_ds


  # print("Saving dataset")
  # tok_ds.save_to_disk(f"{path}/hf_dataset")
    
  return ds_dict, num_labels

def preprocess_function_external(doc,  tokenizer):
    '''
    binary_mode: 
      1: prepare dataset for binary classification
      0: multi-label classification
    '''

    text = doc["fact"]
    encoding = tokenizer(text, truncation=True, padding=True,return_token_type_ids=False, return_attention_mask=False)

    labels_batch = {k: doc[k] for k in doc.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
      labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
      
      
    return encoding


def logits_to_prediction(logits):
    cls_preds = torch.sigmoid(logits)  # Sigmoid to map predictions between 0 and 1
    cls_pred_labels = (cls_preds >= 0.5).long()  # Binarize predictions to 0 and 1
    return cls_pred_labels
### metrics
def compute_metrics_2(y_pred,y_true,test_flag = False):
    # y_pred = p.predictions 
    # y_true = p.label_ids
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    accu = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    
    return {'accuracy': accu, 'micro_f1': micro_f1, 'macro_f1':macro_f1,'precision': precision, 'recall':recall}

def compute_metrics_m(predicted, gold, test_flag = False):
    gold = np.array(gold)
    predicted = np.array(predicted)
    
    if test_flag:
        y_true = np.zeros((gold.shape[0], gold.shape[1] + 1), dtype=np.int32)
        y_true[:, :-1] = gold
        y_true[:, -1] = (np.sum(gold, axis=1) == 0).astype('int32')
            # Fix predictions
        y_pred = np.zeros((predicted.shape[0], predicted.shape[1] + 1), dtype=np.int32)
        y_pred[:, :-1] = predicted
        y_pred[:, -1] = (np.sum(predicted, axis=1) == 0).astype('int32')
    else:
       y_true = gold
       y_pred = predicted
    
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    return {'macro_f1': macro_f1, 'micro_f1': micro_f1}
  
### model init
def model_init(checkpoint, num_labels):
    device = 'cuda'
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels,trust_remote_code=True).to(device)
    return model  

### train func
def train(model, dataloader, optimizer, compute_metrics, binary):
    model.train()
    cls_predicted, cls_gold, loss_batch = [], [], []
    
    for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc="Training"):
        
        input_ids, labels = (v.to(device) for v in batch.values()) #token_types, attention_mask
        # forward
        output = model(input_ids=input_ids)
    
        loss = cls_criterion(output.logits, labels.float())
        optimizer.zero_grad()

        loss_batch.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(filter(lambda p: p.requires_grad, model.parameters())), max_norm=1.0)
        optimizer.step()

        if binary:
            cls_pred_labels = output.logits.argmax(-1)
            labels = labels.argmax(-1)


        else:
            cls_pred_labels = logits_to_prediction(output.logits)


        cls_predicted.extend(cls_pred_labels.cpu().detach().numpy())
        cls_gold.extend(labels.cpu().detach().numpy())

    epoch_loss = (torch.stack(loss_batch, dim=0).sum(dim=0)).item()
    metrics = compute_metrics(cls_predicted, cls_gold, test_flag=False)
    return epoch_loss, metrics
### test func

cls_criterion = nn.BCEWithLogitsLoss()

def test(model, dataloader, compute_metrics, binary, test_log=False):
    """
    Function to test or validate the model.
    :param (torch.nn.Module) model: model object to be trained
    :param (torch.utils.data.DataLoader) dataloader: data loader to iterate over batches
    :param (function) compute_metrics: function to compute metrics
    :return: mean epoch loss and metrics
    """
    model.eval()
    cls_predicted, cls_gold, loss_batch = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc="Evaluating"):
            # Get input and targets and get to cuda
            input_ids, labels = (v.to(device) for v in batch.values()) #token_types, attention_mask
            
            # forward
            output = model(input_ids=input_ids)
            
            loss = cls_criterion(output.logits, labels.float())
            loss_batch.append(loss)

            if binary:
                cls_pred_labels = output.logits.argmax(-1)
                labels = labels.argmax(-1)

            else:
                cls_pred_labels = logits_to_prediction(output.logits)


            cls_gold.extend(labels.cpu().detach().numpy())
            cls_predicted.extend(cls_pred_labels.cpu().detach().numpy())

    epoch_loss = (torch.stack(loss_batch, dim=0).sum(dim=0)).item()
    metrics = compute_metrics(cls_predicted, cls_gold, test_flag=True)
    
    if test_log:
      #round the value
      metrics_rounded = {k: round(v, 4) for k, v in metrics.items()}
      print(metrics_rounded)

    return epoch_loss, metrics
### trainer
def trainer(model, train_dataloader, eval_dataloader, optimizer, scheduler, compute_metrics,binary, epochs=1, save=False, path=None, log_wandb=True, log=False):
  if binary:
    column_names = ["mean_epoch_loss",'accuracy', 'micro_f1','macro_f1', 'precision', 'recall'] #, "roc_auc", "accuracy"]
  else:
    column_names = ["mean_epoch_loss", "macro_f1", "micro_f1"] #, "roc_auc", "accuracy"]
  train_df = pd.DataFrame(columns=column_names)
  dev_df = pd.DataFrame(columns=column_names)
  best_dev = 0

  for epoch in tqdm(range(epochs), desc="Tuning"):
      
      train_mean_epoch_loss, train_metrics = train(model, train_dataloader, optimizer, compute_metrics,binary)

      train_df.loc[len(train_df)] = [round(train_mean_epoch_loss, 4)] + [round(x, 4) for x in train_metrics.values()]
      model_path =f'{path}/model.pth.tar'

        

      dev_mean_epoch_loss, dev_metrics = test(model, eval_dataloader, compute_metrics, binary)
      dev_f1 =  dev_metrics['macro_f1'] 
      dev_df.loc[len(dev_df)] = [round(dev_mean_epoch_loss, 4)] + [round(x, 4) for x in dev_metrics.values()]
      
      scheduler.step(dev_mean_epoch_loss)

      if save:
        # save the hyperparameters to JSON
        config_dict = {'learning_rate':learning_rate,"batch_size":batch_size,'epoch':epoch }
        with open(f"{path}/hyperparameter.json", "w") as file:    
          json.dump(config_dict, file)  
          
        # save the model 
        if dev_f1 > best_dev:
          # print(f'\n+++++++++++++ epoch {epoch} \n dev_f1 > best_dev: {dev_f1 > best_dev}\n dev_f1: {dev_f1} \n best_dev: {best_dev} ++++ save model +++++++++++++++\n')
          best_dev = dev_f1
          torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_mean_epoch_loss,
                }, model_path)  
          
          print(f'model saved at {model_path}')  
          
        if dev_f1 > 0.265 and    dev_f1 < 0.2899999:
          temp_path = f'{path}_epoch{str(epoch)}'
          if not os.path.exists(temp_path):
              os.makedirs(temp_path)
          
          torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_mean_epoch_loss,
                }, temp_path+'/model.pth.tar')  
          
          print(f'+++++ temp 0.27 model saved at {temp_path} +++++')  
          
        
      
        if log_wandb:   # and wandb.run is not None:
          wandb.log({"epoch": epoch, "train/epoch_loss": train_mean_epoch_loss, "eval/epoch_loss": dev_mean_epoch_loss})
          wandb.log({f"train/{k}": v for k,v in train_metrics.items()})
          wandb.log({f"eval/{k}": v for k,v in dev_metrics.items()})
          
        if log:
          print(f"Train: {train_df}")
          print(f"Eval: {dev_df}")
          train_df.to_csv(f'./train_logs/{binary_name[binary]}/train_{model_name}.csv')
          dev_df.to_csv(f'./train_logs/{binary_name[binary]}/dev_{model_name}.csv')
        
      # if test_mode:
      #   test_mean_epoch_loss, test_metrics = test(model, eval_dataloader, compute_metrics, binary)
      #   print(f'test_metrics: {test_metrics}')
        

def wandb_train(config=None):
  print(f'mode: {binary_name[binary]}')
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config

    # classes_num = train_dataset.n_labels
    # cls_criterion = nn.BCEWithLogitsLoss()
    # model = model_init(cls_criterion, classes_num)
    model = model_init(checkpoint, num_labels)
    optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=1, threshold=0.1, threshold_mode='abs',
                                                          verbose=True)
    train_dataloader = DataLoader(dss['train'], batch_size=config.batch_size, shuffle=True,)
    eval_dataloader = DataLoader(dss['eval'], batch_size=config.batch_size, shuffle=True,)


    # trainer(model, train_dataloader, eval_dataloader, optimizer, scheduler, compute_metrics, binary,epochs=config.num_epochs, log=False,log_wandb=True)
    trainer(model, train_dataloader, eval_dataloader, optimizer, scheduler, compute_metrics=compute_metrics,\
      binary = binary, epochs=config.num_epochs, save=False, path=save_path, log_wandb=True, log=False)

if __name__ == '__main__':
    

  seed = 42
  seed_everything(seed)
    
        # sweep config
  sweep_config = {
        'method': 'grid'
    }

    # hyperparameters, see https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#parameters
  parameters_dict = {
        'num_epochs': {
            'values': [1,3,5,7],
            },
        'batch_size': {
            'values':  [2,4]#change for longformer [2],[32,16,8]
            },
        'learning_rate': {
            'values': [5e-6,1e-5, 5e-5, 1e-4]

        },
    }

  metric = {
        'name': 'eval/macro_f1',
        'goal': 'maximize'   
        }

  sweep_config['metric'] = metric
  sweep_config['parameters'] = parameters_dict
    
    
  ############### GPU ################
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
  torch.cuda.set_device(2)
  import gc
  torch.cuda.empty_cache()
  gc.collect()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ################################
  binary = 0


  binary_name = {1:'binary',0:'multi'}
  compute_metrics = compute_metrics_2 if binary else compute_metrics_m
  cls_criterion = nn.BCEWithLogitsLoss()

  # model_name = 'legal-echr'
  model_names_list = ['bert-base','legal-base','legal-echr','longformer']
  for model_name in model_names_list:
    print(f"\n+++++++++++ {model_name} ++++++++++++++++")
    checkpoint = model_names[model_name]
    ### tokenzier
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    save_path = f'../models/{binary_name[binary]}/{model_name}_{binary_name[binary]}'
    # if not os.path.exists(save_path):
    #   os.makedirs(save_path)
    
    # selected_path = f'{save_path}_selected'
    # if not os.path.exists(selected_path):
    #   os.makedirs(selected_path)
    

    ### creat dataloader
    ds_path = "../data/"
    dss, num_labels = create_dataset(tokenizer, ds_path, binary=binary, sample_size=None,retokenize=True)
    dss = dss.with_format("torch")
    
    
   ################# hyperparameter #############
 
            
    ################### trainer ###################
    model = model_init(checkpoint, num_labels)
    optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=1, threshold=0.1, threshold_mode='abs',
                                                          verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    

    # eval_dataloader = DataLoader(dss['eval'], batch_size=batch_size, shuffle=True,)
    # test_dataloader = DataLoader(dss['test'], batch_size=batch_size, shuffle=True,)
    challange_dataloader = DataLoader(dss['challenge'], batch_size=batch_size, shuffle=True)

  # ############# test ###############

    model_path = f'{save_path}/model.pth.tar'
    checkpoint = torch.load(model_path,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
      model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
      model.load_state_dict(checkpoint['model_stsate_dict'])
    # test(model, eval_dataloader, compute_metrics, binary,test_log=True)
    # test(model, test_dataloader, compute_metrics, binary,test_log=True)
    test(model, challange_dataloader, compute_metrics, binary,test_log=True)

############# sweep #################
  # wandb.login() # 85f7fe63e27526c3a6d6ae03624a6d0ce7d795e3
  # sweep_id = wandb.sweep(sweep_config, project=f"{binary_name[binary]}_{model_name}_sweep")
  # wandb.agent(sweep_id, wandb_train, count=18)
  # wandb.finish()
