import re
import numpy as np
import pandas as pd
from random import choices


import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from transformers import BertModel, BertTokenizer
from transformers import logging
logging.set_verbosity_error()

def remove_emojis(data):

    """
    function to remove emoticons from text
    """

    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def stratified_kfold(df, k):

    """
    function for stratified k fold cross validation
    """

    sub_sample_list = []
    for i in df['target'].unique():
        sub_sample = df[df['target'] == i]
        sub_sample.reset_index(inplace=True, drop=True)
        partition = int(len(sub_sample)/k)
        sub_sample['fold'] = np.zeros(len(sub_sample['target']))

        for ii in range(0, k-1):
            sub_sample['fold'].loc[partition*ii:partition*(ii+1)] = ii+1

        for iii in range(k-1,k):
            sub_sample['fold'].loc[partition*iii:] = iii+1
        sub_sample_list.append(sub_sample)
        stratified_df = pd.concat(sub_sample_list)
        stratified_df.reset_index(inplace=True, drop=True)

    return(stratified_df)

def run_gender_model(model, params, X, y, k, vectorizer, multiclass=False):

    """
    function for training gender profiling model on tf-idf features with grid search HPO
    """

    X_train, X_val, y_train, y_val = [
        X['post'][X['fold']!=k], 
        X['post'][X['fold']==k], 
        np.array(y['target'][y['fold']!=k]).astype(int), 
        np.array(y['target'][y['fold']==k]).astype(int)]
    #print(y_val)
    # redefine Xs with tfidf
    X_train_tf = vectorizer.fit_transform(X_train)
    X_train_tf = vectorizer.transform(X_train)
    print("n_samples train: %d, n_features train: %d" % X_train_tf.shape)
    
    X_val_tf = vectorizer.transform(X_val)
    print("n_samples val: %d, n_features val: %d" % X_val_tf.shape)
    #print(X_val_tf)

    clf = GridSearchCV(model, params, n_jobs=1, cv=5)
    clf = clf.fit(X_train_tf, y_train)

    #use best model
    model = clf.best_estimator_
    
    #get metrics
    y_pred = model.predict(X_val_tf)
    accuracy = accuracy_score(y_val, y_pred)
    if multiclass:
        f1 = f1_score(y_val, y_pred, average='micro')
        results = {'accuracy':accuracy, 'f1':f1}
        return(model, results)

    else:
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred)
        results = {'accuracy':accuracy, 'roc_auc':roc_auc, 'f1':f1}
        return(model, results)


def get_bert_average_across_text_tokens(string, tokenizer, model_bert): 
    
    """
    function for retrieving averaged bert embeddings 
    """

    # tokenize
    inputs = tokenizer(string, return_tensors="pt",padding=True, truncation=True)
    # convert input ids to words
    tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    if len(tokens)<=512:
        outputs = model_bert(**inputs)
        bert_av = np.mean(outputs.last_hidden_state[0].detach().numpy(), axis=0)
    else: 
        bert_av = np.zeros(768).tolist()
        
    return bert_av

def run_author_attribution_model(model, params, X, y, k, vectorizer):

    """
    function for training author attributon model on tf-idf and bert embedding features with grid search HPO
    """

    X_train, X_val, y_train, y_val = [
        X['post'][X['fold']!=k], 
        X['post'][X['fold']==k], 
        np.array(y['target'][y['fold']!=k]).astype(int), 
        np.array(y['target'][y['fold']==k]).astype(int)]
    # redefine Xs with tfidf
    
    X_train_tf = vectorizer.fit_transform(X_train)
    X_train_tf = vectorizer.transform(X_train)
    #print("n_samples train: %d, n_features train: %d" % X_train_tf.shape)
    
    X_train_tf_nonsparse = pd.DataFrame.sparse.from_spmatrix(X_train_tf)
    
    bert_feats_list = []
    X_train = X_train.reset_index(drop=True)
    for i in range(0,len(X_train_tf_nonsparse)):
        bert_feats_list.append(pd.DataFrame(get_bert_average_across_text_tokens(string = X_train.loc[i],
                                                                                tokenizer = BertTokenizer.from_pretrained('bert-base-cased'),
                                                                                model_bert = BertModel.from_pretrained('bert-base-cased'))).T)
    bert_df = pd.concat(bert_feats_list)    
    
    X_train_both = pd.merge(X_train_tf_nonsparse, bert_df, left_index=True, right_index=True, suffixes=('_tfidf', '_bert'))
    
    #val
    X_val_tf = vectorizer.transform(X_val)
    #print("n_samples val: %d, n_features val: %d" % X_val_tf.shape)
    
    X_val_tf_nonsparse = pd.DataFrame.sparse.from_spmatrix(X_val_tf)
    
    bert_feats_list = []
    X_val = X_val.reset_index(drop=True)
    for i in range(0,len(X_val_tf_nonsparse)):
        bert_feats_list.append(pd.DataFrame(get_bert_average_across_text_tokens(string = X_val.loc[i],
                                                                                tokenizer = BertTokenizer.from_pretrained('bert-base-cased'),
                                                                                model_bert = BertModel.from_pretrained('bert-base-cased'))).T)
    bert_df = pd.concat(bert_feats_list)    
    #print(bert_df.head())
    
    X_val_both = pd.merge(X_train_tf_nonsparse, bert_df, left_index=True, right_index=True, suffixes=('_tfidf', '_bert'))
    #print(X_val_both.head)

    clf = GridSearchCV(model, params, n_jobs=1, cv=5)
    clf = clf.fit(X_train_both, y_train)

    #use best model
    model = clf.best_estimator_
    
    #get metrics
    y_pred = model.predict(X_val_both)
    accuracy = accuracy_score(y_val, y_pred)
    #roc_auc = roc_auc_score(y_val, y_pred, multi_class='ovo')
    f1 = f1_score(y_val, y_pred, average='micro')
    

    results = {'accuracy':accuracy, 'f1':f1}
    return model, results


def bootstrap(gold, predictions, B=10000, confidence_level=0.95, multiclass=False):
    
    """
    function for bootstrap resampling
    """
    
    critical_value=(1-confidence_level)/2
    lower_sig=100*critical_value
    upper_sig=100*(1-critical_value)
    data=[]
    for g, p in zip(gold, predictions):
        data.append([g,p])

    f1s=[]
    
    for b in range(B):
        choice=choices(data, k=len(data))
        choice=np.array(choice)
        #accuracy=metric(choice[:,0], choice[:,1])
        if multiclass:
            f1 = f1_score(choice[:,0], choice[:,1], average='micro')
        else:
            f1 = f1_score(choice[:,0], choice[:,1])

        f1s.append(f1)
    
    percentiles=np.percentile(f1s, [lower_sig, 50, upper_sig])
    
    lower=percentiles[0]
    median=percentiles[1]
    upper=percentiles[2]
    
    return lower, median, upper, f1s