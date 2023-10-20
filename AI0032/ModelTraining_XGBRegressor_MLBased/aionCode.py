
#Standard Library modules
import warnings
import argparse
import importlib
import operator
import platform
import time
import sys
import json
import logging

#Third Party modules
import joblib
import pandas as pd 
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import mlflow
import numpy as np 

#local modules
from utility import *
warnings.filterwarnings("ignore")

model_name = 'XGBRegressor_MLBased'

IOFiles = {
    "inputData": "featureEngineeredData.dat",
    "testData": "test.dat",
    "metaData": "modelMetaData.json",
    "monitor": "monitoring.json",
    "log": "XGBRegressor_MLBased_aion.log",
    "model": "XGBRegressor_MLBased_model.pkl",
    "performance": "XGBRegressor_MLBased_performance.json",
    "metaDataOutput": "XGBRegressor_MLBased_modelMetaData.json"
}

def get_mlflow_uris(config, path):                    
    artifact_uri = None                    
    tracking_uri_type = config.get('tracking_uri_type',None)                    
    if tracking_uri_type == 'localDB':                    
        tracking_uri = 'sqlite:///' + str(path.resolve()/'mlruns.db')                    
    elif tracking_uri_type == 'server' and config.get('tracking_uri', None):                    
        tracking_uri = config['tracking_uri']                    
        if config.get('artifacts_uri', None):                    
            if Path(config['artifacts_uri']).exists():                    
                artifact_uri = 'file:' + config['artifacts_uri']                    
            else:                    
                artifact_uri = config['artifacts_uri']                    
        else:                    
            artifact_uri = 'file:' + str(path.resolve()/'mlruns')                    
    else:                    
        tracking_uri = 'file:' + str(path.resolve()/'mlruns')                    
        artifact_uri = None                    
    if config.get('registry_uri', None):                    
        registry_uri = config['registry_uri']                    
    else:                    
        registry_uri = 'sqlite:///' + str(path.resolve()/'registry.db')                    
    return tracking_uri, artifact_uri, registry_uri                    


def mlflow_create_experiment(config, path, name):                    
    tracking_uri, artifact_uri, registry_uri = get_mlflow_uris(config, path)                    
    mlflow.tracking.set_tracking_uri(tracking_uri)                    
    mlflow.tracking.set_registry_uri(registry_uri)                    
    client = mlflow.tracking.MlflowClient()                    
    experiment = client.get_experiment_by_name(name)                    
    if experiment:                    
        experiment_id = experiment.experiment_id                    
    else:                    
        experiment_id = client.create_experiment(name, artifact_uri)                    
    return client, experiment_id                    


def mlflowSetPath(path, name):                    
    db_name = str(Path(path)/'mlruns')                    
    mlflow.set_tracking_uri('file:///' + db_name)                    
    mlflow.set_experiment(str(Path(path).name))                    


def logMlflow( params, metrices, estimator,tags={}, algoName=None):                    
    run_id = None                    
    for k,v in params.items():                    
        mlflow.log_param(k, v)                    
    for k,v in metrices.items():                    
        mlflow.log_metric(k, v)                    
    if 'CatBoost' in algoName:                    
        model_info = mlflow.catboost.log_model(estimator, 'model')                    
    else:                    
        model_info = mlflow.sklearn.log_model(sk_model=estimator, artifact_path='model')                    
    tags['processed'] = 'no'                    
    tags['registered'] = 'no'                    
    mlflow.set_tags(tags)                    
    if model_info:                    
        run_id = model_info.run_id                    
    return run_id                    

def get_regression_metrices( actual_values, predicted_values):                    
    result = {}                    
                    
    me = np.mean(predicted_values - actual_values)                    
    sde = np.std(predicted_values - actual_values, ddof = 1)                    
                    
    abs_err = np.abs(predicted_values - actual_values)                    
    mae = np.mean(abs_err)                    
    sdae = np.std(abs_err, ddof = 1)                    
                    
    abs_perc_err = 100.*np.abs(predicted_values - actual_values) / actual_values                    
    mape = np.mean(abs_perc_err)                    
    sdape = np.std(abs_perc_err, ddof = 1)                    
                    
    result['mean_error'] = me                    
    result['mean_abs_error'] = mae                    
    result['mean_abs_perc_error'] = mape                    
    result['error_std'] = sde                    
    result['abs_error_std'] = sdae                    
    result['abs_perc_error_std'] = sdape                    
    return result                    

        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)        
    return config
        
def save_model( experiment_id, estimator, features, metrices, params,tags, scoring):        
        # mlflow log model, metrices and parameters        
        with mlflow.start_run(experiment_id = experiment_id, run_name = model_name):        
            return logMlflow(params, metrices, estimator, tags, model_name.split('_')[0])


def train(log):        
    config = validateConfig()        
    targetPath = Path('aion')/config['targetPath']        
    if not targetPath.exists():        
        raise ValueError(f'targetPath does not exist')        
    meta_data_file = targetPath/IOFiles['metaData']        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    log_file = targetPath/IOFiles['log']        
    log = logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)        
    dataLoc = targetPath/IOFiles['inputData']        
    if not dataLoc.exists():        
        return {'Status':'Failure','Message':'Data location does not exists.'}        
        
    status = dict()        
    usecase = config['targetPath']        
    df = pd.read_csv(dataLoc)        
    prev_step_output = meta_data['featureengineering']['Status']

    # split the data for training        
    selected_features = prev_step_output['selected_features']        
    target_feature = config['target_feature']        
    train_features = prev_step_output['total_features'].copy()        
    train_features.remove(target_feature)        
    X_train = df[train_features]        
    y_train = df[target_feature]        
    if config['test_ratio'] > 0.0:        
       test_data = read_data(targetPath/IOFiles['testData'])        
       X_test = test_data[train_features]        
       y_test = test_data[target_feature]        
    else:        
       X_test = pd.DataFrame()        
       y_test = pd.DataFrame()
    
    #select scorer
    scorer = config['scoring_criteria']
    log.info('Scoring criteria: r2')
    
    #Training model
    log.info('Training XGBRegressor for modelBased')
    features = selected_features['modelBased']            
    estimator = XGBRegressor()            
    param = config['algorithms']['XGBRegressor']
    grid = RandomizedSearchCV(estimator, param, scoring=scorer, n_iter=config['optimization_param']['iterations'],cv=config['optimization_param']['trainTestCVSplit'])            
    grid.fit(X_train[features], y_train)            
    train_score = grid.best_score_ * 100            
    best_params = grid.best_params_            
    estimator = grid.best_estimator_
    
    #model evaluation
    if not X_test.empty:
        y_pred = estimator.predict(X_test[features])
        test_score = round(r2_score(y_test,y_pred),2)
        metrices = get_regression_metrices(y_test,y_pred)
        
    else:        
        test_score = train_score        
        metrices = {}
    metrices.update({'train_score': train_score, 'test_score':test_score})
        
    meta_data['training'] = {}        
    meta_data['training']['features'] = features        
    scoring = config['scoring_criteria']        
    tags = {'estimator_name': model_name}        
    monitoring_data = read_json(targetPath/IOFiles['monitor'])        
    mlflow_default_config = {'artifacts_uri':'','tracking_uri_type':'','tracking_uri':'','registry_uri':''}        
    mlflow_client, experiment_id = mlflow_create_experiment(monitoring_data.get('mlflow_config',mlflow_default_config), targetPath, usecase)        
    run_id = save_model(experiment_id, estimator,features, metrices,best_params,tags,scoring)        
    write_json(meta_data,  targetPath/IOFiles['metaDataOutput'])        
    write_json({'scoring_criteria': scoring, 'metrices':metrices, 'param':best_params},  targetPath/IOFiles['performance'])        
        
    # return status        
    status = {'Status':'Success','mlflow_run_id':run_id,'FeaturesUsed':features,'test_score':metrices['test_score'],'train_score':metrices['train_score']}        
    log.info(f'Test score: {test_score}')        
    log.info(f'Train score: {train_score}')        
    log.info(f'MLflow run id: {run_id}')        
    log.info(f'output: {status}')        
    return json.dumps(status)
        
if __name__ == '__main__':        
    log = None        
    try:        
        print(train(log))        
    except Exception as e:        
        if log:        
            log.error(e, exc_info=True)        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))        