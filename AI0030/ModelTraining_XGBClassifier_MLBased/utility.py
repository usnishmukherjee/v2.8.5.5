#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AI0030_1 usecase.
File generation time: 2023-10-20 15:22:48
'''
#Standard Library modules
import json
import logging
import io

#Third Party modules
import pandas as pd 
        
def read_json(file_path):        
    data = None        
    with open(file_path,'r') as f:        
        data = json.load(f)        
    return data        

        
def write_json(data, file_path):        
    with open(file_path,'w') as f:        
        json.dump(data, f)        

        
def read_data(file_path, encoding='utf-8', sep=','):        
    return pd.read_csv(file_path, encoding=encoding, sep=sep)        

        
def write_data(data, file_path, index=False):        
    return data.to_csv(file_path, index=index)        
        
#Uncomment and change below code for google storage        
#from google.cloud import storage        
#def write_data(data, file_path, index=False):        
#    file_name= file_path.name        
#    data.to_csv('output_data.csv')        
#    storage_client = storage.Client()        
#    bucket = storage_client.bucket('aion_data')        
#    bucket.blob('prediction/'+file_name).upload_from_filename('output_data.csv', content_type='text/csv')        
#    return data        

        
def is_file_name_url(file_name):        
    supported_urls_starts_with = ('gs://','https://','http://')        
    return file_name.startswith(supported_urls_starts_with)        

        
class logger():        
    #setup the logger        
    def __init__(self, log_file, mode='w', logger_name=None):        
        logging.basicConfig(filename=log_file, filemode=mode, format='%(asctime)s %(name)s- %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')        
        self.log = logging.getLogger(logger_name)        
        
    #get logger        
    def getLogger(self):        
        return self.log        
        
    def info(self, msg):        
        self.log.info(msg)        
        
    def error(self, msg, exc_info=False):        
        self.log.error(msg,exc_info)        
        
    # format and log dataframe        
    def log_dataframe(self, df, rows=2, msg=None):        
        buffer = io.StringIO()        
        df.info(buf=buffer)        
        log_text = 'Data frame{}'.format(' after ' + msg + ':' if msg else ':')        
        log_text += '\n\t'+str(df.head(rows)).replace('\n','\n\t')        
        log_text += ('\n\t' + buffer.getvalue().replace('\n','\n\t'))        
        self.log.info(log_text)        
