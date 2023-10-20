'''
*
* =============================================================================
* COPYRIGHT NOTICE
* =============================================================================
*  @ Copyright HCL Technologies Ltd. 2021, 2022,2023
* Proprietary and confidential. All information contained herein is, and
* remains the property of HCL Technologies Limited. Copying or reproducing the
* contents of this file, via any medium is strictly prohibited unless prior
* written permission is obtained from HCL Technologies Limited.
*
'''

import openai
import tiktoken
import numpy as np
import pandas as pd
from pathlib import Path
from openai.embeddings_utils import get_embedding
from sklearn.base import BaseEstimator, TransformerMixin

class embedding(BaseEstimator, TransformerMixin):

    def __init__(self, embedding_engine='Text-Embedding', embedding_ctx_size=8191, encoding_method='cl100k_base'):
        self.embedding_engine = embedding_engine
        self.embedding_ctx_size = embedding_ctx_size
        self.encoding_method = encoding_method
        self.number_of_features = 1536
        
    def fit(self,X,y=None):
        return self

    def transform(self, X):
        setup_openai()
        
        X = map(lambda text: self.len_safe_get_embedding( text), X)
        return list(X)

    def split_large_text(self, large_text):
        encoding = tiktoken.get_encoding( self.encoding_method)
        tokenized_text = encoding.encode(large_text)

        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokenized_text:
            current_chunk.append(token)
            current_length += 1

            if current_length >= self.embedding_ctx_size:
                chunks.append(encoding.decode(current_chunk).rstrip(' .,;'))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(encoding.decode(current_chunk).rstrip(' .,;'))
     
        return chunks
            
    def len_safe_get_embedding(self, text):
        chunk_embeddings = []
        chunk_lens = []
        for chunk in self.split_large_text(text):
            chunk_embeddings.append( get_embedding(chunk, engine=self.embedding_engine))
            chunk_lens.append(len(chunk))

        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=None)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
        return chunk_embeddings

    def get_feature_names_out(self):
        return [str(x) for x in range(self.number_of_features)]

    def get_feature_names(self):
        return self.get_feature_names_out()

"""
Open AI initialization has to be done separately as follows:
    1. During training read the parameters from user 
        a. from config 
        b. SQLite database
        c. From Json file
"""
class setup_openai():

    def __init__( self, config=None):
        param_keys = ['api_type','api_key','api_base','api_version']
        if isinstance(config, dict):
            valid_params = {x:y for x,y in config.items() if x in param_keys}
            self._update_params(valid_params)    
        elif self._is_sqlite():
            self._update_params( self._get_cred_from_sqlite())
        elif ((Path(__file__).parent.parent/'etc')/'openai.json').exists():
            with open(((Path(__file__).parent.parent/'etc')/'openai.json'), 'r') as f:
                import json
                params = json.load(f)
                valid_params = {x:y for x,y in params.items() if x in param_keys}
                self._update_params(valid_params)
        else:
            raise ValueError('Open AI credentials are not provided.')

    def _is_sqlite(self):
        try:
            from AION.appbe.sqliteUtility import sqlite_db
            from AION.appbe.dataPath import DATA_DIR
            db_dir = Path(DATA_DIR)/'sqlite'
            db_file = 'config.db'
            if (db_dir/db_file).exists():
                sqlite_obj = sqlite_db(db_dir,db_file)
                if sqlite_obj.table_exists('openai'):
                    return True
            return False            
        except:
            return False

    def _get_cred_from_sqlite(self):
        from AION.appbe.sqliteUtility import sqlite_db
        from AION.appbe.dataPath import DATA_DIR
        db_dir = Path(DATA_DIR)/'sqlite'
        db_file = 'config.db'
        sqlite_obj = sqlite_db(db_dir,db_file)
        data = sqlite_obj.read_data('openai')[0]
        param_keys = ['api_type','api_key','api_base','api_version']
        return dict((x,y) for x,y in zip(param_keys,data))

    def _update_params(self, valid_params):
        for key, value in valid_params.items():
            if key == 'api_type':
                openai.api_type = value
            elif key == 'api_key':
                openai.api_key = value
            elif key == 'api_base':
                openai.api_base = value
            elif key == 'api_version':
                openai.api_version = value
