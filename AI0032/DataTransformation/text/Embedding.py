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
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# Private function
def unitvec(vec):
    return vec / np.linalg.norm(vec)
    
def __word_average(vectors, sent, vector_size,key_to_index):
    """
    Compute average word vector for a single doc/sentence.
    """
    try:
        mean = []
        for word in sent:
            index = key_to_index.get( word, None)
            if index != None:
                mean.append( vectors[index] )
        if len(mean):
            return unitvec(np.array(mean).mean(axis=0))
        return np.zeros(vector_size)
    except:
        raise

# Private function
def __word_average_list(vectors, docs, embed_size,key_to_index):
    """
    Compute average word vector for multiple docs, where docs had been tokenized.
    """
    try:
        return np.vstack([__word_average(vectors, sent, embed_size,key_to_index) for sent in docs])
    except:
        raise

def load_pretrained(path):
    df = pd.read_csv(path, index_col=0,sep=' ',quotechar = ' ' , header=None, skiprows=1)
    return len(df.columns), df
    
def get_model( df:pd.DataFrame):
    index_to_key = {k:v for k,v in enumerate(df.index)}
    key_to_index = {v:k for k,v in enumerate(df.index)}
    df = df.to_numpy()
    return df, index_to_key, key_to_index
        
def extractFeatureUsingPreTrainedModel(inputCorpus, pretrainedModelPath=None, loaded_model=False,key_to_index={}, embed_size=300):
    """
    Extract feature vector from input Corpus using pretrained Vector model(word2vec,fasttext, glove(converted to word2vec format)
    """
    try:
        if inputCorpus is None:           
            return None
        else:
            if not pretrainedModelPath and ((isinstance(loaded_model, pd.DataFrame) and loaded_model.empty) or (not isinstance(loaded_model, pd.DataFrame) and not loaded_model)):
                inputCorpusWordVectors = None
            else:
                if (isinstance(loaded_model, pd.DataFrame) and not loaded_model.empty) or loaded_model:
                    pretrainedModel = loaded_model
                else:
                    embed_size, pretrainedModel = load_pretrained(pretrainedModelPath)
                pretrainedModel, index_to_key,key_to_index = get_model( pretrainedModel)
                if len(pretrainedModel):
                    input_docs_tokens_list = [word_tokenize(inputDoc) for inputDoc in inputCorpus]
                    inputCorpusWordVectors = __word_average_list(pretrainedModel, input_docs_tokens_list,embed_size,key_to_index)
                else:
                    inputCorpusWordVectors = None
            return inputCorpusWordVectors
    except:
        raise  
