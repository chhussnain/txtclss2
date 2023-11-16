import re
from gensim.models.keyedvectors import KeyedVectors
from transformers import pipeline
import pickle
import numpy as np
import pandas as pd

w2v = KeyedVectors.load('models/word2vec')
w2v_vocab = set(sorted(w2v.index_to_key))
model = pickle.load(open('models/w2v_ovr_svc.sav', 'rb'))
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", framework='pt'
                     )

labels = [
    'communication', 'waiting time',
       'information', 'user interface',
       'facilities', 'location', 'price'
]

sample_file = pd.read_csv('sample.csv').to_csv(index=False).encode('utf-8')

print('utils imported!')

def get_sentiment_label_facebook(list_of_sent_dicts):
    if list_of_sent_dicts['labels'][0] == 'negative':
        return 'negative'
    else:
        return 'positive'

def get_single_prediction(text):
    
    # manipulate data into a format that we pass to our model
    text = text.lower() #lower case
    text = re.sub('[^0-9a-zA-Z\s]', '', text) #remove special char, punctuation

    # Remove OOV words
    text = ' '.join([i for i in text.split() if i in w2v_vocab])
    
    # Vectorise text and store in new dataframe. Sentence vector = average of word vectors
    text_vectors = np.mean([w2v[i] for i in text.split()], axis=0)

    # Make predictions
    results = model.predict_proba(text_vectors.reshape(1,300)).squeeze().round(2)
    pred_prob = pd.DataFrame({'topic': labels, 'probability': results}).sort_values('probability', ascending=True)
    
    # Get sentiment
    sentiment_results = classifier(text, 
                                candidate_labels=['positive', 'negative'], 
                                hypothesis_template='The sentiment of this is {}')
    sentiment_prob = pd.DataFrame({'sentiment': sentiment_results['labels'], 'probability': sentiment_results['scores']})
    
    return (pred_prob, sentiment_prob)

def get_multiple_predictions(csv):
    
    df = pd.read_csv(csv)
    df.columns = ['sequence']

    df['sequence_clean'] = df['sequence'].str.lower() #lower case
    df['sequence_clean'] = df['sequence_clean'].str.strip()
    df['sequence_clean'] = df['sequence_clean'].str.replace('[^0-9a-zA-Z\s]','') #remove special char, punctuation

    # Remove OOV words
    df['sequence_clean'] = df['sequence_clean'].apply(lambda x: ' '.join([i for i in x.split() if i in w2v_vocab]))

    # Remove rows with blank string
    invalid = df[(pd.isna(df['sequence_clean'])) | (df['sequence_clean'] == '')]
    invalid.drop(columns=['sequence_clean'], inplace=True)
    
    # Drop rows with blank string
    df.dropna(inplace=True)
    df = df[df['sequence_clean'] != ''].reset_index(drop=True)
    
    # Vectorise text and store in new dataframe. Sentence vector = average of word vectors
    series_text_vectors = pd.DataFrame(df['sequence_clean'].apply(lambda x: np.mean([w2v[i] for i in x.split()], axis=0)).values.tolist())
    
    # Get predictions
    pred_results = pd.DataFrame(model.predict(series_text_vectors), columns = labels)
    
    # Join back to original sequence
    final_results = df.join(pred_results)
    final_results['others'] = final_results[labels].max(axis=1)
    final_results['others'] = final_results['others'].apply(lambda x: 1 if x == 0 else 0)
    
    # Get sentiment labels
    final_results['sentiment'] = final_results['sequence_clean'].apply(lambda x: get_sentiment_label_facebook(classifier(x, 
                                                            candidate_labels=['positive', 'negative'], 
                                                            hypothesis_template='The sentiment of this is {}'))
                                                                )
    
    final_results.drop(columns=['sequence_clean'], inplace=True)
    
    # Append invalid rows
    if len(invalid) == 0:
        return final_results.to_csv(index=False).encode('utf-8')
    else:
        return pd.concat([final_results, invalid]).reset_index(drop=True).to_csv(index=False).encode('utf-8')