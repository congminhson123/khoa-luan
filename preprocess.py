import pandas as pd
import sys
import numpy as np

import operator
from models import Input, AspectOutput
from tqdm import tqdm
import re
from gensim.models import KeyedVectors
import json


def load_aspect_data_mebe(path):
    """

    :param path:
    :return:
    :rtype: (list of models.Input, list of models.AspectOutput)
    """
    inputs = []
    outputs = []
    df = pd.read_csv(path, encoding='utf-8')
    for _, r in df.iterrows():
        t = str(r['text']).strip()
        inputs.append(Input(t))

        labels = list(range(10))
        scores = [0 if r['aspect{}'.format(i)] == 0 else 1 for i in range(1, 8)]
        outputs.append(AspectOutput(labels, scores))

    return inputs, outputs


def preprocess(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    corpus=[]
    for text in tqdm(inputs):
        text = text.text.lower()
        # text=re.sub(r'[^a-z0-9A-Z]'," ",text)
        text=re.sub(r'[0-9]{1}',"#",text)
        text=re.sub(r'[0-9]{2}','##',text)
        text=re.sub(r'[0-9]{3}','###',text)
        text=re.sub(r'[0-9]{4}','####',text)
        text=re.sub(r'[0-9]{5,}','#####',text)
        corpus.append(text)
    return corpus
inputs, outputs = load_aspect_data_mebe('data/raw_data/mebe_shopee.csv')
pre_text = preprocess(inputs)
def get_vocab(corpus):
    vocab={}
    for text in tqdm(corpus):
        for word in text.split():
            try:
                vocab[word]+=1
            except KeyError:
                vocab[word]=1
    vocab=dict(sorted(vocab.items(),reverse=True ,key=lambda item: item[1]))
    return vocab
model_embed = KeyedVectors.load_word2vec_format('../cc.vi.300.vec')
def check_voc(vocab,model):
    embed_words=[]
    out_vocab={}
    total_words=0
    total_text=0
    list_word_embed = list()
    for key, value in model_embed.items():
        list_word_embed.append(key)
    with open('list_word_embed.txt', 'w') as f:
        f.write(json.dumps(list_word_embed))

    for i in tqdm(vocab):
        try:
            i = i.replace("_", " ")
            vec=model[i]
            embed_words.append(vec) # list các vector của những từ xuất hiện cả embed vocab và vocab dữ liệu
            total_words+=vocab[i] # tổng số từ xuất hiện trong dữ liệu mà có trong cả embed vocab, tính cả từ giống nhau
            print("từ có trong embed:"+ i)
        except KeyError: ## những từ không xuất hiện trong embed model
            out_vocab[i]=vocab[i]
            total_text+=vocab[i] # total text bằng chính tổng lượng từ trong dữ liệu
            print("từ không có trong embed" + i)
    print("The {:.2f}% of vocabularies have Covered of corpus".format(100*len(embed_words)/len(vocab)))
    print("The {:.2f}% of total text had coverded ".format((100*total_words/(total_words+total_text))))
    return out_vocab


vocabulary=get_vocab(pre_text)
oov=check_voc(vocabulary,model_embed)
sort_oov=dict(sorted(oov.items(), key=operator.itemgetter(1),reverse=True))
print(dict(list(sort_oov.items())[:50]))
# print(vocabulary)

def get_word_index(vocab):
    word_index=dict((w,i+1) for i,w in enumerate(vocab.keys()))
    return word_index
def fit_one_hot(word_index,corpus):
    sent=[]
    for text in tqdm(corpus):
        li=[]
        for word in text.split():
            try:
                li.append(word_index[word])
            except KeyError:
                li.append(0)
        sent.append(li)
    return sent
# print(len(vocabulary))
word_index=get_word_index(vocabulary)
encode_data = fit_one_hot(word_index, pre_text)

with open('encode_data.txt', 'w') as f:
    f.write(json.dumps(encode_data))

count=0
embedding_mat=np.zeros((len(vocabulary)+1,300)) # tạo ma trận weight của từ dựa trên từ điển embed
for word,i in tqdm(word_index.items()):
    word = word.replace("_", " ")
    try:
        vec=model_embed[word]
        embedding_mat[i]=vec
    except KeyError:
        count+=1
        continue
with open('embedding_mat.txt', 'w') as f:
    f.write(json.dumps(embedding_mat.tolist()))


print("Number of Out of Vocabulary",count)



def preprocess_tiki(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    pass


def preprocess_dulich(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    pass
