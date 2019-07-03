"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import pandas as pd
import numpy as np
import jieba
import multiprocessing

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from tensorflow.keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

np.random.seed(1440)
import sys
sys.setrecursionlimit(1000000)
import yaml  # pip install PyYaml yaml是一种比xml和json更轻的文件格式，也更简单更强大，它可以通过缩进来表示结构

# 设置超参数
cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10  # 所有频数超过10的词语
window_size = 7
n_epoch = 4
input_length = 100
maxlen = 100
batch_size = 32

# 加载数据
def loadfile():
    neg=pd.read_csv('./data/neg.csv',header=None,index_col=None,encoding='utf-8',sep=None, error_bad_lines=False)
    pos=pd.read_csv('./data/pos.csv',header=None,index_col=None,encoding='utf-8',sep=None, error_bad_lines=False)
    neu=pd.read_csv('./data/neutral.csv', header=None, index_col=None,encoding='utf-8',sep=None, error_bad_lines=False)

    combined = np.concatenate((pos[0], neu[0], neg[0]))
    y = np.concatenate((
        np.ones(len(pos), dtype=int),
        np.zeros(len(neu), dtype=int),
        -1*np.ones(len(neg),dtype=int)))
    return combined,y


# 对句子经行分词，并去掉换行符
def tokenizer(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


# 创建词向量字典
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}     #所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()} #所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen) #每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引（利用word2vec模型）
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined) # input: list
    model.train(combined,total_examples=model.corpus_count,epochs=model.epochs)
    model.save('./store/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return  index_dict, word_vectors,combined

# 获取词向量模型返回的数据进行数据集分割
def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = to_categorical(y_train,num_classes=3)
    y_test = to_categorical(y_test,num_classes=3)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


# 定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/layers/Embedding
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/layers/LSTM
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=3
    model.add(Activation('softmax'))

    # 画出网络结构图
    plot_model(model,to_file = 'img/model.png',show_shapes=True, show_layer_names=True)

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Train...") # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('./store/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('./store/lstm.h5')
    print('Test score:', score)


# 预测
def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('./store/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined


def lstm_predict(string):
    print('loading model......')
    with open('./store/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('./store/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    result=model.predict_classes(data)
    return result
    # print(result)
    # if result[0]==2:
    #     print(string,' positive')
    # elif result[0]==1:
    #     print(string,' neural')
    # else:
    #     print(string,' negative')


if __name__ == "__main__":
    #训练模型，并保存
    # print('Loading Data...')
    # combined,y=loadfile()
    # print(len(combined),len(y))
    #
    # print('Tokenising...')
    # combined = tokenizer(combined)
    #
    # print('Training a Word2vec model...')
    # index_dict, word_vectors,combined=word2vec_train(combined)
    #
    # print('Setting up Arrays for Keras Embedding Layer...')
    # n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    # print("x_train.shape and y_train.shape:")
    # print(x_train.shape,y_train.shape)
    #
    # train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


    # 预测单条句子
    # string = "离老公家很近，地点交通方便，是现在比较流行的快捷酒店，选的地点不是闹市区，不是很吵，这点满好，里面环境还一般，一般水平。"
    # lstm_predict(line)[0]

    # 预测
    with open('./data/predict_test.txt',mode='r',encoding='utf-8') as fr:
        with open('./out/predict_out_lstm.txt',mode='w',encoding='utf-8') as fw:
            for line in fr.readlines():
                fw.write(str(lstm_predict(line)[0]) + '\n')