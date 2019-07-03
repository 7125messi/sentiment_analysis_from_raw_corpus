# _*_ coding:utf-8 _*_

'''
使用传统的机器学习的方法进行文本情感分析
'''

import codecs
import jieba
import numpy as np

# from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


# 对每个句子的所有词向量取均值
# text 需要是切完词的 词列表
# size 一般是词向量的维度
# word_vector_model: 训练好的词向量模型 （一般使用 gensim 中的 WordVector 进行词向量训练）或者是直接加载训练好的模型
def buildWordVector(text, size, word_vector_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += word_vector_model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vecs(x_train, x_test, n_dim):
    '''
    x_train: 训练集
    x_test:  测试集
    n_dim:   训练词向量的维度
    '''
    n_dim = n_dim
    # 初始化模型和生成词汇表
    all_text = x_train + x_test
    text_w2v = Word2Vec(size=n_dim, min_count=5, workers=1)
    text_w2v.build_vocab(all_text)
    text_w2v.train(all_text, total_examples=text_w2v.corpus_count, epochs=5)

    # 分别得到训练集和测试集文本的词向量合集，这个数据集就很大了
    train_vecs = np.concatenate([buildWordVector(text, n_dim, text_w2v) for text in x_train])
    # 将训练集的词向量进行保存
    np.save(storedpaths + "train_vecs.npy", train_vecs)
    print("训练集数据的词向量维度：{}".format(train_vecs.shape))

    test_vecs = np.concatenate([buildWordVector(text, n_dim, text_w2v) for text in x_test])
    # 将测试集的词向量进行保存
    np.save(storedpaths + "test_vecs.npy", test_vecs)
    print("测试集数据的词向量维度：{}".format(test_vecs.shape))

    # 保存词向量模型，供后面使用预训练的词向量模型预测使用
    text_w2v.save(storedpaths + "w2v_model.pkl")


# 加载向量化的文本和标签（利用get_train_vecs函数里保存的训练集词向量和测试集词向量）
def get_data():
    train_vecs = np.load(storedpaths + 'train_vecs.npy')
    y_train = np.load(storedpaths + 'y_train.npy')
    test_vecs = np.load(storedpaths + 'test_vecs.npy')
    y_test = np.load(storedpaths + 'y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


# 训练svm模型并保存模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练朴素贝叶斯模型并保存模型
def NB_train(train_vecs, y_train, test_vecs, y_test):
    gnb = GaussianNB()
    gnb.fit(train_vecs, y_train)
    joblib.dump(gnb, storedpaths + 'model_gnb.pkl')
    test_scores = gnb.score(test_vecs, y_test)
    return test_scores


# 训练决策树模型并保存模型
def decision_tree(train_vecs, y_train, test_vecs, y_test):
    clf = DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=0)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_dtree.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练随机森林算法并保存模型
def random_forest(train_vecs, y_train, test_vecs, y_test):
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=2, n_jobs=1, random_state=0)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_randomforest.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练 ExtraTreesClassifier 分类算法并保存模型
def extract_tree(train_vecs, y_train, test_vecs, y_test):
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=10, min_samples_split=2, n_jobs=1, random_state=0)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_extracttree.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练 GBDT 分类算法并保存模型
def gbdt_classifier(train_vecs, y_train, test_vecs, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_gbdt.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练近邻分类算法并保存模型
def nn_classifier(n_neighbors, train_vecs, y_train, test_vecs, y_test):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_nn.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练 LogisticRegression 分类算法并保存模型
def LR_classifier(train_vecs, y_train, test_vecs, y_test):
    clf = LogisticRegression(C=50. / len(y_train), multi_class='multinomial',penalty='l1', solver='saga', tol=0.1)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_lr.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练 随机梯度下降 分类算法并保存模型
def SGD_classifier(train_vecs, y_train, test_vecs, y_test):
    clf = SGDClassifier(alpha=0.001, max_iter=100)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_sgd.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 训练多层感知机分类算法并保存模型
def MP_classifier(train_vecs, y_train, test_vecs, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, storedpaths + 'model_mp.pkl')
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# 得到待预测单个句子的词向量
# 预先进行分词操作
def get_predict_vecs(string, n_dim, w2v_model_path):
    '''
    string: 输入的句子
    n_dim: 词向量维度
    w2v_model_path: 預训练词向量的模型路径
    '''
    n_dim = n_dim
    text_w2v = Word2Vec.load(w2v_model_path)
    words = [i for i in jieba.cut(string, cut_all=False)]
    train_vecs = buildWordVector(words, n_dim, text_w2v)

    return train_vecs


# 调用训练模型进行预测
def svm_predict(string, trainmodelpath):
    words_vecs = get_predict_vecs(string, n_dim, w2v_model_path)
    clf = joblib.load(trainmodelpath)
    result = clf.predict(words_vecs)

    return result

if __name__ == "__main__":
    """
    先训练模型、保存模型再预测模型
    """
    # datapaths = "./data/"
    # storedpaths = "./store/"
    #
    # positive_data = []
    # y_positive = []
    #
    # neutral_data = []
    # y_neutral = []
    #
    # negative_data = []
    # y_negative = []
    #
    # print("#------------------------------------------------------#")
    # print("加载数据集")
    # with codecs.open(datapaths + "pos.csv", "r", "utf-8") as f1, \
    #         codecs.open(datapaths + "neutral.csv", "r", "utf-8") as f2, \
    #         codecs.open(datapaths + "neg.csv", "r", "utf-8") as f3:
    #     for line in f1:
    #         positive_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
    #         # y_positive.append([1,0,0])
    #         y_positive.append([0])
    #     for line in f2:
    #         neutral_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
    #         # y_neutral.append([0,1,0])
    #         y_neutral.append([1])
    #     for line in f3:
    #         negative_data.append(" ".join(i for i in jieba.lcut(line.strip(), cut_all=False)))
    #         # y_negative.append([0,0,1])
    #         y_negative.append([2])
    #
    # print("positive data:{}".format(len(positive_data)))
    # print("neutral data:{}".format(len(neutral_data)))
    # print("negative data:{}".format(len(negative_data)))
    #
    # x_text = positive_data + neutral_data + negative_data
    # y_label = y_positive + y_neutral + y_negative
    # print("#------------------------------------------------------#")
    # print("\n")
    #
    # # 数据集混洗
    # shuffle_indices = np.random.permutation(np.arange(len(y_label)))
    # train_test_percent = 0.2
    #
    # x_train = []
    # x_test = []
    #
    # y_train = []
    # y_test = []
    #
    # for i in shuffle_indices[:-(int(len(shuffle_indices) * train_test_percent))]:
    #     x_train.append(x_text[i])
    #     y_train.append(y_label[i])
    #
    # for i in shuffle_indices[-(int(len(shuffle_indices) * train_test_percent)):]:
    #     x_test.append(x_text[i])
    #     y_test.append(y_label[i])
    #
    # x_train_pos = 0
    # x_train_neu = 0
    # x_train_neg = 0
    #
    # x_test_pos = 0
    # x_test_neu = 0
    # x_test_neg = 0
    #
    # for i in y_train:
    #     if i[0] == 0:
    #         x_train_pos += 1
    #     elif i[0] == 1:
    #         x_train_neu += 1
    #     else:
    #         x_train_neg += 1
    #
    # for i in y_test:
    #     if i[0] == 0:
    #         x_test_pos += 1
    #     elif i[0] == 1:
    #         x_test_neu += 1
    #     else:
    #         x_test_neg += 1
    #
    # print("#------------------------------------------------------#")
    #
    # print("保存标签数据")   # 这是经过分词处理好后的数据，可以直接使用
    # np.save(storedpaths + "y_train.npy", np.array(y_train))
    # np.save(storedpaths + "y_test.npy", np.array(y_test))
    #
    # print("训练集总数：{}".format(len(x_train)))
    # print("训练集正样本：{}".format(x_train_pos))
    # print("训练集中性样本：{}".format(x_train_neu))
    # print("训练集负样本：{}".format(x_train_neg))
    #
    # print("测试集总数：{}".format(len(x_test)))
    # print("测试集正样本：{}".format(x_test_pos))
    # print("测试集中性样本：{}".format(x_test_neu))
    # print("测试集负样本：{}".format(x_test_neg))
    #
    # print("#------------------------------------------------------#")
    # print("\n")
    #
    # # Train model
    # n_dim = 100 # 默认是100维
    # n_neighbors = 10
    #
    # # 获得词向量模型
    # get_train_vecs(x_train, x_test, n_dim)
    #
    # # 获得向量化的文本和标签
    # train_vecs, y_train, test_vecs, y_test = get_data()

    # 训练模型并在测试集上验证分数
    # test_scores = svm_train(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("SVM测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")

    # test_scores = NB_train(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("NB测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = nn_classifier(n_neighbors, train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("NN测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = LR_classifier(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("LR测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = SGD_classifier(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("SGD测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = decision_tree(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("TREE测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = random_forest(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("Random_Forest测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = extract_tree(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("Extract_Tree测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = gbdt_classifier(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("GBDT_Tree测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")
    #
    # test_scores = MP_classifier(train_vecs, y_train, test_vecs, y_test)
    # print("#----------------------------------------#")
    # print("MP测试集测试得分：{}".format(test_scores))
    # print("#----------------------------------------#")

    # 预测单个句子
    # string = '最大的困扰是穴位有点把不准，不过我牢记中里先生离穴不离经的教诲。'
    # trainmodelpath = './store/model.pkl'
    # w2v_model_path = './store/w2v_model.pkl'
    # n_dim = 300
    # # n_dim = 100
    # print(svm_predict(string, trainmodelpath))

    # 预测
    trainmodelpath = './store/model.pkl'
    w2v_model_path = './store/w2v_model.pkl'
    n_dim = 100
    with open('./data/predict_test.txt',mode='r',encoding='utf-8') as fr:
        with open('./out/predict_out_svm.txt',mode='w',encoding='utf-8') as fw:
            for line in fr.readlines():
                fw.write(str(svm_predict(line,trainmodelpath)[0]) + '\n')