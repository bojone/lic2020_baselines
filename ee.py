#! -*- coding: utf-8 -*-
# 百度LIC2020的事件抽取赛道，非官方baseline
# 直接用RoBERTa+CRF
# 在第一期测试集上能达到0.78的F1，优于官方baseline

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs

# 基本信息
maxlen = 128
epochs = 20
batch_size = 32
learning_rate = 2e-5
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

# bert配置
config_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
                    value = (event['event_type'], argument['role'])
                    arguments[key] = value
            D.append((l['text'], arguments))
    return D


# 读取数据
train_data = load_data('/root/baidu/datasets/ee/train_data/train.json')
valid_data = load_data('/root/baidu/datasets/ee/dev_data/dev.json')

# 读取schema
with open('/root/baidu/datasets/ee/event_schema/event_schema.json') as f:
    id2label, label2id, n = {}, {}, 0
    for l in f:
        l = json.loads(l)
        for role in l['role_list']:
            key = (l['event_type'], role['role'])
            id2label[n] = key
            label2id[key] = n
            n += 1
    num_labels = len(id2label) * 2 + 1

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            labels = [0] * len(token_ids)
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[argument[1]] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output = Dense(num_labels)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text):
    """arguments抽取函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }


def evaluate(data):
    """评测函数（跟官方评测结果不一定相同，但很接近）
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for text, arguments in tqdm(data):
        inv_arguments = {v: k for k, v in arguments.items()}
        pred_arguments = extract_arguments(text)
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)
        Z += len(inv_arguments)
        for k, v in pred_inv_arguments.items():
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = pylcs.lcs(v, inv_arguments[k])
                X += 2. * l / (len(v) + len(inv_arguments[k]))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            arguments = extract_arguments(l['text'])
            event_list = []
            for k, v in arguments.items():
                event_list.append({
                    'event_type': v[0],
                    'arguments': [{
                        'role': v[1],
                        'argument': k
                    }]
                })
            l['event_list'] = event_list
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/baidu/datasets/ee/test1_data/test1.json', 'ee_pred.json')
