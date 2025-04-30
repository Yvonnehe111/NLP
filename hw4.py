import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 数据加载和预处理
def load_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='gbk', errors="replace") as f:
                texts.append(f.read())
                
                  #加了break就只读取第一个文件
    return texts

# 数据预处理：去除特殊字符，清洗文本
def preprocess_text(texts):
    all_text = ' '.join(texts)
    all_text = re.sub(r'\s+', ' ', all_text)  # 去除多余空格
   # all_text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]', '', all_text)  # 只保留中文和数字
    return all_text

# 2. 加载并预处理数据
folder_path = r"C:\jyxstxtqj_downcc.com"  # 替换为存放金庸小说的文件夹路径
texts = load_texts(folder_path)
cleaned_text = preprocess_text(texts)

# 3. 令牌化文本
tokenizer = Tokenizer(char_level=True)  # 按字符级别进行tokenizer
tokenizer.fit_on_texts([cleaned_text])
total_chars = len(tokenizer.word_index) + 1  # 字符表的大小

# 4. 将文本转换为整数序列
sequence_length = 100  # 每个输入序列的长度
sequences = []
for i in range(len(cleaned_text) - sequence_length):
    sequences.append(cleaned_text[i:i + sequence_length + 1])

# 将序列转换为整数
sequence_length = 100
sequences = [cleaned_text[i:i + sequence_length + 1] for i in range(len(cleaned_text) - sequence_length)]
sequences = tokenizer.texts_to_sequences(sequences)
X = np.array([seq[:-1] for seq in sequences])
y = np.array([seq[-1] for seq in sequences])

# 填充序列
X = pad_sequences(X, maxlen=sequence_length, padding='pre')

# 5. 构建LSTM模型
def build_lstm_model(sequence_length, total_chars):
    model_lstm = Sequential()
    model_lstm.add(Embedding(total_chars, 64, input_length=sequence_length))
    model_lstm.add(LSTM(128, return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(128))
    model_lstm.add(Dense(total_chars, activation='softmax'))
    model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adamax')
    

    return model_lstm

# 创建LSTM模型
model_lstm = build_lstm_model(sequence_length, total_chars)

# 6. 训练LSTM模型
model_lstm.fit(X, y, batch_size=64, epochs=2)

# 7. 构建Transformer模型
def transformer_model(sequence_length, total_chars, num_heads=8, ff_dim=512):
    inputs = Input(shape=(sequence_length,))
    embedding = Embedding(total_chars, 128)(inputs)
    
    # Transformer Encoder Layer
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=128)(embedding, embedding)
    attention = Dropout(0.2)(attention)
    attention = LayerNormalization()(embedding + attention)
    
    ff = Dense(ff_dim, activation='relu')(attention)
    ff = Dense(embedding.shape[-1])(ff)
    ff = Dropout(0.2)(ff)
    output = LayerNormalization()(attention + ff)
    
    # 输出层
    output = Dense(total_chars, activation='softmax')(output)
    
    model = Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

# 创建Transformer模型
#model_transformer = transformer_model(sequence_length, total_chars)

# 8. 训练Transformer模型
#model_transformer.fit(X, y, batch_size=64, epochs=10)

# 9. 文本生成函数
def generate_text(model, tokenizer, seed_text, sequence_length, num_chars):
    result = seed_text
    for _ in range(num_chars):
        # 转换为整数序列
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        padded = pad_sequences([encoded], maxlen=sequence_length, padding='pre')
        
        # 预测下一个字符
        pred = model.predict(padded, verbose=0)
        next_char_index = np.argmax(pred)
        next_char = tokenizer.index_word[next_char_index]
        
        # 更新种子文本
        seed_text += next_char
        seed_text = seed_text[1:]
        result += next_char
    return result

# 10. 示例：生成文本
seed_text = "少林寺"  # 你可以更改种子文本
generated_text_lstm = generate_text(model_lstm, tokenizer, seed_text, sequence_length, 100)
generated_text_transformer = generate_text(model_transformer, tokenizer, seed_text, sequence_length, 100)

# 打印生成的文本
print("LSTM 生成的文本：", generated_text_lstm)
print("Transformer 生成的文本：", generated_text_transformer)