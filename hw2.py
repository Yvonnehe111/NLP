import os
import jieba
import numpy as np
from gensim import corpora, models
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

# ================= 参数配置 =================
CORPUS_DIR = r"C:\jyxstxtqj_downcc.com"          # 小说存放目录
K_VALUES = [20, 100, 500, 1000, 3000]  # 文本长度
T_VALUES = [5, 10, 20, 50, 100]     # 主题数量
UNIT_TYPES = ['word', 'char']  # 处理单元
N_SAMPLES = 1000               # 总样本量
N_SPLITS = 10                  # 交叉验证次数

stopwords = [
    # 高频虚词
    "的", "了", "在", "是", "和", "就", "都", "要", "也", "这",
    "有", "或", "及", "等", "与", "而", "但", "又", "并", "且","\u3000",
    
    # 人称代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "自己",
    
    # 时间/数量词
    "年", "月", "日", "时", "分", "秒", "个", "些", "种", "许多",
    
    # 介词/连词
    "对", "从", "向", "以", "为", "关于", "根据", "按照", "通过",
    
    # 通用动词（无主题区分性）
    "进行", "开始", "需要", "可以", "可能", "应该", "必须", "能够",
    
    # 标点符号（根据处理需求可选）
     "，", "。", "！", "？", "：", "；", "“", "”", "（", "）",
    
    # 网络/论坛高频冗余词‌:ml-citation{ref="3,7" data="citationList"}
    "请", "看", "阅读", "点击", "回复", "发表", "分享", "谢谢",
    "请问", "如何", "怎么", "为什么", "有没有", "有没有人","那","呢","道","到","来"
    ,"本书", "来自", "www", "cr173", "com", "免费","txt","小说","下载站","更新","电子书"
]

# ================= 数据加载 =================
def load_novels():
    """加载所有小说文本"""
    novels = {}
    for fname in os.listdir(CORPUS_DIR):
        if fname.endswith('.txt'):
            with open(os.path.join(CORPUS_DIR, fname), 'r', encoding='gbk', errors="replace") as f:
                novels[fname] = f.read()
    return novels

# ================= 文本采样 =================
def sample_chunks(novels, k, unit='word'):
    """均匀采样文本块"""
    samples, labels = [], []
    samples_per_novel = N_SAMPLES // len(novels)
    
    for novel_id, (fname, text) in enumerate(novels.items()):
        # 文本预处理
        if unit == 'word':
            tokens =  [w for w in jieba.cut(text) if w not in stopwords] 
            # 分词处理
        else:
            tokens = list(text)  # 按字符处理
        
        # 有效采样位置
        available_pos = len(tokens) - k
        if available_pos <= 0:
            continue
            
        # 均匀采样
        step = max(1, available_pos // samples_per_novel)
        for pos in range(0, available_pos, step):
            chunk = tokens[pos:pos+k]
            samples.append(chunk)
            labels.append(novel_id)
            if len(samples) >= N_SAMPLES:
                return samples[:N_SAMPLES], labels[:N_SAMPLES]
    
    return samples, labels

# ================= LDA建模 =================
def get_lda_features(docs, t):
    """获取LDA特征表示"""
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda = models.LdaModel(corpus, num_topics=t, passes=2)
    
    # 转换为主题分布
    features = np.zeros((len(docs), t))
    for i, doc in enumerate(corpus):
        topics = lda.get_document_topics(doc, minimum_probability=0)
        features[i] = [prob for _, prob in topics]
    return features

# ================= 实验流程 =================
def run_experiment():
    novels = load_novels()
    results = []
    
    for k in K_VALUES:
        print(f"\nProcessing K={k}".center(50, '='))
        
        for unit in UNIT_TYPES:
            # 采样文本块
            samples, labels = sample_chunks(novels, k, unit)
            
            # 遍历不同主题数
            for t in T_VALUES:
                # 获取LDA特征
                X = get_lda_features(samples, t)
                y = np.array(labels)
                
                # 交叉验证
                rs = ShuffleSplit(n_splits=N_SPLITS, test_size=100, random_state=42)
                accuracies = []
                
                for train_idx, test_idx in rs.split(X):
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X[train_idx], y[train_idx])
                    pred = clf.predict(X[test_idx])
                    accuracies.append(accuracy_score(y[test_idx], pred))
                
                # 记录结果
                avg_acc = np.mean(accuracies)
                results.append({
                    'K': k,
                    'Unit': unit,
                    'T': t,
                    'Accuracy': avg_acc
                })
                print(f"K={k} | Unit={unit} | T={t} | Acc: {avg_acc:.3f}")
    
    # 结果分析
    df = pd.DataFrame(results)
    return df

# ================= 结果可视化 =================
def analyze_results(df):
    # 问题1: 主题数量影响
    pivot_t = df.pivot_table(index='T', columns='Unit', values='Accuracy')
    pivot_t.plot(title='Accuracy by Topic Numbers')
    
    # 问题2: 词vs字对比
    pivot_unit = df.groupby(['K', 'Unit'])['Accuracy'].mean().unstack()
    pivot_unit.plot(kind='bar', title='Word vs Character Comparison')
    
    # 问题3: 文本长度影响
    pivot_k = df.groupby(['K', 'T'])['Accuracy'].mean().unstack()
    pivot_k.plot(kind='line', title='Performance by Text Length')

# ================= 主程序 =================
if __name__ == "__main__":
    result_df = run_experiment()
    result_df.to_csv('experiment_results.csv', index=False)
    analyze_results(result_df)
