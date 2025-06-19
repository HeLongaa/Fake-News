import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flask import Flask, render_template_string, request, jsonify
import joblib
import glob
import config
import pickle
import jieba
from utils.tokenizer import tokenizer as jieba_tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from models.bert_tokenization import BasicTokenizer
import numpy as np

# 可根据实际情况扩展
from keras.models import load_model as keras_load_model
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models')

MODEL_EXTS = ['*.model', '*.pkl', '*.h5']

# 加载特征工程器
TFIDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'train.text.word.tfidf.pkl')
SVD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'train.text.svd.pkl')
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'vocab.txt')
ONEHOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'train.text.onehot.pkl')

# 经典模型特征工程
with open(TFIDF_PATH, 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open(SVD_PATH, 'rb') as f:
    svd_matrix = pickle.load(f)

# OneHot Tokenizer
def load_tokenizer(vocab_path):
    tokenizer = KerasTokenizer()
    word_index = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            k, v = line.split('\t')
            word_index[k] = int(v)
    tokenizer.word_index = word_index
    return tokenizer

tokenizer = load_tokenizer(TOKENIZER_PATH)

# BERT分词器
bert_tokenizer = BasicTokenizer()

# 统一特征处理
# model_type: 'classic', 'deep', 'bert'
def text_to_features(text, model_type):
    if model_type == 'classic':
        # jieba分词+unigram
        seg = jieba_tokenizer(text)
        unigram_str = ' '.join(seg)
        # TFIDF向量化（用fit时的vocab）
        # 这里直接用训练集的tfidf矩阵的vocab
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vec = TfidfVectorizer(vocabulary=tfidf_matrix.vocabulary_)
        tfidf_vec._validate_vocabulary()
        tfidf = tfidf_vec.transform([unigram_str])
        # SVD降维
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=svd_matrix.shape[1])
        svd.fit(tfidf_matrix)
        svd_feature = svd.transform(tfidf)
        # 只返回svd特征，实际可按训练时拼接其他特征
        return svd_feature
    elif model_type == 'deep':
        # jieba分词+unigram
        seg = jieba_tokenizer(text)
        unigram_str = ' '.join(seg)
        seq = tokenizer.texts_to_sequences([unigram_str])
        padded = pad_sequences(seq, maxlen=300)
        return padded
    elif model_type == 'bert':
        tokens = bert_tokenizer.tokenize(text)
        return tokens
    else:
        raise ValueError('未知模型类型')

def list_models():
    files = []
    for ext in MODEL_EXTS:
        files.extend(glob.glob(os.path.join(MODEL_DIR, ext)))
    return [os.path.basename(f) for f in files]

def load_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    if model_name.endswith('.model'):
        # 先尝试CatBoost
        try:
            model = CatBoostClassifier()
            model.load_model(path)
            return model, 'catboost'
        except Exception:
            pass
        # 再尝试xgboost/sklearn
        try:
            model = joblib.load(path)
            return model, 'sklearn'
        except Exception:
            pass
        # 再尝试keras
        try:
            model = keras_load_model(path)
            return model, 'keras'
        except Exception:
            pass
    elif model_name.endswith('.h5'):
        model = keras_load_model(path)
        return model, 'keras'
    elif model_name.endswith('.pkl'):
        model = joblib.load(path)
        return model, 'sklearn'
    # TODO: bert/kashgari等
    raise ValueError('不支持的模型类型或加载失败')

HTML = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>假新闻检测在线演示</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: auto; }
        textarea { width: 100%; height: 120px; }
        select, button { padding: 8px; margin-top: 10px; }
        .result { margin-top: 20px; font-size: 1.2em; color: #333; }
    </style>
</head>
<body>
<div class="container">
    <h2>假新闻检测在线演示</h2>
    <form id="predict-form">
        <label for="model">选择模型：</label>
        <select id="model" name="model">
            {% for m in models %}
            <option value="{{m}}">{{m}}</option>
            {% endfor %}
        </select><br>
        <label for="text">输入待检测文本：</label><br>
        <textarea id="text" name="text" required></textarea><br>
        <button type="submit">提交检测</button>
    </form>
    <div class="result" id="result"></div>
</div>
<script>
    document.getElementById('predict-form').onsubmit = async function(e) {
        e.preventDefault();
        let text = document.getElementById('text').value;
        let model = document.getElementById('model').value;
        let resDiv = document.getElementById('result');
        resDiv.innerHTML = '正在预测...';
        let resp = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text, model})
        });
        let data = await resp.json();
        if(data.success) {
            resDiv.innerHTML = '<b>预测结果：</b> ' + data.result;
        } else {
            resDiv.innerHTML = '<span style="color:red">' + data.error + '</span>';
        }
    }
</script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    models = list_models()
    return render_template_string(HTML, models=models)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    model_name = data.get('model')
    if not text or not model_name:
        return jsonify({'success': False, 'error': '请输入文本并选择模型'})
    try:
        model, mtype = load_model(model_name)
        # 自动判断模型类型
        if 'bert' in model_name.lower():
            model_type = 'bert'
        elif any(x in model_name.lower() for x in ['textcnn', 'rnn', 'dpcnn']):
            model_type = 'deep'
        else:
            model_type = 'classic'
        X = text_to_features(text, model_type)
        if mtype in ['sklearn', 'xgboost', 'catboost']:
            pred = model.predict(X)
            result = str(pred[0])
        elif mtype == 'keras':
            pred = model.predict(X)
            result = str(np.argmax(pred, axis=1)[0])
        else:
            result = '暂不支持该模型类型的在线预测'
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 