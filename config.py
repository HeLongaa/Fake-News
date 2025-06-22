import os.path
import torch
"""
模型配置模块

功能:
    1. 设置模型训练的基本参数(学习率、batch size等)
    2. 配置BERT和ResNet模型的具体结构
    3. 设置模型保存路径和日志路径
    4. 管理训练过程中的设备选择和资源分配
    5. self.bert_name和self.resnet_name决定模型的具体结构
"""
#通过修改self.bert_name和self.resnet_name来决定模型的具体结构
class Config(object):

    def __init__(self):
        '''
        初始化配置参数
        dropout: 随机失活率
        require_improvement: 训练过程中若超过该batch数效果未提升，则提前结束训练
        num_classes: 分类类别数
        num_epochs: 训练的epoch数
        batch_size: mini-batch大小
        pad_size: 每句话处理成的长度
        bert_learning_rate: BERT模型的学习率
        resnet_learning_rate: ResNet模型的学习率
        other_learning_rate: 其他参数的学习率
        frac: 使用数据的比例
        bert_name: BERT模型名称
        bert_fc: BERT全连接层输出维度,bert-base-chinese和chinese-bert-wwm-ext为768，minirbt-h256为256
        resnet_name: ResNet模型名称
        resnet_fc: ResNet全连接层输出维度
        usesloss: 是否使用对比学习
        save_path: 模型保存路径
        log_dir: TensorBoard日志路径
        device: 训练设备
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.3
        self.require_improvement = 2000
        self.num_classes = 2
        self.num_epochs = 20   # epoch数
        self.batch_size =32
        self.pad_size = 128
        self.bert_learning_rate = 1e-5
        self.resnet_learning_rate = 2e-5
        self.other_learning_rate = 2e-5
        self.frac=1

        #['bert_model/bert-base-chinese','bert_model/chinese-bert-wwm-ext','bert_model/minirbt-h256']
        self.bert_name='bert_model/bert-base-chinese'
        self.bert_fc=768
        #['resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']
        self.resnet_name='resnet34'
        self.resnet_fc=self.bert_fc
        self.usesloss=True

        if not os.path.exists('model'):
            os.makedirs('model')
        if self.usesloss:
            self.save_path = 'model/'+'S_'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name+'.ckpt'
            self.log_dir= './log/'+'S_'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name

        else:
            self.save_path = 'model/'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name+'.ckpt'
            self.log_dir= './log/'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name

