from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TraditionalModels:
    """传统机器学习模型"""
    
    @staticmethod
    def train_svm(X_train, y_train, **kwargs):
        """
        训练SVM模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            **kwargs: SVM参数
            
        Returns:
            训练好的SVM模型
        """
        svm = SVC(**kwargs)
        svm.fit(X_train, y_train)
        return svm
    
    @staticmethod
    def train_random_forest(X_train, y_train, **kwargs):
        """
        训练随机森林模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            **kwargs: 随机森林参数
            
        Returns:
            训练好的随机森林模型
        """
        rf = RandomForestClassifier(**kwargs)
        rf.fit(X_train, y_train)
        return rf
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            包含各项评估指标的字典
        """
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
