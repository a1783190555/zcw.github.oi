import cv2
import numpy as np

class FocusEvaluator:
    @staticmethod  
    def variance_method(img):#方差
        # 检查输入图像是否为彩色图像（3通道）
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        
        # 使用拉普拉斯算子计算图像的二阶导数
        laplacian = cv2.Laplacian(img, cv2.CV_32F)
        
        # 平方操作将所有值转为正数，同时放大差异
        return np.mean(laplacian ** 2)

    @staticmethod
    def gradient_method(img):
        """
        梯度法
        原理：使用Sobel算子计算水平和垂直方向的一阶导数，通过梯度值评估清晰度 
        """
        # 检查输入图像是否为彩色图像
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        
        # ksize=3表示使用3x3的Sobel核
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        # 使用Sobel算子计算y方向（垂直）的梯度
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        
        # 这相当于计算梯度幅值的平方的均值
        # 清晰图像的边缘过渡更陡峭，因此梯度值更大，得分也更高
        return np.mean(sobelx**2 + sobely**2)

    @staticmethod
    def hybrid_evaluate(img, weights=None, brightness_threshold=80):
        """
        img: 输入图像
        weights: 权重元组，默认为(0.4, 0.6)
        brightness_threshold: 亮度阈值，默认为80
        """
        if weights is None:
            weights = (0.4, 0.6)  # 默认值
            
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        
        var_score = FocusEvaluator.variance_method(img)
        grad_score = FocusEvaluator.gradient_method(img)
        
        # 根据图像整体亮度调整权重
        mean_intensity = np.mean(img)
        if mean_intensity < brightness_threshold: 
            weights = (0.5, 0.5)  
        
        return weights[0]*var_score + weights[1]*grad_score

    @staticmethod
    def preprocess_image(img, target_size=None):
        """预处理"""
        #使用cv2.resize函数，默认采用双线性插值算法
        if target_size:
            img = cv2.resize(img, target_size) #转为1024分辨率

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR转灰度图
            
        return img.astype(np.float32)  