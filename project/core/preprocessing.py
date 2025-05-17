import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self,config=None):
        self.config = config or {}
        
        self.denoise_methods = {
            'gaussian(高斯滤波)': {
                'func': cv2.GaussianBlur,
                'params': {
                    'light': {'ksize': (3,3), 'sigmaX': 0},
                    'medium': {'ksize': (5,5), 'sigmaX': 0},
                    'heavy': {'ksize': (7,7), 'sigmaX': 0}
                }
            },
            'median(中值滤波)': {
                'func': cv2.medianBlur,
                'params': {
                    'light': {'ksize': 3},
                    'medium': {'ksize': 5},
                    'heavy': {'ksize': 7}
                }
            },
            'bilateral(双边滤波)': {
                'func': cv2.bilateralFilter,
                'params': {
                    'light': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
                    'medium': {'d': 15, 'sigmaColor': 75, 'sigmaSpace': 75},
                    'heavy': {'d': 21, 'sigmaColor': 75, 'sigmaSpace': 75}
                }
            }
        }

    def estimate_noise(self, img):
        """判断噪声级别"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 从配置中获取阈值
        thresholds = self.config.get('thresholds', {})
        light_threshold = thresholds.get('light', 500)
        medium_threshold = thresholds.get('medium', 1000)
        
        if noise < light_threshold:
            return 'light'
        elif noise < medium_threshold:
            return 'medium'
        return 'heavy'

    def denoise(self, img, method='bilateral(双边滤波)'):

        noise_level = self.estimate_noise(img)
        method_info = self.denoise_methods[method]
        
        # 根据噪声级别获取参数
        params = method_info['params'][noise_level]
        
        # 调用相应的OpenCV函数
        return method_info['func'](img, **params)

    def enhance_contrast(self, img):
        """对比度增强"""
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 分离L通道
        l, a, b = cv2.split(lab)
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # 应用CLAHE到L通道
        cl = clahe.apply(l)
        # 合并通道
        merged = cv2.merge((cl,a,b))
        # 转换回BGR颜色空间
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)