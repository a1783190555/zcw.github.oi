import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from test.base_tester import BaseTester  # 导入基础测试类

class DenoisePerformanceTester(BaseTester):  # 继承 BaseTester
    def __init__(self, master, frames, preprocessor):
        super().__init__(master)  # 调用父类初始化方法
        self.frames = frames
        self.preprocessor = preprocessor
        
    def calculate_psnr(self, original, denoised):
        """
        计算峰值信噪比（PSNR）
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        """
        # 确保图像类型一致
        original = original.astype(np.float32)
        denoised = denoised.astype(np.float32)
        
        # 计算MSE（均方误差）
        mse = np.mean((original - denoised) ** 2)
        if mse == 0:  # 如果图像完全相同
            return float('inf')
            
        # 计算PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        return psnr

    def add_noise(self, image, noise_type='gaussian', severity=0.05):  # 降低噪声强度
        noisy = image.copy()
        if noise_type == 'gaussian':
            mean = 0
            noise = np.random.normal(mean, severity * 255, image.shape)
            noisy = cv2.add(image, noise.astype(np.uint8))
        return noisy

    def test_denoise_method(self, method, noisy_frame):
        """测试单个去噪方法的性能"""
        start_time = time.time()
        denoised = self.preprocessor.denoise(noisy_frame, method)
        processing_time = time.time() - start_time
        return denoised, processing_time

    def run_test(self):
        """运行去噪性能测试"""
        if not self.frames:
            return
            
        # 使用基类方法创建窗口
        test_window = self.create_test_window("去噪性能测试", "1200x800")
        
        # 创建进度条和状态标签
        _, progress, status_label = self.create_standard_progress_frame()
        
        # 创建结果表格
        columns = ('去噪方法', '平均PSNR(dB)', '平均处理时间(ms)')
        tree = ttk.Treeview(test_window, columns=columns, show='headings')
        tree.pack(pady=20, fill='x', padx=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor='center')
        
        # 创建图表区域
        fig = plt.Figure(figsize=(12, 5))
        canvas = FigureCanvasTkAgg(fig, master=test_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        def run_tests():
            try:
                # 测试参数
                methods = ['gaussian(高斯滤波)', 'median(中值滤波)', 'bilateral(双边滤波)']
                noise_types = ['gaussian', 'salt_pepper']
                results = {method: {
                    'psnr_values': [],
                    'processing_times': []
                } for method in methods}
                
                total_steps = len(self.frames) * len(methods) * len(noise_types)
                current_step = 0
                
                # 对每一帧进行测试
                for frame_idx, frame in enumerate(self.frames):
                    if not self.window_active:
                        break
                        
                    for noise_type in noise_types:
                        if not self.window_active:
                            break
                            
                        # 添加噪声
                        noisy_frame = self.add_noise(frame, noise_type)
                        
                        for method in methods:
                            if not self.window_active:
                                break
                                
                            # 使用当前方法进行去噪
                            denoised_frame, proc_time = self.test_denoise_method(method, noisy_frame)
                            
                            # 计算PSNR
                            psnr = self.calculate_psnr(frame, denoised_frame)
                            
                            # 记录结果
                            results[method]['psnr_values'].append(psnr)
                            results[method]['processing_times'].append(proc_time * 1000)  # 转换为毫秒
                            
                            current_step += 1
                            
                            # 安全更新进度
                            self.update_progress(progress, (current_step / total_steps) * 100)
                            self.update_status(status_label, f"测试中... 帧 {frame_idx + 1}/{len(self.frames)}")
                
                # 检查窗口是否仍然存在
                if not self.window_active:
                    return
                
                # 安全更新表格
                try:
                    tree.delete(*tree.get_children())
                    for method in methods:
                        if not results[method]['psnr_values']:
                            continue  # 跳过没有数据的方法
                            
                        avg_psnr = np.mean(results[method]['psnr_values'])
                        avg_time = np.mean(results[method]['processing_times'])
                        
                        tree.insert('', 'end', values=(
                            method,
                            f"{avg_psnr:.2f}",
                            f"{avg_time:.1f}"
                        ))
                except Exception as e:
                    print(f"更新表格错误: {str(e)}")
                
                # 安全更新图表
                try:
                    if not self.window_active:
                        return
                        
                    fig.clear()
                    
                    # 检查是否有足够数据绘图
                    valid_methods = [m for m in methods if results[m]['psnr_values']]
                    
                    if valid_methods:
                        # PSNR对比图
                        ax1 = fig.add_subplot(121)
                        box_data = [results[m]['psnr_values'] for m in valid_methods]
                        ax1.boxplot(box_data, labels=[m.split('(')[0] for m in valid_methods])
                        ax1.set_title('PSNR分布对比')
                        ax1.set_ylabel('PSNR (dB)')
                        
                        # 处理时间对比图
                        ax2 = fig.add_subplot(122)
                        times = [np.mean(results[m]['processing_times']) for m in valid_methods]
                        ax2.bar([m.split('(')[0] for m in valid_methods], times)
                        ax2.set_title('平均处理时间对比')
                        ax2.set_ylabel('处理时间 (ms)')
                        
                        fig.tight_layout()
                        canvas.draw()
                except Exception as e:
                    print(f"更新图表错误: {str(e)}")
                
                # 安全更新状态
                self.update_status(status_label, "测试完成!")
                
            except Exception as e:
                error_msg = f"测试执行错误: {str(e)}"
                print(error_msg)
                self.update_status(status_label, error_msg)
                
        # 在新线程中运行测试
        threading.Thread(target=run_tests, daemon=True).start()