import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.focus_eval import FocusEvaluator
from test.base_tester import BaseTester  # 导入基础测试类

class CorrelationAnalyzer(BaseTester):  # 继承 BaseTester
    def __init__(self, master, frames, focus_evaluator):
        super().__init__(master)  # 调用父类初始化方法
        self.frames = frames
        self.focus_evaluator = focus_evaluator
        
    def run_analysis(self):
        """运行清晰度评价方法的相关性分析"""
        if not self.frames:
            return None
        
        # 使用基类方法创建窗口
        analysis_window = self.create_test_window("清晰度评价方法相关性分析", "1000x800")
        
        # 创建进度条和状态标签
        _, progress, status_label = self.create_standard_progress_frame()
        
        # 创建表格显示相关系数
        columns = ('评价方法对比', '相关系数')
        tree = ttk.Treeview(analysis_window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor='center')
        
        tree.pack(pady=20, fill='x', padx=20)
        
        # 创建散点图区域
        fig = plt.Figure(figsize=(12, 4), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=analysis_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 使用线程处理计算
        def analyze():
            try:
                variance_scores = []
                gradient_scores = []
                hybrid_scores = []
                total_frames = len(self.frames)

                # 批量处理以提高效率
                batch_size = 10
                for i in range(0, total_frames, batch_size):
                    # 检查窗口是否已关闭
                    if not self.window_active:
                        break
                        
                    batch_end = min(i + batch_size, total_frames)
                    batch_frames = self.frames[i:batch_end]
                    
                    # 批量计算评分
                    for frame in batch_frames:
                        if len(frame.shape) == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        variance_scores.append(FocusEvaluator.variance_method(frame))
                        gradient_scores.append(FocusEvaluator.gradient_method(frame))
                        hybrid_scores.append(self.focus_evaluator(frame))
                    
                    # 安全更新进度
                    progress_value = (batch_end / total_frames) * 100
                    self.update_progress(progress, progress_value)
                    self.update_status(status_label, f"正在分析第 {batch_end}/{total_frames} 帧")
                
                # 检查窗口是否仍然存在
                if not self.window_active:
                    return
                
                # 计算相关系数
                corr_var_grad = stats.pearsonr(variance_scores, gradient_scores)[0]
                corr_var_hyb = stats.pearsonr(variance_scores, hybrid_scores)[0]
                corr_grad_hyb = stats.pearsonr(gradient_scores, hybrid_scores)[0]
                
                # 安全更新UI
                def update_ui():
                    if not self.window_active:
                        return
                    
                    try:
                        # 更新表格
                        tree.delete(*tree.get_children())
                        correlations = [
                            ('方差法-梯度法', f'{corr_var_grad:.3f}'),
                            ('方差法-混合评价', f'{corr_var_hyb:.3f}'),
                            ('梯度法-混合评价', f'{corr_grad_hyb:.3f}')
                        ]
                        for item in correlations:
                            tree.insert('', 'end', values=item)
                        
                        # 更新散点图
                        fig.clear()
                        
                        # 创建三个子图
                        ax1 = fig.add_subplot(131)
                        ax2 = fig.add_subplot(132)
                        ax3 = fig.add_subplot(133)
                        
                        # 归一化分数
                        def normalize(scores):
                            return (np.array(scores) - np.min(scores)) / (np.max(scores) - np.min(scores))
                        
                        norm_variance = normalize(variance_scores)
                        norm_gradient = normalize(gradient_scores)
                        norm_hybrid = normalize(hybrid_scores)
                        
                        # 绘制散点图和趋势线
                        def plot_correlation(ax, x, y, title, corr):
                            ax.scatter(x, y, alpha=0.5)
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            ax.plot(x, p(x), "r--", alpha=0.8)
                            ax.set_title(f'{title}\n相关系数: {corr:.3f}')
                        
                        plot_correlation(ax1, norm_variance, norm_gradient, '方差法 vs 梯度法', corr_var_grad)
                        plot_correlation(ax2, norm_variance, norm_hybrid, '方差法 vs 混合评价', corr_var_hyb)
                        plot_correlation(ax3, norm_gradient, norm_hybrid, '梯度法 vs 混合评价', corr_grad_hyb)
                        
                        fig.tight_layout()
                        canvas.draw()
                        
                        # 更新状态
                        self.update_status(status_label, "分析完成!")
                        self.update_progress(progress, 100)
                    
                    except Exception as e:
                        print(f"更新UI错误: {str(e)}")
                
                # 安全更新UI
                if self.window_active:
                    self.master.after(0, update_ui)
                
            except Exception as e:
                print(f"分析错误: {str(e)}")
                if self.window_active:
                    self.update_status(status_label, f"分析出错: {str(e)}")
        
        # 启动分析线程
        threading.Thread(target=analyze, daemon=True).start()