import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from core.focus_eval import FocusEvaluator
from test.base_tester import BaseTester  # 导入基础测试类

PREPROCESSING_SIZE = (1024, 1024)  # 统一的预处理尺寸

class PerformanceTester(BaseTester):  # 继承 BaseTester
    def __init__(self, master, frames, af, preprocessing_size=None):
        super().__init__(master)  # 调用父类初始化方法
        self.frames = frames
        self.af = af
        # 使用配置值或回退到默认值
        self.preprocessing_size = preprocessing_size or PREPROCESSING_SIZE
        # 使用与AutoFocus相同的评估器
        self.focus_evaluator = af.evaluator
        
    def preprocess_consistently(self, frame):
        """统一的预处理方法"""
        # 使用FocusEvaluator中的预处理方法，避免重复实现
        return FocusEvaluator.preprocess_image(frame, target_size=self.preprocessing_size)

    def calculate_accuracy(self, found_idx, reference_idx):
        """计算准确度 - 改进的评分机制"""
        total_frames = len(self.frames)
        distance = abs(found_idx - reference_idx)
        
        # 调整评分标准，使其更合理
        if distance == 0:
            base_score = 100  # 完全匹配
        elif distance <= 1:
            base_score = 95   # 相邻帧
        elif distance <= 2:
            base_score = 90   # 非常接近
        elif distance <= 3:
            base_score = 85   # 比较接近
        elif distance <= 4:
            base_score = 80   # 较为接近
        else:
            # 使用更平滑的衰减
            base_score = max(60, 80 - (distance - 4) * 3)
        
        # 添加帧质量评分，但降低其权重
        quality_score = self._evaluate_frames_similarity(found_idx, reference_idx)
        
        # 调整权重比例，更重视位置准确性
        final_score = 0.8 * base_score + 0.2 * quality_score
        
        return round(final_score, 1)

    def _evaluate_frames_similarity(self, found_idx, reference_idx):
        """评估两帧之间的相似度"""
        found_frame = self.frames[found_idx]
        reference_frame = self.frames[reference_idx]
        
        # 获取两帧的评分，使用相同的预处理和评价方法
        processed_found = self.preprocess_consistently(found_frame)
        processed_ref = self.preprocess_consistently(reference_frame)
        
        # 使用与AutoFocus相同的评估方法
        found_score = self.focus_evaluator(processed_found)
        reference_score = self.focus_evaluator(processed_ref)
        
        # 计算评分的相对差异
        score_diff = abs(found_score - reference_score)
        max_score = max(found_score, reference_score)
        
        # 使用对数比例计算质量得分，使评分更合理
        if max_score == 0:
            return 0
        quality_ratio = 1 - (score_diff / max_score)
        quality_score = 100 * quality_ratio
        
        return quality_score

    def find_best_frame_for_method(self, method_name):
        """根据不同方法找出其认为的最佳帧"""
        best_score = float('-inf')
        best_idx = 0
        start_time = time.time()
        
        try:
            if method_name == "分层搜索":
                # 执行搜索
                start, end, _ = self.af.coarse_search(self.frames)
                best_frame, best_idx, _ = self.af.fine_search_with_index(self.frames, start, end)
            
            elif method_name == "穷举搜索":
                # 穷举搜索遍历所有帧
                for i, frame in enumerate(self.frames):
                    processed = self.preprocess_consistently(frame)
                    score = self.focus_evaluator(processed)
                    if score > best_score:
                        best_score = score
                        best_idx = i
            
            elif method_name == "跳跃搜索":
                # 使用跳跃搜索策略，先大步探索，再细化
                jump_size = max(1, len(self.frames) // 10)  # 初始跳跃步长
                best_idx_temp = 0
                
                # 第一阶段：大步跳跃搜索
                for i in range(0, len(self.frames), jump_size):
                    processed = self.preprocess_consistently(self.frames[i])
                    score = self.focus_evaluator(processed)
                    if score > best_score:
                        best_score = score
                        best_idx_temp = i
                
                # 第二阶段：在最佳跳跃点附近细化搜索
                start = max(0, best_idx_temp - jump_size)
                end = min(len(self.frames) - 1, best_idx_temp + jump_size)
                
                best_score = float('-inf')  # 重置分数进行细化搜索
                for i in range(start, end + 1):
                    processed = self.preprocess_consistently(self.frames[i])
                    score = self.focus_evaluator(processed)
                    if score > best_score:
                        best_score = score
                        best_idx = i
            
            elif method_name == "二分搜索":
                # 实现二分搜索算法
                # 注意：这里假设图像清晰度分数在某个点达到峰值，两侧递减
                # 这是一个单峰函数的二分搜索变体
                
                left, right = 0, len(self.frames) - 1
                
                while left <= right:
                    if right - left <= 2:  # 区间很小时直接比较
                        best_idx = left
                        best_score = float('-inf')
                        
                        for i in range(left, right + 1):
                            processed = self.preprocess_consistently(self.frames[i])
                            score = self.focus_evaluator(processed)
                            if score > best_score:
                                best_score = score
                                best_idx = i
                        break
                    
                    # 取三个点进行比较
                    mid1 = left + (right - left) // 3
                    mid2 = right - (right - left) // 3
                    
                    processed1 = self.preprocess_consistently(self.frames[mid1])
                    processed2 = self.preprocess_consistently(self.frames[mid2])
                    
                    score1 = self.focus_evaluator(processed1)
                    score2 = self.focus_evaluator(processed2)
                    
                    # 根据比较结果缩小搜索范围
                    if score1 > score2:
                        right = mid2
                    else:
                        left = mid1
            
            elapsed_time = time.time() - start_time
            return best_idx, elapsed_time
            
        except Exception as e:
            print(f"方法 {method_name} 测试出错: {str(e)}")
            return 0, time.time() - start_time

    def find_reference_frame(self):
        """使用混合评价方法找出参考标准帧"""
        best_score = float('-inf')
        best_idx = 0
        
        for i, frame in enumerate(self.frames):
            processed = self.preprocess_consistently(frame)
            # A使用与AutoFocus相同的评估方法
            score = self.focus_evaluator(processed)
            if score > best_score:
                best_score = score
                best_idx = i
                
        return best_idx

    def run_test(self):
        """运行性能对比测试"""
        if not self.frames:
            return
            
        # 使用基类方法创建窗口
        test_window = self.create_test_window("算法性能对比", "1000x800")
        
        # 创建进度条和状态标签
        _, progress, status_label = self.create_standard_progress_frame()
        
        # 创建表格
        columns = ('算法', '准确度', '平均用时(秒)', '最佳帧位置')
        tree = ttk.Treeview(test_window, columns=columns, show='headings')
        tree.pack(pady=20, fill='x', padx=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor='center')
        
        # 创建图表区域
        fig = Figure(figsize=(12, 5))
        canvas = FigureCanvasTkAgg(fig, master=test_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        def run_tests():
            try:
                # 首先找出参考标准帧
                reference_idx = self.find_reference_frame()
                
                # 安全更新状态
                self.update_status(status_label, f"参考标准帧位置: {reference_idx}")
                
                # 测试参数
                methods = ["穷举搜索", "分层搜索", "跳跃搜索", "二分搜索"] 
                num_runs = 5  # 每个方法运行5次
                results = {method: {'accuracy': [], 'times': []} for method in methods}
                
                total_steps = len(methods) * num_runs
                current_step = 0
                
                # 运行测试
                for method in methods:
                    if not self.window_active:
                        break
                        
                    for i in range(num_runs):
                        if not self.window_active:
                            break
                            
                        found_idx, elapsed_time = self.find_best_frame_for_method(method)
                        accuracy = self.calculate_accuracy(found_idx, reference_idx)
                        
                        results[method]['times'].append(elapsed_time)
                        results[method]['accuracy'].append(accuracy)
                        
                        current_step += 1
                        
                        # 安全更新进度
                        self.update_progress(progress, (current_step / total_steps) * 100)
                        self.update_status(status_label, f"测试{method} - 第{i+1}次...")
                
                # 检查窗口是否仍然存在
                if not self.window_active:
                    return
                
                # 安全更新表格
                try:
                    tree.delete(*tree.get_children())
                    for method in methods:
                        if not results[method]['accuracy'] or not results[method]['times']:
                            continue  # 跳过没有完整数据的方法
                            
                        avg_accuracy = np.mean(results[method]['accuracy'])
                        avg_time = np.mean(results[method]['times'])
                        best_idx, _ = self.find_best_frame_for_method(method)
                        
                        tree.insert('', 'end', values=(
                            method,
                            f"{avg_accuracy:.1f}%",
                            f"{avg_time:.3f}",
                            str(best_idx)
                        ))
                except Exception as e:
                    print(f"更新表格错误: {str(e)}")
                
                # 安全更新图表
                try:
                    if not self.window_active:
                        return
                        
                    fig.clear()
                    
                    # 筛选有完整数据的方法
                    valid_methods = [m for m in methods if results[m]['accuracy'] and results[m]['times']]
                    
                    if valid_methods:  # 确保有有效数据
                        # 准确度对比图
                        ax1 = fig.add_subplot(121)
                        accuracies = [np.mean(results[m]['accuracy']) for m in valid_methods]
                        accuracy_std = [np.std(results[m]['accuracy']) for m in valid_methods]
                        ax1.bar(valid_methods, accuracies, yerr=accuracy_std, capsize=5)
                        ax1.set_title('准确度对比')
                        ax1.set_ylabel('准确度 (%)')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # 用时对比图
                        ax2 = fig.add_subplot(122)
                        times = [np.mean(results[m]['times']) for m in valid_methods]
                        time_std = [np.std(results[m]['times']) for m in valid_methods]
                        ax2.bar(valid_methods, times, yerr=time_std, capsize=5)
                        ax2.set_title('平均用时对比')
                        ax2.set_ylabel('用时 (秒)')
                        ax2.tick_params(axis='x', rotation=45)
                    
                    fig.tight_layout()
                    canvas.draw()
                except Exception as e:
                    print(f"更新图表错误: {str(e)}")
                
                # 安全更新状态
                self.update_status(status_label, "测试完成!")
                
            except Exception as e:
                error_msg = f"测试执行错误: {str(e)}"
                print(error_msg)  # 总是打印到控制台
                self.update_status(status_label, error_msg)
                
        # 在新线程中运行测试
        threading.Thread(target=run_tests, daemon=True).start()