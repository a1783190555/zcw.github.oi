import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.preprocessing import ImagePreprocessor
from core.focus_eval import FocusEvaluator
from core.autofocus import AutoFocus
import threading,yaml,matplotlib,cv2,sys,os,time,traceback
from test.correlation_analysis import CorrelationAnalyzer
from test.performance_test import PerformanceTester
from test.denoise_test import DenoisePerformanceTester
from config import ConfigEditor
import concurrent.futures
import multiprocessing

class AutoFocusApp:
    # Windows 系统字体设置
    matplotlib.rc('font', family='Microsoft YaHei')
    def __init__(self, master):
        self.master = master
        self.master.title("工业显微镜自动聚焦系统")
        self.master.geometry("1200x900")
        # self.prep = ImagePreprocessor()
        # self.focus_evaluator = FocusEvaluator.hybrid_evaluate
        # self.af = AutoFocus(self.focus_evaluator)
        
        # 添加停止标志
        self.stop_processing = False    #停止激活时为True
        self.processing_active = False  #是否有正在进行的自动对焦处理
        self.calculating_scores = False  # 添加计算评分的状态标志
        
        
        self.create_widgets()
        self.load_config()
        # 使用配置文件中的参数
        self.prep = ImagePreprocessor(self.config['processing']['denoise'])
        self.focus_evaluator = lambda img: FocusEvaluator.hybrid_evaluate(
            img, 
            weights=self.config['evaluation']['weights'],
            brightness_threshold=self.config['evaluation']['brightness_threshold']
        )
        
        self.af = AutoFocus(
            self.focus_evaluator,
            self.config["autofocus"]["coarse_step"],
            self.config["autofocus"]["thread_pool"]["use_cpu_count"],
            self.config["autofocus"]["thread_pool"]["max_workers"],
            self.config["processing"]["preprocessing"]["target_size"]
            )
        
        # 使用配置的显示尺寸
        # self.current_frame = None
        self.original_image = None
        # self.test_results = None
        self.master.bind("<Configure>", self.on_window_resize)
        
        self.frame_scores = []  # 用于存储每一帧的清晰度评分
        

    def load_config(self):
        if getattr(sys, 'frozen', False):
            # 如果是打包后的exe文件
            base_path = sys._MEIPASS
        else:
            # 如果是脚本运行
            base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, 'config.yaml')
        # 指定编码为 UTF-8
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        return self.config

    def create_widgets(self):
        # 创建主布局框架
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 控制面板 - 使用固定宽度
        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        ttk.Button(control_frame, text="加载视频", command=self.load_video).pack(pady=5)

        self.denoise_var = tk.StringVar(value='gaussian(高斯滤波)')
        ttk.Combobox(control_frame, textvariable=self.denoise_var, 
                    values=['gaussian(高斯滤波)', 'median(中值滤波)', 'bilateral(双边滤波)']).pack(pady=5)

        # 创建开始和停止按钮的框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5)

        self.start_button = ttk.Button(button_frame, text="开始对焦", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop_processing_command, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=2)

        # 添加"开始计算"按钮
        score_button_frame = ttk.Frame(control_frame)
        score_button_frame.pack(pady=5)

        self.calculate_button = ttk.Button(score_button_frame, text="开始计算清晰度", 
                                        command=self.async_calculate_scores, state='disabled')
        self.calculate_button.pack(side=tk.LEFT, padx=2)

        self.stop_calculate_button = ttk.Button(score_button_frame, text="停止计算", 
                                            command=self.stop_calculate_scores, state='disabled')
        self.stop_calculate_button.pack(side=tk.LEFT, padx=2)

        # 处理状态标签和进度条
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(pady=5)
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(pady=5, fill='x')

        # 右侧内容框架
        right_frame = ttk.Frame(self.main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建一个容器框架用于并排显示视频预览和聚焦结果
        container_frame = ttk.Frame(right_frame)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 设置两个列，每列权重相等
        container_frame.columnconfigure(0, weight=1)
        container_frame.columnconfigure(1, weight=1)
        container_frame.rowconfigure(0, weight=1)
        
        # 视频预览框 - 左侧，使用grid而不是pack
        video_frame = ttk.LabelFrame(container_frame, text="视频预览")
        video_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        
        # 聚焦结果框 - 右侧，使用grid而不是pack
        result_frame = ttk.LabelFrame(container_frame, text="聚焦结果")
        result_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        
        # 创建固定大小的视频显示容器
        video_container = ttk.Frame(video_frame, width=400, height=300)
        video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        video_container.pack_propagate(False)  # 防止内容影响容器大小
        
        # 创建固定大小的结果显示容器
        result_container = ttk.Frame(result_frame, width=400, height=300)
        result_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        result_container.pack_propagate(False)  # 防止内容影响容器大小
        
        # 视频预览区域
        self.video_canvas = tk.Canvas(video_container, highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 聚焦结果图像区域
        self.result_canvas = tk.Canvas(result_container, highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 视频播放状态区域 - 放在视频预览框的底部
        playback_status_frame = ttk.Frame(video_frame)
        playback_status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 添加播放帧信息标签和播放进度条
        self.frame_label = ttk.Label(playback_status_frame, text="帧: 0/0")
        self.frame_label.pack(side=tk.TOP, fill=tk.X)
        
        self.playback_progress = ttk.Progressbar(playback_status_frame, mode='determinate', length=100)
        self.playback_progress.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.processing_label = ttk.Label(playback_status_frame, text="未开始处理")
        self.processing_label.pack(side=tk.TOP, fill=tk.X)
        
        self.processing_progress = ttk.Progressbar(playback_status_frame, mode='determinate', length=100)
        self.processing_progress.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # 聚焦结果信息区域 - 放在聚焦结果框的底部
        result_status_frame = ttk.Frame(result_frame)
        result_status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 添加结果信息标签
        self.result_info_label = ttk.Label(result_status_frame, text="未执行聚焦", font=('Arial', 10, 'bold'))
        self.result_info_label.pack(side=tk.TOP, fill=tk.X)

        # 曲线图区域
        self.figure = plt.Figure(figsize=(8, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # 在控制面板中添加性能测试按钮
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="性能测试").pack(pady=5)
        ttk.Button(control_frame, text="运行算法性能对比", 
                command=self.run_performance_test).pack(pady=5)
        # 在create_widgets方法中添加测试按钮
        ttk.Button(control_frame, text="运行去噪性能测试", 
                command=self.run_denoise_test).pack(pady=5)
        
        # 在性能测试按钮后面添加相关性分析按钮
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="算法分析").pack(pady=5)
        ttk.Button(control_frame, text="清晰度评价相关性分析", 
                command=self.run_correlation_analysis).pack(pady=5)
        
        # 将此添加到AutoFocusApp类的create_widgets方法中
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="配置管理").pack(pady=5)
        ttk.Button(control_frame, text="重新加载配置", 
                command=self.reload_config).pack(pady=5)

        # 将此按钮添加到AutoFocusApp类的create_widgets方法中
        ttk.Button(control_frame, text="编辑配置", 
                command=self.open_config_editor).pack(pady=5)
        
        ttk.Button(control_frame, text="恢复默认配置", 
        command=self.reset_config_to_default).pack(pady=5)

    def stop_calculate_scores(self):
            """停止计算评分"""
            self.calculating_scores = False
            self.stop_calculate_button.config(state='disabled')
            self.calculate_button.config(state='normal')
            self.status_label.config(text="计算已停止")

    # 修改窗口大小变化处理方法
    def on_window_resize(self, event):
        """处理窗口大小改变事件，确保图像正确对齐"""
        # 如果有视频图像，需要先更新视频画布
        # 这确保display_size和image_offsets首先由视频画布设置
        if hasattr(self, 'original_video_image') and self.original_video_image is not None:
            self._update_canvas_image(self.original_video_image, self.video_canvas, "video_tk_image")
        
        # 然后更新结果画布，使用与视频相同的尺寸和偏移
        if hasattr(self, 'original_image') and self.original_image is not None:
            self._update_canvas_image(self.original_image, self.result_canvas, "result_tk_image")

        # 更新图表大小
        if hasattr(self, 'canvas'):
            plot_width = self.canvas.get_tk_widget().winfo_width()
            plot_height = self.canvas.get_tk_widget().winfo_height()
            if plot_width > 0 and plot_height > 0:
                dpi = self.figure.get_dpi()
                self.figure.set_size_inches(plot_width/dpi, plot_height/dpi)
                self.canvas.draw()
    # 添加视频播放功能
    def load_video(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return
        
        self.status_label.config(text="加载中...")
        self.progress.start(10)
        self.master.config(cursor="watch")
        
        # 先打开视频播放
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.status_label.config(text="视频加载失败")
            self.progress.stop()
            self.master.config(cursor="arrow")
            return
        
        # 获取视频信息
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 重置播放状态
        self.frame_label.config(text=f"帧: 0/{self.total_frames}")
        self.playback_progress.config(maximum=self.total_frames, value=0)
        self.processing_label.config(text="未开始处理")
        self.processing_progress.config(maximum=self.total_frames, value=0)
        
        # 开始播放
        self.is_playing = True
        self.current_frame_idx = 0
        self.frames = []  # 清空之前的帧
        self.play_next_frame()
        
        # 启动后台线程处理帧
        self.video_thread = threading.Thread(target=self.process_video_frames, args=(file_path,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def play_next_frame(self):
        """用于视频预览播放视频下一帧"""
        if not self.is_playing or not hasattr(self, 'cap') or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 更新视频预览区域
            self.update_video_frame(frame)
            self.current_frame_idx += 1
            
            # 更新播放状态区域
            self.frame_label.config(text=f"帧: {self.current_frame_idx}/{self.total_frames}")
            self.playback_progress.config(value=self.current_frame_idx)
            
            # 计算下一帧的延迟时间
            delay = int(1000 / self.video_fps)
            # 设置下一帧的播放
            self.master.after(delay, self.play_next_frame)
        else:
            # 视频播放完毕
            self.is_playing = False
            self.status_label.config(text="视频播放完成")
            
            # 重置视频帧索引，以便重新播放
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 检查是否已处理完所有帧
            if self.video_thread and self.video_thread.is_alive():
                self.frame_label.config(text="播放完成")
            else:
                self.frame_label.config(text="播放完成")
                self.processing_label.config(text="处理完成")

    def process_video_frames(self, file_path):
        """在后台处理视频帧，使用多线程加速"""
        try:
            # 确保导入必要的库

            
            # 打开另一个视频捕获对象用于处理
            process_cap = cv2.VideoCapture(file_path)
            
            if not process_cap.isOpened():
                self.master.after(0, lambda: self.status_label.config(text="处理视频失败"))
                self.master.after(0, lambda: self.progress.stop())
                self.master.after(0, lambda: self.master.config(cursor=""))
                return
            
            total_frames = int(process_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 更新处理状态初始值
            self.master.after(0, lambda: self.processing_label.config(text="开始处理"))
            self.master.after(0, lambda: self.processing_progress.config(maximum=total_frames, value=0))
            
            # 读取所有帧
            frames_to_process = []
            while process_cap.isOpened():
                ret, frame = process_cap.read()
                if not ret: break
                frames_to_process.append(frame)
            
            process_cap.release()
            
            # 使用线程池并行处理帧
            self.frames = []  # 初始化为空列表，不预分配
            processed_count = 0
            
            # 确定线程数
            n_cores = multiprocessing.cpu_count()  # 简化线程数确定逻辑
            print(f"视频处理使用 {n_cores} 个线程")
            
            # 定义一个函数来处理单个帧
            def process_frame(frame_idx, frame):
                try:
                    denoise_method = self.denoise_var.get()
                    processed = self.prep.denoise(frame, denoise_method)
                    processed = self.prep.enhance_contrast(processed)
                    return frame_idx, processed
                except Exception as e:
                    print(f"处理帧 {frame_idx} 时出错: {e}")
                    return frame_idx, None
            
            # 使用线程池并行处理
            processed_frames = [None] * len(frames_to_process)  # 预分配结果列表
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
                # 提交所有任务
                future_list = []
                for i, frame in enumerate(frames_to_process):
                    future = executor.submit(process_frame, i, frame)
                    future_list.append(future)
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_list):
                    try:
                        idx, processed_frame = future.result()
                        if processed_frame is not None:
                            processed_frames[idx] = processed_frame
                        
                        # 更新进度信息
                        processed_count += 1
                        
                        # 每10帧更新一次，避免UI过度刷新
                        if processed_count % 10 == 0:
                            count = processed_count  # 创建局部变量以避免lambda中使用循环变量
                            self.master.after(0, lambda p=count: 
                                self.processing_label.config(text=f"处理: {p}/{total_frames}"))
                            self.master.after(0, lambda p=count: 
                                self.processing_progress.config(value=p))
                    except Exception as e:
                        print(f"获取处理结果时出错: {e}")
            
            # 移除任何处理失败的帧（值为None的项）
            self.frames = [f for f in processed_frames if f is not None]
            
            # 处理完成后更新状态
            self.master.after(0, lambda: self.status_label.config(text="视频帧处理完成，可以开始聚焦"))
            self.master.after(0, lambda: self.processing_label.config(text="处理完成"))
            self.master.after(0, lambda: self.processing_progress.config(value=total_frames))
            self.master.after(0, lambda: self.progress.stop())
            self.master.after(0, lambda: self.master.config(cursor=""))
            
        except Exception as e:
            error_msg = traceback.format_exc()  # 获取完整的错误堆栈
            print(f"处理视频时出错: {error_msg}")
            self.master.after(0, lambda: self.status_label.config(text=f"处理出错: {str(e)}"))
            self.master.after(0, lambda: self.processing_label.config(text=f"处理失败"))
            self.master.after(0, lambda: self.progress.stop())
            self.master.after(0, lambda: self.master.config(cursor=""))
                
    
    def update_video_frame(self, frame):
        """更新视频预览区域的图像"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # 存储原始视频图像，用于同步两个显示区域
        self.original_video_image = img
        
        # 使用辅助方法更新画布
        self._update_canvas_image(img, self.video_canvas, "video_tk_image")

    
    def update_image(self, frame):
        """更新聚焦结果区域的图像"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.original_image = img
        
        # 使用辅助方法更新画布
        self._update_canvas_image(img, self.result_canvas, "result_tk_image")

    def calculate_display_size(self, img, container_width, container_height):
        """计算适合容器的图像显示尺寸"""
        img_ratio = img.width / img.height
        container_ratio = container_width / container_height
        
        if img_ratio > container_ratio:
            # 图像比容器更"宽"，以宽度为基准缩放
            new_width = container_width
            new_height = int(container_width / img_ratio)
        else:
            # 图像比容器更"高"，以高度为基准缩放
            new_height = container_height
            new_width = int(container_height * img_ratio)
        
        return new_width, new_height

    def start_processing(self):
        """开始处理"""
        if not hasattr(self, 'frames') or not self.frames:
            self.status_label.config(text="请先加载视频")
            return
            
        # 确保视频已经处理完成
        if hasattr(self, 'video_thread') and self.video_thread.is_alive():
            self.status_label.config(text="请等待视频处理完成")
            return
        
        self.stop_processing = False
        self.processing_active = True
        
        # 更新UI状态
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="开始自动聚焦...")
        
        # 启动进度条
        self.progress.start(10)
        
        # 使用线程执行聚焦过程
        self.processing_thread = threading.Thread(target=self.run_autofocus_process)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 定期检查处理状态
        self.check_processing_status()
            
    #处理状态检查方法
    def check_processing_status(self):
        """检查处理状态"""
        if self.processing_active:
            # 如果处理仍在进行，继续检查
            self.master.after(100, self.check_processing_status)
        else:
            # 处理已结束，重置UI状态
            self.progress.stop()
            self.master.config(cursor="arrow")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
        
    def stop_processing_command(self):
        """停止处理命令"""
        self.stop_processing = True
        self.status_label.config(text="正在停止...")
        self.stop_button.config(state='disabled')
        
        # 确保进度条停止和鼠标状态恢复
        self.master.after(0, lambda: self.progress.stop())
        self.master.after(0, lambda: self.master.config(cursor=""))
    
    #聚焦和结果信息显示
    def run_autofocus_process(self):
        """运行自动对焦过程"""
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 初始化帧评分列表
            self.frame_scores = [None] * len(self.frames)
            
            # 执行粗搜索
            self.master.after(0, lambda: self.status_label.config(text="执行粗搜索..."))
            start, end, coarse_scores = self.af.coarse_search(self.frames)
            
            # 存储粗搜索的评分结果
            for idx, score in coarse_scores.items():
                self.frame_scores[idx] = score 

            if not self.stop_processing:
                # 执行精搜索
                self.master.after(0, lambda: self.status_label.config(text="执行精搜索..."))
                
                #返回索引和评分zidain
                best_frame, best_frame_idx, fine_scores = self.af.fine_search_with_index(self.frames, start, end)
                self.best_frame_idx = best_frame_idx  #存储最佳帧索引
                
                # 存储精搜索的评分结果
                for idx, score in fine_scores.items():
                    self.frame_scores[idx] = score

                # 计算用时
                elapsed_time = time.time() - start_time
                
                if not self.stop_processing:
                    # 更新UI显示最佳帧
                    self.master.after(0, lambda: self.update_image(best_frame))
                    
                    # 准备结果信息
                    result_info = "对焦完成"
                    if best_frame_idx is not None:
                        # 计算时间信息
                        if hasattr(self, 'video_fps') and self.video_fps > 0:
                            seconds = best_frame_idx / self.video_fps
                            minutes = int(seconds / 60)
                            seconds = seconds % 60
                            time_info = f"{minutes}分{seconds:.2f}秒"
                        else:
                            time_info = "未知时间"
                        
                        result_info = f"最佳帧: {best_frame_idx} ({time_info})"
                    
                    # 更新结果信息
                    self.master.after(0, lambda info=result_info: self.result_info_label.config(text=info))
                    
                    # 显示聚焦用时
                    self.master.after(0, lambda time=elapsed_time: self.status_label.config(
                        text=f"对焦完成! 用时: {time:.3f}秒"))
                    
                    # 启用"开始计算"按钮
                    self.master.after(0, lambda: self.calculate_button.config(state='normal'))
                    
                    # 更新曲线图(添加这一行)
                    self.master.after(0, self.update_focus_plot)
            
        except Exception as e:
            self.master.after(0, lambda e=e: self.status_label.config(text=f"处理出错: {str(e)}"))

        if self.stop_processing:
            self.master.after(0, lambda: self.status_label.config(text="处理已停止"))

        self.processing_active = False
        self.master.after(0, lambda: self.start_button.config(state='normal'))
        self.master.after(0, lambda: self.stop_button.config(state='disabled'))

    def update_focus_plot(self):
        """根据已收集的评分数据更新曲线图"""
        # 确保有评分数据
        if not hasattr(self, 'frame_scores') or not self.frame_scores or all(score is None for score in self.frame_scores):
            return
        
        # 过滤出非空评分，同时保留索引信息
        scores_with_idx = [(idx, score) for idx, score in enumerate(self.frame_scores) if score is not None]
        
        if not scores_with_idx:
            return
        
        # 分离索引和评分
        indices, scores = zip(*scores_with_idx)
        
        # 按索引排序的完整评分列表(对于不存在的评分用None填充)
        full_scores = [None] * len(self.frames)
        for idx, score in zip(indices, scores):
            full_scores[idx] = score
        
        # 清除图形
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 创建x轴标签(帧索引)
        x = list(indices)
        
        # 绘制散点图，只显示有评分的点
        ax.scatter(indices, scores, c='blue', s=20, alpha=0.7)
        
        # 如果点足够多，尝试用虚线连接点来估计趋势
        if len(indices) > 1:
            # 按索引排序
            sorted_indices = sorted(range(len(indices)), key=lambda k: indices[k])
            sorted_x = [indices[i] for i in sorted_indices]
            sorted_y = [scores[i] for i in sorted_indices]
            ax.plot(sorted_x, sorted_y, 'b--', linewidth=1, alpha=0.5)
        
        ax.set_title('图像清晰度评分曲线', fontsize=10)
        ax.set_xlabel('帧序号', fontsize=9)
        ax.set_ylabel('清晰度评分', fontsize=9)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 突出显示最佳帧
        if hasattr(self, 'best_frame_idx') and self.best_frame_idx is not None:
            best_score = self.frame_scores[self.best_frame_idx]
            if best_score is not None:
                ax.plot(self.best_frame_idx, best_score, 'ro', markersize=8, label='最佳帧')
                ax.annotate(f'最佳帧: {self.best_frame_idx}', 
                        xy=(self.best_frame_idx, best_score),
                        xytext=(self.best_frame_idx+5, best_score),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                        fontsize=8)
        
        # 调整布局并绘制
        self.figure.tight_layout()
        self.canvas.draw()
    def async_calculate_scores(self):
        """计算并显示清晰度评分曲线"""
        # Windows 系统字体设置
        matplotlib.rc('font', family='Microsoft YaHei')
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 检查是否已有评分数据
        if hasattr(self, 'frame_scores') and any(score is not None for score in self.frame_scores):
            # 已有部分评分数据，只计算缺失的
            self.calculating_scores = True
            
            # 更新按钮状态
            self.calculate_button.config(state='disabled')
            self.stop_calculate_button.config(state='normal')
            
            # 获取需要计算的帧索引
            frames_to_calculate = [(idx, frame) for idx, (frame, score) in enumerate(zip(self.frames, self.frame_scores)) 
                                if score is None]
            
            if not frames_to_calculate:
                # 如果所有帧都已有评分，直接更新图表
                self.update_focus_plot()
                self.status_label.config(text="使用已有评分数据")
                self.calculating_scores = False
                self.calculate_button.config(state='normal')
                self.stop_calculate_button.config(state='disabled')
                return
            
            total_to_calculate = len(frames_to_calculate)
            
            def score_generator():
                for i, (idx, frame) in enumerate(frames_to_calculate):
                    if not self.calculating_scores:
                        break
                    # 添加预处理步骤
                    processed_frame = FocusEvaluator.preprocess_image(frame, target_size=(1024, 1024))
                    score = self.focus_evaluator(processed_frame)
                    self.frame_scores[idx] = score  # 存储计算的评分
                    yield i, total_to_calculate, idx, score
            
            gen = score_generator()
            
            def update_plot():
                try:
                    if self.calculating_scores:
                        i, total, idx, score = next(gen)
                        
                        # 更新进度
                        progress_text = f"计算进度: {i+1}/{total} (仅计算缺失评分)"
                        self.status_label.config(text=progress_text)
                        
                        # 每计算10帧或计算完成时更新图表
                        if (i+1) % 10 == 0 or i+1 == total:
                            self.update_focus_plot()
                        
                        self.master.after(1, update_plot)
                        
                except StopIteration:
                    if self.calculating_scores:
                        self.status_label.config(text="计算完成")
                        self.update_focus_plot()
                        self.calculating_scores = False
                    self.calculate_button.config(state='normal')
                    self.stop_calculate_button.config(state='disabled')

            update_plot()
            
        else:
            # 没有现有数据，计算所有帧
            self.frame_scores = [None] * len(self.frames)
            self.calculating_scores = True
            
            # 更新按钮状态
            self.calculate_button.config(state='disabled')
            self.stop_calculate_button.config(state='normal')
            
            def score_generator():
                for idx, frame in enumerate(self.frames):
                    if not self.calculating_scores:
                        break
                    # 添加预处理步骤
                    processed_frame = FocusEvaluator.preprocess_image(frame, target_size=(1024, 1024))
                    score = self.focus_evaluator(processed_frame)
                    self.frame_scores[idx] = score  # 存储计算的评分
                    yield idx, score
            
            gen = score_generator()
            
            def update_plot():
                try:
                    if self.calculating_scores:
                        idx, score = next(gen)
                        
                        # 每计算10帧或计算最后一帧时更新图表
                        if (idx+1) % 10 == 0 or idx+1 == len(self.frames):
                            self.update_focus_plot()
                        
                        self.status_label.config(text=f"计算进度: {idx+1}/{len(self.frames)}")
                        self.master.after(1, update_plot)
                        
                except StopIteration:
                    if self.calculating_scores:
                        self.status_label.config(text="计算完成")
                        self.update_focus_plot()
                        self.calculating_scores = False
                    self.calculate_button.config(state='normal')
                    self.stop_calculate_button.config(state='disabled')

            update_plot()

    # 在 AutoFocusApp 类中添加以下方法
    def run_performance_test(self):
        """运行性能对比测试"""
        if not hasattr(self, 'frames') or not self.frames:
            self.status_label.config(text="请先加载视频")
            return
        
        # 将相关配置部分传递给测试器
        tester = PerformanceTester(
            self.master, 
            self.frames, 
            self.af,
            preprocessing_size=self.config['processing']['preprocessing']['target_size']
        )
        tester.run_test()

    def run_correlation_analysis(self):
        """运行相关性分析"""
        if not hasattr(self, 'frames') or not self.frames:
            self.status_label.config(text="请先加载视频")
            return
            
        analyzer = CorrelationAnalyzer(self.master, self.frames, self.focus_evaluator)
        analyzer.run_analysis()
    
    # 在AutoFocusApp类中添加
    def run_denoise_test(self):
        """运行去噪性能测试"""
        if not hasattr(self, 'frames') or not self.frames:
            self.status_label.config(text="请先加载视频")
            return
            
        tester = DenoisePerformanceTester(self.master, self.frames, self.prep)
        tester.run_test()


    def reload_config(self):
        """重新加载配置文件并应用更改"""
        # 存储旧配置以便比较
        old_config = self.config.copy() if hasattr(self, 'config') else None
        
        # 加载新配置
        self.load_config()
        
        # 应用新配置
        if old_config is None or old_config != self.config:
            # 如果去噪设置改变，更新预处理器
            if old_config is None or old_config['processing']['denoise'] != self.config['processing']['denoise']:
                self.prep = ImagePreprocessor(self.config['processing']['denoise'])
                
            # 如果权重或亮度阈值改变，更新对焦评估器
            if (old_config is None or 
                old_config['evaluation']['weights'] != self.config['evaluation']['weights'] or
                old_config['evaluation']['brightness_threshold'] != self.config['evaluation']['brightness_threshold']):
                
                self.focus_evaluator = lambda img: FocusEvaluator.hybrid_evaluate(
                    img, 
                    weights=self.config['evaluation']['weights'],
                    brightness_threshold=self.config['evaluation']['brightness_threshold']
                )
                
            # 如果粗搜索步长改变，更新自动对焦组件
            if old_config is None or old_config['autofocus'] != self.config['autofocus']:
                print("检测到 autofocus 配置变更，重新创建 AutoFocus 实例")
                self.af = AutoFocus(
                    self.focus_evaluator,
                    coarse_step=self.config['autofocus']['coarse_step'],
                    config=self.config
                )
                    
            # 记录配置更改
            self.status_label.config(text="配置已重新加载")
            
            return True
        return False
    
    
    
    # 将此方法添加到AutoFocusApp类
    def open_config_editor(self):
        """打开配置编辑器"""
        if getattr(sys, 'frozen', False):
            # 如果是作为可执行文件运行
            base_path = sys._MEIPASS
        else:
            # 如果是作为脚本运行
            base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, 'config.yaml')
        
        # 创建编辑器，并在保存时重新加载配置
        ConfigEditor(self.master, config_path, on_save_callback=self.reload_config)
    
    
    def reset_config_to_default(self):
        """
        在主程序中调用ConfigEditor的重置默认配置功能
        """
        # 获取配置文件路径
        if getattr(sys, 'frozen', False):
            # 如果是打包后的exe文件
            base_path = sys._MEIPASS
        else:
            # 如果是脚本运行
            base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, 'config.yaml')
        
        # 创建临时的ConfigEditor实例
        temp_editor = ConfigEditor(self.master, config_path, on_save_callback=self.reload_config)
        
        # 调用其重置方法
        temp_editor.reset_to_default_config()
        
        # 关闭编辑器窗口(如果还打开的话)
        if hasattr(temp_editor, 'window') and temp_editor.window:
            temp_editor.window.destroy()
        
        # 重新加载配置以应用默认设置
        self.reload_config()
        
        # 更新状态提示
        self.status_label.config(text="已恢复默认配置")
        
    #图像对齐并且显示图像
    def _update_canvas_image(self, img, canvas, image_attr_name):
        """
        参数:
        img -- PIL.Image对象
        canvas -- 目标Canvas对象
        image_attr_name -- 存储TkImage的属性名（"video_tk_image"或"result_tk_image"）
        """
        # 获取Canvas尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # 如果Canvas尺寸还未确定，使用默认值
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        # 计算适合Canvas的图像尺寸
        new_width, new_height = self.calculate_display_size(img, canvas_width, canvas_height)
        
        # 检查是否来自视频画布的更新
        is_video_canvas = (canvas == self.video_canvas)
        
        if is_video_canvas:
            # 对于视频画布，保存尺寸供两个画布共用
            self.display_size = (new_width, new_height)
            
            # 计算图像在Canvas中的居中位置
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # 保存偏移量供两个画布共用
            self.image_offsets = (x_offset, y_offset)
        else:
            # 对于结果画布，使用与视频相同的尺寸和偏移
            if hasattr(self, 'display_size'):
                new_width, new_height = self.display_size
            if hasattr(self, 'image_offsets'):
                x_offset, y_offset = self.image_offsets
            else:
                # 如果没有预先计算的值，则计算（不太可能发生）
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2
        
        # 调整图像大小
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 更新图像引用
        tk_image = ImageTk.PhotoImage(img_resized)
        setattr(self, image_attr_name, tk_image)
        
        # 清除Canvas并创建图像
        canvas.delete("all")
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=getattr(self, image_attr_name))

    # # 辅助方法用于统一UI状态更新
    # def update_ui_status(self, text, is_error=False):
    #     """统一的UI状态更新方法"""
    #     self.status_label.config(text=text)
    #     if is_error:
    #         print(f"错误: {text}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = AutoFocusApp(root)
    root.mainloop()