import tkinter as tk
from tkinter import ttk

class BaseTester:
    """
    所有测试类的基类，提供共用的窗口状态管理功能
    """
    def __init__(self, master):
        self.master = master
        self.window_active = False
        self.test_window = None
    
    def create_test_window(self, title, size="1000x800"):
        """
        创建测试窗口并设置关闭处理
        
        参数:
            title (str): 窗口标题
            size (str): 窗口尺寸 (默认: "1000x800")
            
        返回:
            tk.Toplevel: 测试窗口对象
        """
        self.test_window = tk.Toplevel(self.master)
        self.test_window.title(title)
        self.test_window.geometry(size)
        
        # 设置窗口状态标志
        self.window_active = True
        
        # 设置窗口关闭时的回调
        def on_window_close():
            self.window_active = False
            self.test_window.destroy()
        
        self.test_window.protocol("WM_DELETE_WINDOW", on_window_close)
        return self.test_window
    
    def safe_update_widget(self, widget, update_func, *args, **kwargs):
        """
        安全地更新窗口部件，避免窗口关闭后的错误
        
        参数:
            widget: 要更新的窗口部件
            update_func: 更新函数 (例如 lambda w, v: w.configure(text=v))
            *args, **kwargs: 传递给更新函数的参数
            
        返回:
            bool: 更新是否成功
        """
        if self.window_active:
            try:
                # 检查部件是否仍然存在
                widget.winfo_exists()
                update_func(widget, *args, **kwargs)
                return True
            except (tk.TclError, AttributeError, Exception) as e:
                print(f"更新部件出错: {e}")
                return False
        return False
    
    def check_window_exists(self):
        """
        检查测试窗口是否仍然存在
        
        返回:
            bool: 窗口是否存在并活跃
        """
        return self.window_active and self.test_window and self.test_window.winfo_exists()
    
    def create_standard_progress_frame(self):
        """
        创建标准的进度条框架
        
        返回:
            tuple: (progress_frame, progress, status_label)
        """
        if not self.check_window_exists():
            return None, None, None
        
        progress_frame = ttk.Frame(self.test_window)
        progress_frame.pack(fill='x', padx=20, pady=10)
        
        progress = ttk.Progressbar(progress_frame, mode='determinate', length=300)
        progress.pack(side=tk.LEFT, padx=10)
        
        status_label = ttk.Label(progress_frame, text="准备测试...", font=('Arial', 10))
        status_label.pack(side=tk.LEFT, padx=10)
        
        return progress_frame, progress, status_label
    
    def update_progress(self, progress, value):
        """
        安全地更新进度条
        
        参数:
            progress: 进度条部件
            value: 进度值
            
        返回:
            bool: 更新是否成功
        """
        return self.safe_update_widget(progress, lambda w, v: w.configure(value=v), value)
    
    def update_status(self, status_label, text):
        """
        安全地更新状态标签
        
        参数:
            status_label: 状态标签部件
            text: 要显示的文本
            
        返回:
            bool: 更新是否成功
        """
        return self.safe_update_widget(status_label, lambda w, v: w.configure(text=v), text)