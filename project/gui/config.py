import tkinter as tk
from tkinter import ttk, messagebox
import yaml

class ConfigEditor:
    def __init__(self, master, config_path, on_save_callback=None):
        self.master = master
        self.config_path = config_path
        self.on_save_callback = on_save_callback
        
        # 创建编辑器窗口
        self.window = tk.Toplevel(master)
        self.window.title("配置编辑器")
        self.window.geometry("600x500")
        self.window.grab_set()  # 使窗口为模态
        
        # 创建主框架
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建文本编辑器
        self.editor = tk.Text(main_frame, wrap=tk.NONE, font=("Courier", 10))
        self.editor.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        y_scrollbar = ttk.Scrollbar(self.editor, orient=tk.VERTICAL, command=self.editor.yview)
        x_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.editor.xview)
        self.editor.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 创建保存按钮
        save_button = ttk.Button(button_frame, text="保存配置", command=self.save_config)
        save_button.pack(side=tk.RIGHT, padx=5)
        
        # 创建取消按钮
        cancel_button = ttk.Button(button_frame, text="取消", command=self.window.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # 加载配置
        self.load_config()
        
    def load_config(self):
        """将配置从文件加载到编辑器"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()
            self.editor.delete(1.0, tk.END)
            self.editor.insert(tk.END, config_text)
        except Exception as e:
            messagebox.showerror("加载错误", f"无法加载配置文件: {str(e)}")
            self.window.destroy()
            
    def save_config(self):
        """将配置从编辑器保存到文件"""
        try:
            # 获取配置文本
            config_text = self.editor.get(1.0, tk.END)
            
            # 验证YAML
            try:
                yaml.safe_load(config_text)
            except yaml.YAMLError as e:
                messagebox.showerror("验证错误", f"YAML格式错误: {str(e)}")
                return
                
            # 保存到文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(config_text)
                
            messagebox.showinfo("保存成功", "配置已保存")
            
            # 如果提供了回调函数，则调用它
            if self.on_save_callback:
                self.on_save_callback()
                
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("保存错误", f"无法保存配置文件: {str(e)}")
    def reset_to_default_config(self):
        """重置配置为默认值"""
        try:
            # 定义默认配置的YAML文本
            default_config_yaml = """processing:
  preprocessing:
    target_size: [1024, 1024]
  denoise:
    thresholds:
      light: 500
      medium: 1000

evaluation:
  weights: [0.4, 0.6]
  brightness_threshold: 80

autofocus:
  coarse_step: 5
  thread_pool:
    use_cpu_count: true
    max_workers: null
    """
            
            # 直接将YAML文本写入配置文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(default_config_yaml)
                    
            messagebox.showinfo("重置成功", "配置已重置为默认值")
            
            # 如果提供了回调函数，则调用它
            if self.on_save_callback:
                self.on_save_callback()
                
            # 关闭编辑器窗口
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("重置错误", f"无法重置配置: {str(e)}") 