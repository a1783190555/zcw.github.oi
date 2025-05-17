import concurrent.futures
import multiprocessing,time
from core.focus_eval import FocusEvaluator
from test.performance_test import PREPROCESSING_SIZE

class AutoFocus:
    def __init__(self, evaluator, coarse_step=5, use_cpu_count=True, max_workers=None, target_size=PREPROCESSING_SIZE):
        self.evaluator = evaluator
        
        # 设置默认值
        self.coarse_step = coarse_step
        self.target_size = target_size
        self.use_cpu_count = use_cpu_count
        self.max_workers = max_workers
        
        # if config:
        #     # 提取预处理配置
        #     if 'processing' in config and 'preprocessing' in config['processing']:
        #         preproc_config = config['processing']['preprocessing']
        #         if 'target_size' in preproc_config:
        #             self.target_size = tuple(preproc_config['target_size'])
        
        # 调试输出
        print(f"AutoFocus配置: coarse_step={self.coarse_step}, use_cpu_count={self.use_cpu_count}, max_workers={self.max_workers}")

    def coarse_search(self, frames):
        max_score = float('-inf')
        best_pos = 0
        print(f"开始粗搜索，使用步长: {self.coarse_step}")
        # 准备要处理的帧索引
        frame_indices = list(range(0, len(frames), self.coarse_step))
        frame_scores = {}#储存评分
        # 确定线程数，留出至少一个核心给UI线程
        if self.use_cpu_count:
            n_cores = max(1, multiprocessing.cpu_count() - 1)  #保留一个核心
        else:
            n_cores = self.max_workers if self.max_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        print(f"粗搜索使用 {n_cores} 个线程进行处理")
        # 使用线程池并行处理
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = {}
            # 提交所有任务
            for i in frame_indices:
                futures[executor.submit(self._evaluate_frame, frames[i])] = i
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(futures):
                frame_idx = futures[future]
                try:
                    score = future.result()
                    if score is not None:
                        frame_scores[frame_idx] = score
                        # 更新最大分数
                        if score > max_score:
                            max_score = score
                            best_pos = frame_idx
                except Exception as e:
                    print(f"Error processing frame {frame_idx} in coarse search: {e}")
        
        end_time = time.time()
        print(f"粗搜索完成，耗时: {end_time - start_time:.4f}秒，检查了 {len(frame_indices)} 帧")
        
        # 返回搜索范围和收集的帧评分
        start = max(0, best_pos - self.coarse_step)
        end = min(len(frames) - 1, best_pos + self.coarse_step)
        print(f"粗搜索确定的范围: {start} 到 {end}")
        return start, end, frame_scores  

    def fine_search(self, frames, start, end):
        """仅返回最佳帧的精细搜索"""
        best_frame, _ = self.fine_search_with_index(frames, start, end)
        return best_frame
        
    def fine_search_with_index(self, frames, start, end):
        """精细搜索，返回最佳帧及其索引，以及评分信息"""
        start_time = time.time()
        print(f"开始精搜索(带索引)，范围从 {start} 到 {end}")
        frame_scores = {}
        # 根据配置确定工作线程数
        if self.use_cpu_count:
            n_cores = multiprocessing.cpu_count()
        else:
            n_cores = self.max_workers if self.max_workers is not None else multiprocessing.cpu_count()
        print(f"精搜索(带索引)使用 {n_cores} 个线程进行处理")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = {}
            for i in range(start, end + 1):
                futures[executor.submit(self._evaluate_frame, frames[i])] = i
            for future in concurrent.futures.as_completed(futures):
                frame_idx = futures[future]
                try:
                    score = future.result()
                    if score is not None:
                        frame_scores[frame_idx] = score
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
        if not frame_scores:
            end_time = time.time()
            print(f"精搜索(带索引)未找到有效结果，返回起始帧。耗时: {end_time - start_time:.4f}秒")
            return frames[start], start, {}  # 返回空的评分字典
        # 直接返回得分最高的帧和索引
        best_frame_idx = max(frame_scores.items(), key=lambda x: x[1])[0]
        print(best_frame_idx)
        end_time = time.time()
        print(f"精搜索(带索引)完成，最佳帧索引: {best_frame_idx}，耗时: {end_time - start_time:.4f}秒")
        return frames[best_frame_idx], best_frame_idx, frame_scores  # 额外返回帧评分信息
    
    def _evaluate_frame(self, frame):
        """简单的帧评估方法"""
        try:
            processed = FocusEvaluator.preprocess_image(frame, target_size=self.target_size)
            return self.evaluator(processed)
        except Exception as e:
            print(f"Frame evaluation error: {e}")
            return None
        
