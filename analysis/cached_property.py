from functools import cached_property
import time

class DataProcessor:
    def __init__(self, data_source):
        self.data_source = data_source
        self.process_count = 0
    
    # 使用cached_property的版本
    @cached_property
    def processed_data(self):
        print("开始处理数据...")
        self.process_count += 1
        time.sleep(2)  # 模拟耗时操作
        print(f"数据处理完成，这是第{self.process_count}次处理")
        return [x * 2 for x in self.data_source]
    
    # 不使用cached_property的版本
    def processed_data_no_cache(self):
        print("开始处理数据...")
        self.process_count += 1
        time.sleep(2)  # 模拟耗时操作
        print(f"数据处理完成，这是第{self.process_count}次处理")
        return [x * 2 for x in self.data_source]


# 使用示例
data = [1, 2, 3, 4, 5]
processor = DataProcessor(data)

# 使用cached_property
print("\n使用cached_property:")
start = time.time()
print(processor.processed_data[:3])  # 第一次调用，会执行处理
print(f"耗时: {time.time() - start:.2f}秒")

start = time.time()
print(processor.processed_data[:3])  # 第二次调用，直接返回缓存结果
print(f"耗时: {time.time() - start:.2f}秒")

# 不使用cached_property
print("\n不使用cached_property:")
start = time.time()
print(processor.processed_data_no_cache()[:3])  # 第一次调用
print(f"耗时: {time.time() - start:.2f}秒")

start = time.time()
print(processor.processed_data_no_cache()[:3])  # 第二次调用，重新执行
print(f"耗时: {time.time() - start:.2f}秒")

# (oc-040-vllm-072) root@di-20250312114317-s565s /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero ➜ git:(main) [03-25 22:05:45] python analysis/cached_property.py       

# 使用cached_property:
# 开始处理数据...
# 数据处理完成，这是第1次处理
# [2, 4, 6]
# 耗时: 2.00秒
# [2, 4, 6]
# 耗时: 0.00秒

# 不使用cached_property:
# 开始处理数据...
# 数据处理完成，这是第2次处理
# [2, 4, 6]
# 耗时: 2.00秒
# 开始处理数据...
# 数据处理完成，这是第3次处理
# [2, 4, 6]
# 耗时: 2.00秒
# (oc-040-vllm-072) root@di-20250312114317-s565s /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero ➜ git:(main) [03-25 22:06:00] 