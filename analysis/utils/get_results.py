import os
import re
import pandas as pd

def extract_metrics(directory):
    # 修改正则表达式，正确处理小数点
    pattern = r'eval_output_iter(\d+)_math5000\.([\d]+)_aime2024(0\.\d+)_gpqa_diamond(0\.\d+)_needlebench_atc(0\.\d+)\.jsonl'
    
    results = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.startswith('eval_output_iter'):
            print(f"处理文件: {filename}")
            try:
                match = re.match(pattern, filename)
                if match:
                    iter_num = int(match.group(1))
                    
                    # math5000格式不同，需要特殊处理
                    math5000 = float(f"0.{match.group(2)}")
                    
                    # 其他指标已经是完整的浮点数格式
                    aime2024 = float(match.group(3))
                    gpqa_diamond = float(match.group(4))
                    needlebench_atc = float(match.group(5))
                    
                    results.append({
                        'iteration': iter_num,
                        'math5000': math5000,
                        'aime2024': aime2024,
                        'gpqa_diamond': gpqa_diamond,
                        'needlebench_atc': needlebench_atc
                    })
                    print(f"  成功提取数据: iter={iter_num}, math={math5000}, aime={aime2024}, gpqa={gpqa_diamond}, needle={needlebench_atc}")
                else:
                    # 如果第一种模式匹配失败，尝试直接提取数字
                    parts = filename.split('_')
                    if len(parts) >= 6:
                        try:
                            iter_num = int(parts[2].replace('iter', ''))
                            math_part = parts[3].split('.')
                            math5000 = float(f"0.{math_part[1]}")
                            
                            aime_part = parts[4].split('diamond')
                            aime2024 = float(aime_part[0].replace('aime2024', ''))
                            
                            gpqa_part = aime_part[1].split('needlebench')
                            gpqa_diamond = float(gpqa_part[0])
                            
                            needle_part = gpqa_part[1].split('.')
                            needlebench_atc = float(needle_part[0].replace('atc', ''))
                            
                            results.append({
                                'iteration': iter_num,
                                'math5000': math5000,
                                'aime2024': aime2024,
                                'gpqa_diamond': gpqa_diamond,
                                'needlebench_atc': needlebench_atc
                            })
                            print(f"  备用方法提取数据: iter={iter_num}, math={math5000}, aime={aime2024}, gpqa={gpqa_diamond}, needle={needlebench_atc}")
                        except Exception as e:
                            print(f"  备用方法解析失败: {e}")
                    else:
                        print(f"  无法匹配文件名: {filename}")
            except Exception as e:
                print(f"  处理文件 {filename} 时出错: {e}")
    
    print(f"总共找到 {len(results)} 个文件")
    
    # 检查是否有匹配的结果
    if not results:
        print("警告：没有找到匹配的文件！")
        return None
    
    # 转换为DataFrame并按迭代次数排序
    df = pd.DataFrame(results)
    df = df.sort_values('iteration')
    
    # 保存为CSV
    output_file = 'training_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")
    
    return df

# 使用示例
directory = "orz_ckpt/debug_orz_7b_ppo_needlebench_mix_math_instruct"
df = extract_metrics(directory)
if df is not None:
    print("\n数据预览：")
    print(df.head())