# Analysis 工具目录

该目录包含用于分析Open-Reasoner-Zero项目中各种组件的工具和脚本。

## 目录结构

- **sequence_packing**: 序列打包相关的脚本和文档，用于分析和展示序列打包技术
- **eval_analysis**: 评估和分析相关的脚本，用于分析模型输出和经验数据
- **model_training**: 模型训练相关的脚本，用于训练和微调模型
- **utils**: 实用工具脚本，提供各种辅助功能
- **model_inference**: 模型推理相关的脚本，用于使用训练好的模型生成输出
- **visualizations**: 可视化结果存放目录
- **images**: 图片文件存放目录

## 使用指南

1. 序列打包分析: 查看 `sequence_packing/` 目录下的脚本和文档
2. 评估结果分析: 查看 `eval_analysis/` 目录下的分析工具
3. 可视化结果: 查看 `visualizations/` 目录下的图片和可视化结果

## 快速入门

如果您想理解序列打包原理:
```bash
python analysis/sequence_packing/experience_visualizer.py
```

如果您想分析现有的经验数据:
```bash
jupyter notebook analysis/eval_analysis/analyze_experience_buffer.ipynb
``` 