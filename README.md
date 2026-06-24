# KPL 电竞数据分析系统

王者荣耀职业联赛 (KPL) 数据驱动的比赛分析与胜负预测系统，面向战队数据分析师场景构建。

## 功能模块

| 模块 | 功能 |
|------|------|
| **赛季总览** | 战队实力排行、英雄热度榜 |
| **BP 模拟器** | 基于 LightGBM 模型的阵容胜负预测、协同/克制分析 |
| **选手分析** | 个人雷达图、英雄池、赛季表现趋势、近期比赛详情 |
| **选手对比** | 两名选手的多维度数据并排对比 |
| **战队分析** | 队伍统计、队员构成、常用英雄体系、近期战绩 |
| **英雄分析** | 赛季胜率趋势图、最佳搭档、克制/被克制关系 |

## 技术架构

```
数据采集 (Scraper)
    ↓
MySQL 数据库 (13张表)
    ↓
特征工程 (Python/Pandas)  →  LightGBM 预测模型
    ↓
Flask REST API
    ↓
Web 面板 (Chart.js 可视化)
```

## 技术栈

- **后端**: Python / Flask / PyMySQL / DBUtils
- **机器学习**: LightGBM / scikit-learn / Pandas / NumPy
- **前端**: Vanilla JS / Chart.js / 响应式 CSS
- **数据库**: MySQL 8.0
- **数据源**: KPL 官方 API (pvp.qq.com)

## 快速启动

### 环境要求

- Python 3.9+
- MySQL 8.0+
- 已导入的 KPL 比赛数据

### 安装

```bash
cd kpl
pip install -r requirements.txt
```

### 配置

```bash
cp config.py config_local.py
# 编辑 config_local.py 填入你的数据库密码
```

### 启动

```bash
python dashboard/app.py
# 访问 http://localhost:5000
```

## 项目结构

```
kpl/
├── config.py                  # 默认配置
├── config.local.py            # 本地配置 (不提交Git)
├── database.py                # 数据库连接池与查询
├── dashboard/
│   ├── app.py                 # Flask 应用
│   ├── templates/
│   │   └── index.html         # 页面模板
│   └── static/
│       ├── style.css           # 样式
│       └── app.js              # 前端逻辑与图表
├── scripts/                   # 待整理
│   ├── importer.py            # 数据导入ETL
│   ├── scraper.py             # 数据爬取
│   └── v3_model.py            # 模型训练
├── schema.sql                 # 数据库建表
├── raw_data/                  # 原始JSON数据
├── bp_model.txt               # 训练好的预测模型
├── bp_model_features.json     # 模型特征列表
└── requirements.txt
```

## 模型说明

BP 预测模型基于 LightGBM，输入特征包括：

- **英雄强度**: 各英雄的历史胜率
- **阵容协同**: 10对英雄组合的历史协同胜率
- **克制关系**: 25对跨阵容英雄克制胜率
- **队伍实力**: 队伍的历史胜率

模型训练代码见 `v3_model.py`，使用时间序列交叉验证避免未来信息泄露。
