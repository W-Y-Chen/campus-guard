## 这个 PR 做了什么

## 类型

- [ ] feat 新功能 / 新算子 / 新通道
- [ ] fix 缺陷修复
- [ ] docs 文档
- [ ] test 测试
- [ ] refactor 重构（不改变行为）

## 自检清单

- [ ] `python -m pytest tests -q` 全通过（70 项）
- [ ] `python run_demo.py` 正常
- [ ] 未在判定链引入大模型推理或 GPU 依赖
- [ ] 核心路径仍为零第三方依赖（不装 scikit-learn 也能跑）
- [ ] 若改变检测/进化逻辑，`python -m bench.evaluate` 后 `docs/benchmark.md` 已更新
- [ ] README / 文档未出现"零误报""100% 有效"等被口径否定的表述

## 指标影响

| 配置 | Recall | FPR（严格） | 严格口径 MS | ASR |
|---|---|---|---|---|
| 改动前 | | | | |
| 改动后 | | | | |

## 是否涉及诚实结论章节

<!-- 若引入新的局限或修复了已列缺陷，请同步更新 README 第五节/第六节 -->
