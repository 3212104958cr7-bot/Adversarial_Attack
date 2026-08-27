# 会话记录 / 续聊交接文档

> 这份文档是给下次继续对话用的。下次打开终端后，告诉 Claude
> "读一下 SESSION_NOTES.md，我们接着聊"，或者直接说想做"下一步建议"里的哪一项即可。

## 项目是什么

对 RAVEN/I-RAVEN 抽象视觉推理模型（PredRNet）做对抗鲁棒性评估：训练模型 → 白盒
PGD/C&W 攻击 → 分析攻击为何生效 → 两种轻量防御对比。完整需求见项目根目录的
`coding_agent_prompt.md`，面向 TIFS/TDSC 级别研究论文的可复现代码库。

## 当前进度状态

| Stage | 内容 | 代码 | 是否已用真实数据跑过 |
|---|---|---|---|
| 0 | 项目脚手架/目录结构 | ✅ 完成 | — |
| 1 | 训练 PredRNet | ✅ `train.py` 已写完 | ❌ 未跑（`checkpoints/` 是空的） |
| — | I-RAVEN 数据集 | ✅ `data/prepare_iraven.sh` 已写完 | ❌ 未生成（`data/i-raven/` 是空的） |
| 2 | PGD/C&W 攻击 + ASR 评估 | ✅ `attack/*.py` 已写完 | ❌ 未跑（只用合成假数据冒烟测试过） |
| 3 | 攻击效果分析 | ✅ `attack/analysis.py` 已写完 | ❌ 未跑 |
| 4 | 两种防御 + 对比 | ✅ `defense/*.py` 已写完 | ❌ 未跑 |

**结论：所有代码都已写完并且端到端跑通过（用假数据验证了不会报错），但一次都没有用真实
I-RAVEN 数据训练/攻击/防御过。`checkpoints/`、`data/i-raven/`、`logs/`、`results/`
现在都是空目录。**

## 本次会话做了什么（时间线）

1. 读取 `coding_agent_prompt.md`，确认这是合法的防御性/学术安全研究项目
2. 询问用户这一步要做到什么程度 → 用户选择"先只搭建代码骨架"（不下载数据、不训练）
3. 创建目录结构：`attack/ checkpoints/ common/ data/ defense/ logs/ models/ results/`
4. `git clone` 官方 PredRNet 仓库到 `models/predrnet/`（**原样未改动**）
5. 通读 PredRNet 源码，搞清楚：数据集 npz 格式（16张图=8上下文+8候选）、模型
   forward 签名、像素归一化方式（[0,255]→[-1,1]）、损失函数（BCE 非 softmax CE）、
   checkpoint 格式（DataParallel 的 `module.` 前缀）、7 种构型的文件夹命名
6. 编写 `common/`（数据集/模型封装，统一用 `[0,1]` 像素空间接口）
7. 编写 `train.py`（Stage 1，只保存验证集最优的一份 checkpoint，覆盖写）
8. 编写 `attack/pgd_attack.py`、`cw_attack.py`、`evaluate_attack.py`、`analysis.py`
9. 编写 `defense/preprocessing_defense.py`、`adversarial_training.py`、`evaluate_defense.py`
10. 编写 `data/prepare_iraven.sh`（I-RAVEN 生成器是 Python **2.7** 代码，脚本会建一个
    临时 conda 环境跑它）
11. 编写 `requirements.txt`、`README.md`、`.gitignore`
12. **造了一份假的合成 npz 数据**，跑通了 train→attack→analysis→defense 全流程冒烟
    测试，中途发现并修复了几个 `None` 格式化崩溃的 bug（当某个构型没有"干净预测正确"
    的样本时 ASR 会是 `None`）
13. 清理测试产物（假数据、假 checkpoint、假日志/结果、`__pycache__`），把目录恢复成
    干净的空脚手架
14. 问答环节，解释了：
    - `models/` 里每个子文件/文件夹的用途，哪些被我们的代码实际用到、哪些只是原仓库
      自带的参考资料（`main.py`/`main_ssl.py`/`checkpoint.py`/`loss.py`/
      `report_acc_regime.py`/`utils.py`/`script_*.sh`/`figures/` 都没被我们的代码用）
    - 确认 `models/predrnet/` 是官方仓库**逐字节原样** clone，`git status` 干净
    - 确认这只是模型结构代码，权重是随机初始化的，还没训练
    - 解释 checkpoint 保存机制：只保存验证集最优的一轮，覆盖写同一个文件，不保留
      每轮历史；训练过程中的参数变化本身不落盘，只有标量指标（loss/acc）记录在
      `logs/training_log.json`
    - 解释为什么从 PredRNet 仓库 clone 下来的代码不是"已训练好的模型"：GitHub 仓库
      通常只发布源码，不发布权重文件（体积大、权重和数据集强绑定）；确认了官方
      README 里也没有预训练权重下载链接

## 关键技术决策 / 设计备忘（细节见 README.md 的 "Design notes"）

- 攻防代码统一用 `[0,1]` 像素空间，`RPMModel` 内部转换成 PredRNet 需要的 `[-1,1]`
- 只扰动 8 张 context 图片，候选答案图片保持干净（更贴近真实威胁模型）
- PGD：无目标攻击，最大化对真实答案的交叉熵
- C&W：定向攻击，目标是"和正确答案像素 L2 距离最远的那个干扰项"（最坏干扰项）
- 防御评估威胁模型：预处理防御假设攻击者不自适应（对未防御模型白盒生成一次攻击，
  再过预处理）；对抗训练防御假设攻击者完全白盒（直接对微调后的权重重新生成攻击）
- 本次会话所在环境：PyTorch 是 CPU-only 编译版本（`torch.cuda.is_available()` 为
  `False`，WSL 里虽然 `nvidia-smi` 能看到 GPU，但驱动版本和这个 torch 编译版本不匹配）；
  网络访问正常（能 clone GitHub）

## 还没做 / 待你决定的事

- [ ] 之前问过是否要给 `train.py` 加 `--save-every-epoch`/`--checkpoint-interval`
      参数（保存训练中间快照）—— 还没回复
- [ ] 之前问过是否要精简 `models/predrnet/` 里用不到的参考文件（`figures/`、
      `main_ssl.py`、`data/pgm.py` 等）—— 还没回复
- [ ] 没有决定要不要 `git init` 这个项目（目前不是 git 仓库）

## 下一步建议（按顺序）

1. `bash data/prepare_iraven.sh 10000` —— 生成真实 I-RAVEN 数据集（需要 Python 2.7
   环境，约 3.5GB，耗时较长）
2. `python train.py --data-dir data/i-raven --epochs 100 ...` —— 正式训练 PredRNet
   （建议在有 CUDA 的机器上跑，否则会很慢）
3. `python attack/evaluate_attack.py ...` —— 跑真实的 PGD/C&W 攻击评估
4. `python attack/analysis.py ...` —— 生成攻击效果分析报告
5. `python defense/adversarial_training.py ...` + `python defense/evaluate_defense.py ...`
   —— 训练并评估两种防御

完整命令参数见项目根目录 `README.md`。
