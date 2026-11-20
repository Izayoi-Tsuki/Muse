# Muse

用小型自回归 Transformer 生成休闲背景音乐：学习 MIDI 中的音符事件，再通过 SoundFont 渲染为音频。

项目以《动物森友会》式轻松 BGM 为风格目标，采用 **MIDI → REMI token → Decoder-only Transformer → MIDI → WAV** 的符号音乐路线，不直接生成音频波形。本项目并非 Nintendo 官方项目。

## 试听

当前微调模型自由生成的两段样本，均从 BOS 开始，未使用真实曲目片段作为引子：

| 样本 | 音频 | MIDI | 时长 |
| --- | --- | --- | --- |
| 01 | [gen_000.wav](samples/gen_000.wav) | [gen_000.mid](samples/gen_000.mid) | 约 61 秒 |
| 02 | [gen_001.wav](samples/gen_001.wav) | [gen_001.mid](samples/gen_001.mid) | 约 60 秒 |

下载 WAV 后用本地播放器试听。音色来自 FluidSynth + GeneralUser GS，而非模型对原作音频的直接复刻。样本 02 渲染时降低了增益以保留峰值余量。

## 当前进展

- **M1：数据管线**已实现，包括数据获取、过滤、量化、调性归一、移调增广、乐器映射、去重、REMI 编码和数据集划分。
- **M2：基线训练与试听**已完成：预训练 19,020 步 + 微调 3,348 步，共 22,368 步，已有两段 MIDI/WAV 试听。
- **M3：无限播放**尚未完成。模型已有滑动上下文续写，但目前仍是按指定 token 数离线生成，没有实时播放器、流式调度或长程和声约束。
- **M4：风格调优**待继续，包括数据来源复核、分来源评估、听感迭代和长时间连贯性改善。

> 数据质量说明：当前 `animal_crossing` 是收集来源标签，不等于经过核实的动森曲目清单。BitMidi 下载脚本尚未严格筛选搜索结果标题，已发现非动森曲目被归入该来源。报告中的“337 首”应理解为该标签的入库数量，不能作为 337 首真实动森曲目的统计。当前样本对应的是这版尚待清理的数据训练结果。

## 模型与原理

MIDI 记录音高、起音时间、时值、力度和乐器。预处理将这些事件转换为离散 token，模型通过预测下一个 token 学习音符之间的统计关系。推理时逐个采样事件，再解码回 MIDI，最后由音源合成声音。

| 项目 | 当前配置 |
| --- | --- |
| 架构 | Decoder-only Transformer，RoPE，Pre-LayerNorm |
| 参数量 | 约 25.4M，输入嵌入与输出权重共享 |
| 隐藏维度 / 层数 / 注意力头 | 512 / 8 / 8 |
| FFN 隐藏维度 | 2,048 |
| 上下文长度 | 2,048 token，实际音乐时长随音符密度变化 |
| 表示 | MidiTok REMI，当前词表 388 |
| 训练目标 | 自回归 next-token 交叉熵 |
| 训练精度 | CUDA bfloat16 |
| 采样 | temperature + top-p；超过上下文后保留最近窗口 |

模型、训练和数据参数分别在 [configs/train.yaml](configs/train.yaml) 与 [configs/data.yaml](configs/data.yaml) 中配置。

## 环境准备

推荐 Python 3.11。当前训练验证环境为 Windows、RTX 4090 24GB、PyTorch 2.7.0+cu128、MidiTok 3.1.0、symusic 0.6.0。

- 训练脚本当前要求 NVIDIA CUDA GPU，默认配置按 24GB 显存设计。
- 生成脚本可退回 CPU，但会更慢。
- 当前自动工具下载与 WAV 渲染代码针对 Windows DLL 布局；Linux/macOS 用户需要自行适配 FluidSynth 路径。
- 准备 Git、网络连接和足够磁盘空间。完整数据及多轮检查点可能占用数十 GB。

在 PowerShell 中运行：

```powershell
git clone https://github.com/Izayoi-Tsuki/Muse.git
Set-Location Muse
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA 版 PyTorch 单独安装；按显卡驱动选择兼容版本。
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

# 以下是当前代码实测的两个关键库版本。
# requirements.txt 的版本范围较宽，新版本兼容性尚未验证。
python -m pip install miditok==3.1.0 symusic==0.6.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如控制台中文乱码，可在当前 PowerShell 会话设置：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 数据准备

**本仓库不分发原始训练数据或模型检查点。** 数据需从各来源自行获取，并核实相应使用权限。仓库保留本次运行的 tokenizer、词表、数据配置快照和统计报告。

```powershell
# 社区 MIDI：当前自动采集结果需要人工复核标题和来源。
python scripts/download_ac_midi.py

# POP909：目标目录须为 data/raw/pop909。
git clone --depth 1 https://github.com/music-x-lab/POP909-Dataset.git data/raw/pop909

# Lakh：Hugging Face 镜像，支持分段断点续传及解压。
python scripts/download_lakh.py
```

数据源：

- [BitMidi](https://bitmidi.com/) 与 [VGMusic](https://www.vgmusic.com/)：社区 MIDI 转录。
- [POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset)：流行钢琴编曲数据。
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)；下载脚本使用 [Hugging Face 镜像](https://huggingface.co/datasets/AghaTizi/lakh_lmd_full)。

### 预处理

```powershell
python scripts/build_dataset.py --config configs/data.yaml

# 可选：小规模验证，每个来源最多处理 15 个候选文件。
# 注意：该命令也会覆盖正式输出，请勿对需要保留的数据直接运行。
# python scripts/build_dataset.py --limit 15
```

主要处理：

1. 按时长、速度、音符数、轨道数和拍号过滤；Lakh 额外按 55–140 BPM、每秒最多 12 个音符筛选。
2. 统一为 480 ticks/quarter，每四分音符 4 格，即十六分音符分辨率。
3. 使用 Krumhansl 调性估计，大调移至 C、小调移至 a；估调属于启发式，并非准确和声分析。
4. 映射为钢琴、尼龙吉他、原声贝斯、马林巴、弦乐等小编制，保留鼓轨。
5. 对动森标签及 POP909 数据额外生成最多两个 ±3 半音范围内的移调副本。
6. 按曲目划分 train/val/test，增广副本跟随原曲；这不等同于已验证所有转录版本之间不存在泄漏。

输出目录：`data/interim/`、`data/tokenized/`、`reports/`。完整运行会重建并覆盖 tokenizer 和数据划分；已有检查点必须与训练时的 tokenizer 保持配对。

### 本次数据统计

| 来源标签 | 基础曲目数 | 时长 |
| --- | ---: | ---: |
| animal_crossing（待复核） | 337 | 13.41 小时 |
| pop909 | 906 | 62.83 小时 |
| lakh（启发式休闲过滤） | 5,000 | 223.76 小时 |
| 合计 | 6,243 | 约 300 小时 |

增广后 8,705 条序列；train/val/test 分别为 7,823 / 448 / 434 条，训练集 51,992,164 token。详情见 [数据统计报告](reports/dataset_report.md)。这些是当前数据快照，不代表完成了曲目来源或风格的人工审核。

## 训练

```powershell
python scripts/train.py --config configs/train.yaml
```

默认流程：

- 混合预训练 12 个 epoch，来源选择权重为动森标签 4.0、POP909 1.0、Lakh 0.5。
- 动森标签 + POP909 微调 4 个 epoch。
- AdamW、warmup + 余弦学习率、梯度裁剪、定期保存检查点与验证日志。

权重作用于**来源的采样概率**，不是简单把每首动森曲目复制四次。`checkpoints/` 保存模型和优化器状态；`logs/train.jsonl` 保存训练记录。

当前 `--resume` 参数虽可加载检查点，但还不是精确恢复训练进度：阶段调度会重新启动，且未完整恢复随机状态。需要严格续训时应先完善恢复逻辑。

本次预训练最后一次记录的验证损失约 1.133；微调阶段由约 1.244 升至 1.309。验证只取前 40 个 batch，不能代替全量、分来源评估或试听，也不能据此宣称微调效果更好。

## 生成与渲染

训练完成后，使用与该检查点配对的 `data/tokenized/tokenizer.json`：

```powershell
# Windows WAV 渲染需要先获取工具；只生成 MIDI 时可跳过。
python scripts/download_tools.py

# 从 BOS 自由生成，不使用原曲引子。
python scripts/generate.py --checkpoint checkpoints/finetune_final.pt --tokens 4096 --n-samples 2 --seed 0 --render

# 只输出 MIDI：去掉 --render。
# 可调采样参数：--temperature 0.95 --top-p 0.95。

# 单独渲染已有 MIDI。
python scripts/render_midi.py samples/gen_000.mid
```

输出为 `samples/gen_<seed>.mid` 和可选的 `.wav`。相同 seed 会使用相同文件名，重新运行前请保存需要保留的样本。4096 token 不对应固定时长，取决于速度、密度和事件组成。

可选 `--prompt-from-data` 会使用训练集中某条 `animal_crossing` 标签记录的前 64 token 作为引子；该标签尚待复核，且输出开头包含真实数据片段，因此不应当作完全自由生成。

FluidSynth 当前 Windows 构建可能打印缺少 `none` MIDI 输入驱动的提示；本项目离线文件播放不依赖物理 MIDI 输入，应以 WAV 是否正常生成为准。

## 项目结构

```text
Muse/
├── configs/                 # 数据与训练配置
├── src/muse/                # MIDI 处理、tokenizer、模型和统计报告
├── scripts/                 # 下载、数据构建、训练、生成和渲染入口
├── data/tokenized/          # 提交小型 tokenizer/词表/配置快照，不提交序列数据
├── reports/                 # 数据集统计报告
├── samples/                 # 两段公开试听 WAV/MIDI
├── requirements.txt
└── README.md
```

`data/raw/`、`data/interim/`、训练序列、`checkpoints/`、`logs/`、`tools/`、网络探测缓存及虚拟环境仅保留本地，不进入 Git。下载与训练脚本会按需创建目录。

## 权利与使用范围

本仓库暂未指定代码开源许可证；公开可访问并不意味着授予任意使用、修改或再分发许可。

社区转录、POP909、Lakh、FluidSynth 与 GeneralUser GS 各自的版权、许可和使用条款独立于本项目。请核实数据训练、音源使用与生成内容发布的适用权限，勿因 MIDI 可下载就默认可商用或再分发。即使未使用原曲引子，模型仍可能复现训练片段，本项目不保证生成结果的原创性或商业使用权。

《动物森友会》及相关名称和音乐的权利归各自权利人所有。
