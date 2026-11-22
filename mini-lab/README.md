# MiniOneRec 实验手册

先对一件商品走完三步。把过往行为变成模型输入，生成下一件商品的语义 ID（SID），用完整目录检查这个 ID 对应什么商品。然后回头拆 SID 是怎么训练出来的，以及 SFT 和 RL 分别改变什么。[MiniOneRec 论文](https://arxiv.org/abs/2510.24431)

## 实验版本与输入

- 评测代码来自论文作者的 [AkaliKong/MiniOneRec 固定版本](https://github.com/AkaliKong/MiniOneRec/tree/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1)。`upstream/` 收录本次阅读和执行的文件，保留上游许可证。完整训练在上述仓库操作。
- 数据和推理权重来自[作者发布的固定版本](https://huggingface.co/kkknight/MiniOneRec/tree/365a03fc6601ed36abd69cc9a0a59025a3d31cdc)。本例使用 Amazon Reviews 2018 的 Industrial and Scientific 品类，推荐目标是历史交互之后的下一件商品。作者训练的 Qwen2.5-1.5B-Instruct 权重与其发布的 SID 词表、商品目录配套使用。
- 这里的运行环境是 Windows、RTX 3060 Laptop 6 GiB。推理解释器为 Python 3.11、PyTorch 2.5.1+cu121、Transformers 4.46.3。作者仓库 requirements 固定的 PyTorch 是 2.6.0、Transformers 是 4.57.1。本地采用现有 GPU 环境，因此复跑全量时还要记录依赖差异。[上游依赖](upstream/requirements.txt)

## 先读测试集第 1 条

这位用户近期交互的 10 件商品都是 3D 打印耗材。PLA 红色耗材对应 `<a_223><b_80><c_165>`，PLA 白色对应 `<a_223><b_80><c_159>`，最后一件蓝色 PETG 对应 `<a_223><b_72><c_93>`。这些商品共享第一层码，部分商品还共享第二层码。可以用它解释模型如何共享前缀；仅凭这个样本无法把某个码解释为固定材料名或颜色。[测试切片](evidence/test-first16.csv)

1. `EvalSidDataset.pre(test=True)` 只把时序历史、任务描述和回答开头写入推理输入。本机 tokenizer 给这条输入编了 **121 个 token**，末尾是 `### Response:` 及换行。真实下一件为橙色 ABS 耗材，商品 ID 3681，目标 SID 是 `<a_223><b_80><c_216>`，单独留给评分器。[上游数据类](upstream/data.py)
2. `evaluate.py` 先读商品目录，建立“已生成前缀 → 下一步可用 token”表。全目录合法首码有 **48** 种。如果第一步选中 `<a_223>`，合法第二码有 **66** 种；再选中 `<b_80>`，合法第三码还剩 **21** 种，其中一个通往真实目标商品。每步由 `ConstrainedLogitsProcessor` 屏蔽非法 token，beam search 根据模型累计分数保留路径，某条合法路径仍可能被其他高分路径挤出。五条 beam 对应下一件商品的五个备选。[评测入口](upstream/evaluate.py) · [解码处理器](upstream/LogitProcessor.py)
3. 作者的 `calc.py` 直接比较预测 SID 与真实 SID。目录里 3,686 个商品只有 3,670 个不同的 SID，15 个 SID 对应多个商品。评价“精确 SKU 命中”需要另定一对多展开、候选去重和打破平局的规则。若目标排第 3，则单条样本 HR@5=1，NDCG@5=1/log₂(3+1)=0.5。[作者评分器](upstream/calc.py) · [商品目录](Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt)

## 已跑出的推荐结果

作者 checkpoint 通过单卡完成 Industrial 测试文件前 16 行（14 位用户）的生成和评分。输入固定为文件顺序，五条 beam，共 80 个输出。逐条回查目录得到 80/80 合法 SID，行内重复 0，三个输出位置的 SID 映射到多个 SKU。上游约束处理器在解码第 5 步报告部分尝试路径没有合法后继，最终输出按目录逐个计数。上游评分器的 CC 为 0。[原始预测](evidence/predictions-first16-beam5.json) · [上游评分输出](evidence/scores-first16-beam5.txt) · [逐行审计](evidence/audit-first16-beam5.json)

| 观察 | 实际输出 | 可以得出的结论 |
| --- | --- | --- |
| 第 1 条，橙色 ABS 打印耗材 | 首选是银色 PLA 耗材，ID 2909，`<a_223><b_80><c_84>`。五个候选都以 `<a_223>` 开头，其中四个还匹配目标第二码 `<b_80>`。真实目标在 Top‑5 之外。 | 本例已选到有意义的商品邻域；最终 SKU 的选择仍可能出错。 |
| 第 12 条，自融硅橡胶胶带 | 模型在第 1 位产生真实目标 `<a_202><b_198><c_80>`。 | 本次切片包含一次准确命中。 |
| 前 16 条整体 | HR@1、@3、@5 均为 1/16；对应三个 NDCG 均为 0.0625。 | 这是生成和评测验收。固定小切片、同一用户多行且 beam=5，全量、随机样本和其他 beam 的结果需要各自测量。 |

**橙色 ABS 到底丢在哪一步？** 对同一权重做了一次 teacher forcing。依次喂入正确前缀；只在目录允许的下一码之间，检查目标 token 的 logit 排名。首码为第 **1/48**，给定首码后第二码为第 **1/66**，给定前两码后末码为第 **19/21**，三步均无同分。仅同前缀内就有 18 个末码在真实目标前面，beam=5 到这里一定会丢掉它。这个观察能区分“目录里没有橙色耗材”和“模型给了正确末码较低的分数”。增加 beam 可以检验搜索宽度的影响，更大的 beam 是否把目标排入最终展示前五仍需重测。[真实前缀排序](evidence/target-prefix-trace.json) · [诊断脚本](trace_target.py)

## 本机怎样复跑

在当前项目根目录运行以下 PowerShell 命令。推理命令会重新截取测试前 16 行，调用原版 `evaluate.py`、`calc.py`，另行逐条检查预测 SID 的合法性、列表内重复及与目标行的对应关系。固定 row 0–15、batch=1、beam=5、length penalty=0、最多 8 个新 token。上游的 `CC` 计数器会在命中时提前跳出，每个候选的合法性看本地额外的全列表检查。[复跑脚本](run_smoke.py)

```powershell
& 'E:\cie6007_venv\Scripts\python.exe' 'mini-lab/run_smoke.py' --rows 16 --beams 5
& 'E:\cie6007_venv\Scripts\python.exe' 'mini-lab/trace_target.py'
py -3.11 mini-lab/run_rq_step.py
```

换机器时，推理解释器需要 GPU 可用的 PyTorch、Transformers、Accelerate、pandas、NumPy、tqdm、Fire 和 bitsandbytes，RQ 小实验另需 scikit-learn。本机的 Fire、termcolor、bitsandbytes 位于实验目录 `.infer-deps`，复跑脚本自动加入它们的导入路径。核验或断点补下权重可运行 `py -3.11 mini-lab/fetch_weights.py`。它只针对上述 Industrial 权重，按作者发布文件的 SHA-256 `57531f3d80db89634368829980fbf077938e55a2c5c3f028d943e31e0bf86b55` 校验 3,555,990,368 字节的模型。[权重目录](https://huggingface.co/kkknight/MiniOneRec/tree/365a03fc6601ed36abd69cc9a0a59025a3d31cdc/Industrial_ckpt)

### 单独验证 RQ-VAE 如何产生 SID

作者提供的矩阵为 3,686 行 × 2,560 维。取前 64 行输入作者原始 RQVAE，使用三个大小为 16 的小码本和随机种子 42，在 CPU 上真实执行一次 AdamW 更新。第一层编码器权重最大变化 0.0010012，输出索引形状为 64×3。更新后第 0 行取码 `[15, 1, 1]`，64 件商品中仅 20 种不同编码。这个数字回答“压缩后还能否唯一区分商品”。小实验的重构误差和码本用于观察机制，推理分支仍读取作者训练权重所对应的 SID。训练时间、码本容量和初始化都影响冲突率，单次结果不能归因给其中一个因素。[实验代码](run_rq_step.py) · [实跑记录](evidence/rq-step.json) · [残差量化循环](upstream/rq/models/rq.py)

## 从会用 checkpoint 到能训练项目

| 顺序 | 在作者仓库操作 | 输入、产物与检查点 |
| --- | --- | --- |
| 1　先确定数据 | 阅读 `data/amazon18_data_process.py` 和 `convert_dataset.py`。直接使用同一发布版的 train/valid/test CSV 与 index、info 文件。 | 随机抽商品同时查 `item_id`、`item_sid` 与 info 表。标注目标是下一次交互，数据不含所有被展示但未交互的商品。 |
| 2　复建 SID | 在 `rq/text2emb/` 配好文本 encoder 路径执行 `amazon_text2emb.sh`。在 `rq/` 执行 `rqvae.sh`，然后编辑 `generate_indices.py` 的 checkpoint、设备与输出路径并运行。 | 输入标题与描述，得到每件商品的内容向量，经编码器与逐层残差量化得到离散码，再生成 item→SID。检查重构、各层字典利用率、冲突和历史记录的映射覆盖；更换码本必须同步重建训练数据、约束目录和模型 SID 词表。 |
| 3　SFT | 读 `sft.py` 的 TokenExtender 与数据集拼接。修改 `sft.sh` 内的基座路径、码表、元数据、输出目录和 GPU 数，运行 `bash sft.sh`。 | 在 Qwen tokenizer 中加入 SID token。学会“历史→下一 SID”，并利用自然语言与 SID 的辅助任务对齐。检查一次 batch 的输入、被 mask 的 prompt、有效监督 token 与新词嵌入，存 checkpoint 与 tokenizer，并留 SFT 结果作基线。 |
| 4　推荐 RL | 将 `rl.sh` 的 `model_path` 指向 SFT 产物，核对 `config/zero2_opt.yaml` 和并行度，运行 `bash rl.sh`。 | 默认 shell 示例为 `reward_type=ranking`、`num_generations=16`、`beam_search=True`。追踪一个历史的候选组、目标在第几名、各项奖励、组内差异与 advantage。观察组内全错/全同奖的比例，确定奖励是否给出学习信号。 |
| 5　同口径评测 | 全量直接给 `evaluate.py` 传同域 test CSV、info、SFT 或 RL 模型路径；大型运行按硬件改 `evaluate.sh` 的分片与卡数，再用 `calc.py`。 | 固定目录、测试行、K、beam、重复处理、超参数与测量设备。两组均记 HR/NDCG、非法率、唯一推荐、长尾覆盖、延迟。报告评测行数、有无跳过、是 SID 还是 SKU。对同一请求成对比较时，保留输出差异最大的失败样本。 |

这些 shell 脚本在固定版本里把模型路径与卡数写在文件内部。训练脚本默认 8 进程，evaluate.sh 默认卡号 0–7 并设 beam=50；直接在脚本末尾追加 README 示例参数不能覆盖脚本里硬编码的配置。作者的完整训练配置面向 4–8 张 A100/H100 量级，单卡推理可直接走本机命令。此处的实际参数更新限于上面的 RQ-VAE 一步，完整 SFT、GRPO、在线用户收益属于后续实验。[训练脚本](upstream/sft.sh) · [RL 脚本](upstream/rl.sh) · [评测 shell](upstream/evaluate.sh) · [作者的完整入口](https://github.com/AkaliKong/MiniOneRec/tree/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1)
