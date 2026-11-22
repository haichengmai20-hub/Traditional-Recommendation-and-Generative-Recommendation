# 09　MiniOneRec 代码与实验：一卷橙色耗材怎样掉出 Top-5

[全书目录](README.md) · [上一篇](08-OneRec-V2.md) · [下一篇](10-怎样决定替换哪个组件.md)

打开测试集第 1 行。这位用户的十件历史交互商品都是 3D 打印耗材，下一次交互的是橙色 ABS 耗材，商品编号 3681。作者发布的模型首推银色 PLA 耗材，编号 2909。两件商品的前两层语义码完全相同。我们要定位的是，橙色那一卷在哪一步丢了，以及要花力气改码本、改训练信号，还是改解码。下面的命令都从含有 `mini-lab` 的**项目根目录**运行。[这一行的输入和预测](../mini-lab/evidence/audit-first16-beam5.json) · [作者代码版本](https://github.com/AkaliKong/MiniOneRec/tree/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1)

## 1　沿着一行 CSV 找到三种 ID

| 表示 | 橙色 ABS 耗材的实际值 | 谁读取它 |
| --- | --- | --- |
| 商品编号 `item_id` | `3681` | 业务目录、传统召回、SKU 级评价 |
| 商品语义码 `item_sid` | `<a_223><b_80><c_216>` | 训练样本、前缀约束表、作者的评价器 |
| 语言模型 token ID | `151688, 151947, 152098` | 输入嵌入层和输出词表，用三个位置来预测一个商品 |

测试 CSV 的 `history_item_id`、`history_item_sid` 和 `history_item_title` 是同一个人按顺序交互过的商品，在三列中逐位对应。最后一件历史商品是蓝色 PETG 耗材，SID 为 `<a_223><b_72><c_93>`。下一次交互另存在 `item_id`、`item_sid` 和 `item_title`。推理输入到蓝色耗材为止。这一条记录支持验证“预测下一件”，曝光过哪些备选、曝光位置、竞价和预算都需要另一种日志。[测试文件](../mini-lab/evidence/test-first16.csv) · [固定版本数据与权重](https://huggingface.co/kkknight/MiniOneRec/tree/365a03fc6601ed36abd69cc9a0a59025a3d31cdc)

```text
训练样本     行为历史 → 拼 prompt → 拼真实下一件 SID → 算被监督位置的交叉熵
在线式推理   行为历史 → 拼 prompt → 逐个生成 SID token → 查目录 → 得到商品备选
离线评分     留出的真实下一件 SID ─────────────────────────→ 比对备选的命中位置
```

**先猜一下。** 一个模型能稳定生成三个合法 token，最终交给产品同学的是否一定是一件唯一的 SKU？真实目录有 3,686 件商品和 3,670 种 SID。其中 15 种 SID 发生碰撞，例如 `<a_210><b_231><c_0>` 同时对应商品 7 和 8。上游的命中以 SID 为单位。要把结果送给业务，先决定碰撞时展开哪件、展开几件、如何再排序，然后才能评价 SKU 命中。[商品目录](../mini-lab/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt) · [逐项回表](../mini-lab/run_smoke.py)

## 2　SID 从哪来：让残差决定下一层码

一个商品先有一条连续向量。作者发布的 Industrial 商品向量矩阵形状是 `[3686, 2560]`。RQ-VAE 的编码器把一行向量压到潜空间，第一个码本找一个接近它的向量，第二个码本逼近上一步剩下的差，第三个继续逼近剩余误差。三个码字向量相加，再由解码器重构原始 2,560 维输入。商品的 SID 记录每次选了哪个码字。只凭前两码相同，可以说这两件商品共享量化路径的一部分。给每个码指定“材料”“颜色”等人工语义，需要另查码下全部商品。[RQVAE](../mini-lab/upstream/rq/models/rqvae.py) · [逐层残差循环](../mini-lab/upstream/rq/models/rq.py)

**手算一个教学例子。** 暂把潜向量缩成一维 2.7。第一层码本为 `[0, 3]`，选第 1 号码字 3，残差变成 −0.3。第二层为 `[-0.4, 0.4]`，选第 0 号码字 −0.4，残差变成 0.1。第三层为 `[0, 0.125]`，选第 1 号码字 0.125。三个索引是 `[1,0,1]`，近似值为 2.725。换一件非常接近的商品，就可能三次都作出同样选择。上游最近邻分支按平方欧氏距离取最小值，在启用相应参数时也可用 Sinkhorn 分配。[量化器的距离、选码与损失](../mini-lab/upstream/rq/models/vq.py)

**本机的小码本实验。** 取发布矩阵前 64 行，构造 `[64,2560] → [64,16] → [64,3] 个索引`，每层有 16 个码字。前向计算中，重构误差推动解码结果靠近输入，码本损失让选中码字靠近编码结果，commitment loss 约束编码结果靠近码字。源码用 detach 让这两项更新各自的一侧，另外用直通估计让重构梯度传到编码器。做一次 AdamW 更新后，第一层编码器权重最大绝对变化为 0.0010012，64 件商品得到 20 种不同编码。这里真的更新了一次量化模型参数。用于后面推荐的权重仍配作者发布的码表，重新训练出的编号需要配套训练新的推荐模型。[小实验](../mini-lab/run_rq_step.py) · [实跑记录](../mini-lab/evidence/rq-step.json)

```powershell
py -3.11 mini-lab/run_rq_step.py
```

核对输出中的 `source_shape=[3686,2560]`、`completed_optimizer_steps=1` 和 `codes_after_step_shape=[64,3]`。如果要判断“加大码本能否改善推荐”，至少要在固定商品与后续训练预算下，同时量碰撞、训练集中各码的有效样本量、最终推荐命中。只把碰撞压低会遗漏稀码难学的问题。

## 3　同一行在 SFT 和推理时长什么样

`sft.py` 用 `TokenExtender` 从商品→SID 的索引收集不重复的层级 token，加入 tokenizer 并调整模型嵌入矩阵的行数。于是 `<c_216>` 可以作为一个动作学习。若替换 tokenizer 后这串字符被拆成数个普通词，词表行号、训练监督、前缀约束都会失配。实验所用 checkpoint 自带配套 tokenizer。[SFT 入口与 TokenExtender](https://github.com/AkaliKong/MiniOneRec/blob/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1/sft.py) · [权重与 tokenizer](https://huggingface.co/kkknight/MiniOneRec/tree/365a03fc6601ed36abd69cc9a0a59025a3d31cdc/Industrial_ckpt)

用测试行观察两种编码视图，不调用优化器。训练视图的 `SidSFTDataset.pre` 先把答案暂存，然后清空 prompt 中的回答字段，拼好指令和历史，最后把真实答案作为监督后缀放回。推理视图的 `EvalSidDataset(test=True)` 到回答开头就停止。两者的英文提问措辞略有不同，因此这里实测的 prompt 长度也不同。[两个 Dataset 的实现](../mini-lab/upstream/data.py)

```text
row 0 训练视图    125 个 prompt token＋3 个 SID token＋换行 198＋EOS 151645，共 130 个
input_ids        [        prompt         | 151688 151947 152098 198 151645 ]
attention_mask   [        都是 1          |    1      1      1    1    1   ]
labels           [        都是 -100       | 151688 151947 152098 198 151645 ]

row 0 推理视图    121 个 prompt token，结尾为回答标记及换行，等待模型生成首码
```

`attention_mask=1` 让模型利用历史，`labels=-100` 让这些位置跳过直接的交叉熵监督。因果语言模型内部对齐相邻位置，所以首码的损失来自“读完 prompt 时预测的下一 token 分布”。优化器也会通过生成答案所依赖的上下文更新参数。不能把“prompt 不计直接损失”理解为“模型看不见历史”。

下面按作者的左侧 padding、补到 8 的倍数，实际拼 row 0、1 的小 batch。运行前先估算，短样本中被忽略的标签会比长样本多多少？

```powershell
$env:PYTHONUTF8 = '1'
@'
import sys
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
sys.path.insert(0, "mini-lab/upstream")
from data import SidSFTDataset, EvalSidDataset

f = "mini-lab/evidence/test-first16.csv"
t = AutoTokenizer.from_pretrained("mini-lab/Industrial_ckpt")
train_view = SidSFTDataset(f, t)
eval_view = EvalSidDataset(f, t, test=True)
print("eval prompt length", len(eval_view[0]["input_ids"]))
print("gold", [x for x in train_view[0]["labels"] if x != -100])
t.pad_token, t.padding_side = t.eos_token, "left"
collate = DataCollatorForSeq2Seq(t, pad_to_multiple_of=8, return_tensors="pt")
b = collate([train_view[0], train_view[1]])
print("shape", list(b["input_ids"].shape))
print("padding", (b["attention_mask"] == 0).sum(1).tolist())
print("ignored labels", (b["labels"] == -100).sum(1).tolist())
'@ | & 'E:\cie6007_venv\Scripts\python.exe' -
```

预期输出为 `121`、`[151688,151947,152098,198,151645]`、`[2,136]`、`[6,51]`、`[131,131]`。两个原样本长度分别为 130 和 85，补齐后各保留五个有效答案标签。项目中的 SFT 还拼接两种辅助数据集。`SidItemFeatDataset` 将 SID 和标题双向翻译，`FusionSeqRecDataset` 把历史 SID 与下一件商品的文本关联起来。验证集走 `SidSFTDataset`。因此 SFT 验证损失主要读出在真实前缀下预测答案的能力，生成过程还要单独看能否保住正确路径。[训练数据拼接与 Trainer](https://github.com/AkaliKong/MiniOneRec/blob/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1/sft.py) · [数据集定义](../mini-lab/upstream/data.py)

## 4　解码：答案在目录里，为什么仍然选不中

`evaluate.py` 把实验目录中每个商品的表示写成“回答前缀＋SID＋换行”，分词后补 EOS，建立前缀到下一合法 token 的映射。首步使用回答前缀的末尾 token 查表，后续用已经生成的 token 查表。`ConstrainedLogitsProcessor` 给非法后继加负无穷。beam search 跨所有仍存活的路径，按累计 token 对数概率保留前 B 条。本次 B=5、长度惩罚为 0，评测直接指定 `do_sample=False`。这里使用的是数据集中发布的商品目录，生产上的实时可售状态仍需要单独提供。[前缀表和 generate](../mini-lab/upstream/evaluate.py) · [屏蔽非法后继](../mini-lab/upstream/LogitProcessor.py)

手算 B=2 的分叉。假设当前两条路径的对数分数为 −0.1 和 −0.2，第一条的合法后继分别增加 −0.9、−1.0，第二条的合法后继增加 −0.4、−1.3。四条路径总分为 −1.0、−1.1、−0.6、−1.5，只保留 −0.6 和 −1.0。即使正确答案就在 −1.1 那条合法路径下，它也已退出。屏蔽非法项解决“能否回到目录”，模型累计分数决定“哪几条路径获准继续”。

对橙色耗材，直接让同一个 checkpoint 沿真实前缀再走一次。每步都只在目录允许的下一码中给真实 token 排名。这是为了定位错误的 teacher forcing 诊断，部署时没有正确前缀可借用。[诊断脚本](../mini-lab/trace_target.py) · [实测 logits 排名](../mini-lab/evidence/target-prefix-trace.json)

| 已经喂给模型的正确前缀 | 想得到的下一码 | 合法兄弟中的实际排名 |
| --- | --- | --- |
| 历史和回答开头 | `<a_223>` | 1 / 48 |
| 再加 `<a_223>` | `<b_80>` | 1 / 66 |
| 再加 `<b_80>` | `<c_216>` | **19 / 21**，该步第一名是 `<c_84>` |

末码前已有 18 个同前缀、分数更高的备选，且没有同分，宽度 5 一定保不住橙色耗材。实际首选 `<a_223><b_80><c_84>` 就是银色 PLA。排查应从相近耗材的区分、历史是否足以反映材质偏好、以及码本如何分配末码入手。增加搜索宽度时应同时记录“目标路径是否存活”和“目标是否进入展示 Top-5”。两项指标分别测搜索截断与最终排序。仅凭一条用户历史，也无法断定其当时一定需要某种颜色。

## 5　decode、查商品、给分

模型生成完成后，上游先截掉输入前缀，再解码新 token、清理响应标记和首尾空白，把每个请求的五条输出放进 `predict`。评分时，`calc.py` 读取 info 文件第一列作为合法 SID 集合。目标位于第 $r$ 名且 $r\le K$ 时，该请求的 HR@K 为 1，NDCG@K 为 $1/\log_2(r+1)$，其余为 0。这里每一行只有一个目标。假设两行分别在第 2 名命中和完全未中，则 HR@5=0.5，NDCG@5≈0.3155。[评分器](../mini-lab/upstream/calc.py) · [原始预测](../mini-lab/evidence/predictions-first16-beam5.json)

本机固定测试前 16 行，包含 14 位用户和 80 个输出位置。所有输出都能回到目录，每行内部没有重复，三个输出位置的 SID 对应多个 SKU。第 12 行的自融硅橡胶胶带排首位，另 15 行都落在 Top-5 外，故 HR@5 和 NDCG@5 都是 0.0625。相同 16 行相同 SID 口径下 MostPop、ItemCF 均命中 0 行。这个小切片能验收推理入口和错误归因，模型优劣需要固定预算的完整测试。上游 `CC` 遇到正确目标后就跳出单行循环，命中之后的非法候选可能被漏计，本地审计另行扫过全部 80 个输出。[评分原文](../mini-lab/evidence/scores-first16-beam5.txt) · [生成式审计](../mini-lab/evidence/audit-first16-beam5.json) · [传统基线](../traditional-lab/industrial-results.json)

```powershell
# 读已有证据，仅这一步无需 GPU
$a = Get-Content -Raw -Encoding utf8 'mini-lab/evidence/audit-first16-beam5.json' | ConvertFrom-Json
$a.cases[0] | Select-Object target_item_id,target_sid,predicted_sids,target_rank
$a.metrics_by_k.'5'

# 重跑上游评测与金标准前缀诊断
& 'E:\cie6007_venv\Scripts\python.exe' mini-lab/run_smoke.py --rows 16 --beams 5
& 'E:\cie6007_venv\Scripts\python.exe' mini-lab/trace_target.py
```

`target_rank` 为空表示本条 Top-5 未命中。复跑使用现成的约 3.56 GB 权重；若环境更换，先核对 tokenizer、目录和 Transformers 版本，再检查最后输出是否都能回表。本地推理使用 4.46.3，上游该提交固定 4.57.1。本次第 5 步有部分无合法后继的内部路径告警，最终有效率是根据回表结果统计。[运行环境与校验值](../mini-lab/README.md)

## 6　推荐 RL 的 reward 到底奖谁

在这套代码里，奖励来自离线记录中的真实下一件商品，排序奖励按生成组里的返回位置计算。`rl.py` 的函数默认值是 `rule`，作者提供的 `rl.sh` 显式传入 `ranking`、每个 prompt 生成 16 个候选、启用 beam、关闭强插真实答案和动态采样、KL 系数 0.001。RL 的 beam 分支在固定版本里配置了 `do_sample=True`，要与上节确定性的离线评测区分。下面按这个 shell 的奖励组合讲；本机没有重训这份 RL 权重。[训练启动参数](../mini-lab/upstream/rl.sh) · [奖励实现](https://github.com/AkaliKong/MiniOneRec/blob/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1/rl.py) · [生成与更新](https://github.com/AkaliKong/MiniOneRec/blob/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1/minionerec_trainer.py)

1. **为同一个 prompt 凑成一组。** 数据集建立 prompt→历史→目标的映射，训练器得到同一 prompt 的多个完成串。若同一历史键在数据里对应多个不同目标，字典可能覆盖早期目标，移植新数据时要检查这种冲突。先用一条原始样本核对查表结果，尤其是跨 train、eval 合并映射后。
2. **逐个计算两项奖励。** `rule_reward` 对精确匹配给 1，其余 0。`ndcg_rule_reward` 看整组是否至少有一项命中。若有，正确候选拿该项的 0，排在第 $j$ 位的错误候选拿负惩罚 $-d_j/\sum_i d_i$，其中 $d_j=1/\log_2(j+1)$，$j$ 从 1 起。排得越靠前的错误扣得越多。若整组全错，第二项对整组全给 0。两个奖励函数默认等权相加。此处名为 `ndcg_rule_reward` 的函数用于训练，评测报告里的 NDCG 来自上一节的另一种计算。
3. **从组内差异得到 advantage。** Trainer 聚合每项奖励后，在同一组内减去均值、除以样本标准差加 $10^{-4}$。先按候选算 advantage，再把该候选的 advantage 用在其回答 token 上。有效回答 mask 覆盖至首个 EOS，后续 padding 跳过。脚本默认分支先按每条完成的有效长度平均，再跨样本平均，并计及参考策略 KL。

**手算一个缩成三项的训练组。** 假设返回顺序为“错误银色 PLA、正确橙色 ABS、另一个错误商品”。三个位置的折扣约为 1、0.631、0.5。位置惩罚的归一化分母是三项折扣总和 2.131。合并奖励依次约为 **−0.469、1、−0.235**，对应 advantage 约为 **−0.719、1.142、−0.422**。如果三个都错，按这两个奖励函数就得到 `[0,0,0]`，该组基于奖赏的策略梯度项为零，KL 项仍按策略和参考模型的差异计算。橙色耗材那条评测样本的五个最终备选若直接用这套规则打分，也属于整组全错；实际训练取 16 个候选，不能据此断言训练中这一组也全错。

如果增训后 HR 不动，先统计每组命中数分布、全零奖励组比例，以及真实目标是否能由目录生成。目录匹配有问题，先修映射；目标合法但几乎没有组内奖励差异，先看 SFT、候选生成与奖励覆盖；组内有信号仍难区分末码，再看是否需要更贴近细粒度偏好的监督。这些判断都能从一组候选复查。直接增加 GRPO 步数没有解决无差异组的原因。

## 7　怎样判定自己跑懂了

合上本章，用第 1 行回答三个问题：**首码如何得到监督、橙色耗材为什么在 beam=5 时必掉、这五个候选若全错时排序奖励怎样分配。** 然后换第 12 行，从 CSV 独立追到预测、目录和评分。能分别指出 SFT 的五个有效 token、末码的 19/21，以及头名命中时 HR/NDCG 对 16 行均值的贡献，就能把每个结论指到具体组件。

本机执行证据覆盖 RQ-VAE 的一次参数更新、SFT 数据拼 batch、作者 checkpoint 的单卡推荐、teacher forcing 定位与离线评分。完整 SFT 和推荐 RL 的行为来自固定版本源码。若下一阶段打算验证“RL 相对 SFT 改善了末码”，需要配套 SFT/RL 两组 checkpoint，在同一目录、测试集和解码配置上成对检验首两码命中后末码的排名变化，并记录整个系统的 HR/NDCG。[实操手册](../mini-lab/README.md) · [作者报告](https://arxiv.org/abs/2510.24431)
