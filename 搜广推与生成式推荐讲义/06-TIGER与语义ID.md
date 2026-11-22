# 06　TIGER 如何把一件商品找回来

[全书目录](README.md) · [推荐主线上一篇：排序](03-粗排精排与重排.md) · [扩展阅读：广告](05-广告预估竞价与预算.md) · [下一篇：OneRec](07-OneRec与工业版V1.md)

首页刷新时，系统只知道这个人之前和哪些商品有过交互。MiniOneRec 的一条 Industrial 测试记录里，这位用户最近互动的十件商品都是 3D 打印耗材，包含 PLA、TPU 和 PETG，下一条记录对应商品 **3681，橙色 1.75 mm ABS 线材**。现在遮住 3681，只给历史和商品目录，召回层要想办法把它送进候选集。第 05 章讲的竞价、预算和扣费属于广告链路。本章回到自然推荐，研究生成式模型怎样接手**候选生成**。[测试记录](../mini-lab/evidence/test-first16.csv) · [TIGER 原文](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

## 从双塔到生成式检索，请求换了一条路

双塔离线为每件商品计算向量 $v_i$ 并建索引，在线把用户历史 $H$ 编成向量 $q$，搜索内积 $q^\top v_i$ 较大的 K 件商品。近似近邻搜索（ANN）解决“怎样在时限内从大量向量中取近邻”。它的优势是物品计算可以提前完成。代价是召回分数必须适合快速检索，候选模型也只能把入围结果交给精排。先把相同 K 下的精确内积结果和 ANN 结果比较，可以区分“表示本身没把 3681 排高”和“索引近似误差漏掉了它”。[TIGER §1、图 1](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

~~~text
双塔   商品标题/行为 → 物品向量与 ANN 索引 ┐
       用户历史      → 用户向量             ┴→ 查近邻 → 候选商品 → 精排

TIGER  商品标题等    → 内容向量 → 量化 → SID ↔ 商品目录
       用户历史      → 历史商品的 SID → 预测下件商品的第 1 码
                                     → 读第 1 码，预测第 2 码 → … → 查目录 → 候选商品 → 精排
~~~

TIGER 学的是 $p(\text{下件商品的 SID}\mid H)$。把商品拆成短码后，相近商品有机会共享预测前缀，训练一件商品的前几位时也会更新其他商品共用的码词参数。能共享到什么程度，取决于商品表征与量化后的实际分布。精排继续负责在给定候选集内比较个性化收益。[TIGER §3、§4.2](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

### 中间做一次对照　SASRec 能否只靠历史顺序找对下一件

沿用第 02 章教学双塔的用户侧。把「先看红色 PLA，后看蓝色 PETG」调换次序，只要掩码、用户、设备和请求时间相同，masked mean 完全相同，用户向量和两件商品的内积分数也相同。假如最近一次材料的权重应该更大，这个用户塔天然看不见这一差别。先试 SASRec 能回答一个更小的问题：**保留对整件 SKU 打分，只改变历史表示，能救回多少请求？** 现实中双塔用户侧也可以用序列网络，这里的基线专指采用均值池化的教学实现。[第 02 章・多路召回](02-多路召回.md) · [SASRec 原论文 §III](https://arxiv.org/pdf/1808.09781)

SASRec 给每个历史 SKU 查一行物品 embedding，再加可学习的位置向量，用 causal self-attention 让第 $t$ 个位置只读取第 $1\ldots t$ 个历史。举例取一批 2 次请求，每次最多 20 件历史，示意隐层宽度 $d=64$。输入 ID 是 $[2,20]$，输出状态是 $[2,20,64]$。拿最后一个有效位置 $h_t$ 与每件商品 $i$ 的向量 $e_i$ 做内积，得到 $s_{t,i}=h_t^\top e_i$。若全目录精确打分，就是 $[2,64] \times [64,N] \rightarrow [2,N]$，取前 K 个列号再映回 SKU。也可把这些内积交给近似检索，届时需要另测索引近似误差。[SASRec §III-D](https://arxiv.org/pdf/1808.09781)

原论文把历史向左错开一位作标签，例如输入「红 PLA，白 PLA，蓝 PETG」，对应目标「白 PLA，蓝 PETG，橙 ABS」。对每个有效位置取真实下一件 $i^+$ 和一个采样的未交互物品 $j^-$，最小化 $-\log\sigma(s_{t,i^+})-\log\sigma(-s_{t,j^-})$，再对有效位置聚合。若打分为 2 和 −1，这项损失约为 $0.127+0.313=0.440$。梯度更新物品表、位置表、注意力和前馈参数。作者采用共享的输入／输出物品表。这里每个位置预测**一个完整物品**，有自注意力、有下一步监督，都不足以据此推断“模型逐码生成 SID”。[SASRec §III-D–E](https://arxiv.org/pdf/1808.09781)

| 同一请求要交 K 个候选 SKU | 模型内最后一个决定 | 离线训练及服务发布的依赖 |
|---|---|---|
| 教学双塔 + ANN | 一个用户向量匹配 $N$ 个物品向量，ANN 查近邻 | 批内对比 CE 联合更新两塔。冻结版本后导出物品向量、SKU 映射并建索引。 |
| SASRec，按整件商品检索 | 有序历史变成状态，对商品表做内积选 Top-K | 原方法按每个有效历史位置训练共享物品表与序列网络。精确打分要发布同版模型、商品表与行号映射，只有启用 ANN 才另建索引。 |
| TIGER，逐码检索 | 一条路径分多步预测一件商品的 SID，完整码回表得到候选 | 先学内容量化器并生成带区分码的目录，再用固定 SID 监督序列模型。新码本改变了旧 token 的含义时，要重新编码历史和目标并适配生成模型。 |

**归因练习。** SASRec 比均值双塔强，TIGER 又比 SASRec 强，能否直接把两段提升归为“顺序”和“生成”？**参考解答。** 原版双塔示例用批内对比 CE，SASRec 用逐位置采样 BCE，TIGER 还获得内容向量、语义码及不同的解码网络。先固定样本、用户与物品字段、可训练的物品表示、负例和目标函数，只换历史聚合器，比较均值池化与带位置编码的因果注意力。一并测交换两件有效历史前后这两个模型的输出，均值组应给出完全相同的分数。再固定生成骨干的结构、训练预算、历史窗口和解码预算，比较随机短码和语义码，检验前缀结构有无增益。最后用完整方法同目录、同 K 对比。若使用原版纯 ID 的 SASRec，单列训练中见过／未见过的目标，并增加可读相同商品内容的基线。SASRec 原文的采样负例评测不能直接与全目录 HR 对比。本节给的是对照设计，本地已有实测见第 10 章。[SASRec §IV-D](https://arxiv.org/pdf/1808.09781) · [TIGER §4.2 表 2](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) · [第 10 章](10-怎样决定替换哪个组件.md)

## 第一步　目录中的商品如何得到语义 ID

把商品标题、品牌、类别、价格等内容转成向量 $x$。TIGER 原文以 Sentence-T5 得到 768 维表示。RQ-VAE 的编码器 $E$ 再把它压到 32 维的 $z$。现在要用很少的整数表示 $z$。做法是分多次记下“最接近哪一行”，每次只让下一层处理上次还没表示好的残差。[TIGER §3.1、图 3、§4](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

令 $\ell$ 表示码本层，$C_\ell\in\mathbb R^{K_c\times d}$，$c_\ell$ 是一个行号。取 $r_0=E(x)$，每层依次执行

$$
c_\ell=\arg\min_k\|r_\ell-C_\ell[k]\|_2^2,\quad
r_{\ell+1}=r_\ell-C_\ell[c_\ell],\quad
\hat x=D\left(\sum_\ell C_\ell[c_\ell]\right).
$$

**手算一卷线材。** 为演示把 $z$ 缩成一个数 2.7。第一个码本两行是 0、2，选第 1 行（从零编号），残差变为 0.7。第二个码本两行是 0、1，选第 1 行，残差变成 −0.3。量化向量为 $2+1=3$，两位语义码为 $(1,1)$。若另一款线材被编码为同一码，目录反查会产生两个答案。论文为三层量化码另外附一个区分码，碰撞商品可分别使用后缀 0、1。第四位的作用是区分商品身份。[TIGER §3.1](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

码本本身也要训练。设 $\operatorname{sg}$ 是停止梯度，$e_\ell=C_\ell[c_\ell]$，损失为

$$
L_{\rm RQ}=\|x-\hat x\|_2^2+
\sum_\ell\left(\|\operatorname{sg}(r_\ell)-e_\ell\|_2^2+
\beta\|r_\ell-\operatorname{sg}(e_\ell)\|_2^2\right).
$$

第一项要求解码后能还原原内容向量。第二项的残差被固定，梯度把选中码字拉向残差。第三项的码字被固定，约束编码结果靠近已选码字。若所有商品总选少数几行，词表看起来大，实际可用的信息却少。检查各层码字使用率、碰撞比例，再看下一物品预测的效果，能分辨量化丢信息和推荐器没学好这两种问题。[TIGER §3.1](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

## 第二步　把用户历史变成监督样本

假设教学 batch 里只有一个用户，两件历史物品。沿用原论文每件 4 码的设置，编码器输入为 1 个用户哈希 token 加 8 个历史码，形状 $[1,9]$。记真值物品的码为 $c_1,c_2,c_3,c_4$。解码器训练时用前三个真值码右移，因果掩码让第 $j$ 个位置只能看到之前的码。以下省略填充与终止标记。

~~~text
编码器输入      [user, A1,A2,A3,A4, B1,B2,B3,B4]   → 历史状态
解码器输入      [start, c1, c2, c3]                   → 4 个预测位置
交叉熵标签      [c1,    c2, c3, c4]
逻辑 logits     [1, 4, V]                             V 为完整词表大小
~~~

损失 $-\sum_{j=1}^4\log p_\theta(c_j\mid H,c_{<j})$ 用历史预测下件商品。取 batch 维为 $B$、历史长度为 $n$，有 $[B,1+4n]$ 个输入位置和 $[B,4,V]$ 的目标 logits。RQ-VAE 的训练样本是单件商品及其内容，推荐器的样本是带时间顺序的历史和目标。量化重建好，只能说明原表示还原得好。检索质量还得在后者上测量。[TIGER §3.2、§4](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

## 第三步　推理时哪一个前缀决定 3681 能否入围

推理时真值被遮住，每扩展一步就要用当前保留的前缀重新计算。假设简化后的 3681 路径是 $(7,2)$，模型第一步给 7 的概率是 0.40，给 4 的概率是 0.50，其余共 0.10。第二步给 $p(2|7)=0.90$，$p(8|4)=0.40$。前缀 7 起步稍弱，完整路径分数却为 $0.40\times0.90=0.36$，高于 $(4,8)$ 的 0.20。宽度一开始就取 1 时，第二步已经没有 7 这个前缀。宽度 2 允许两支继续竞争。对数域下做加法可避免多个小概率相乘造成数值下溢。

~~~python
# 逻辑伪代码。一个 beam 保存一条前缀与累计 log 概率。
beams = [((), 0.0)]
for position in range(number_of_codes):
    extensions = []
    for prefix, score in beams:
        log_probs = decoder(history_states, prefix).log_softmax(dim=-1)
        for code in choices_for_this_position(position):
            extensions.append((prefix + (code,), score + log_probs[code]))
    beams = highest_scoring(extensions, beam_width)
items = [directory[sid] for sid, score in beams if sid in directory]
~~~

这里的目录过滤发生在完整路径生成后。若先用目录建一棵合法前缀树，每一步只扩展能对应商品的子节点，就成为**约束解码**。它保证搜索沿合法路径前进，但合法线材也可能得分太低。原论文讨论的是生成后过滤及增大 beam 补足有效候选，并单列无效码率。第 09 章的 MiniOneRec 实验实际使用约束解码。[TIGER §4.5](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) · [项目解码处理器](../mini-lab/upstream/LogitProcessor.py)

**对照真实样本。** MiniOneRec 目录中，3681 的三位实际码是 `<a_223><b_80><c_216>`。固定正确前缀，在每一步的合法选项中读其名次，依次为 1/48、1/66、19/21。第三步有 18 条同前缀路径必然排在它前面，所以 beam=5 会在末码丢掉这个商品。这里核验了代码执行中的一次真实失败，使用的是 MiniOneRec 的三位码与权重。TIGER 的论文设置是三位量化码加一位区分码，两套目录各自解释。[逐位排序记录](../mini-lab/evidence/target-prefix-trace.json)

## 怎样比较两条召回路线

| 想解决的问题 | 只改变什么 | 读什么证据 |
|---|---|---|
| 有结构的码能帮助预测吗 | 同类生成器与输入表征，比较随机码、LSH 码、RQ-VAE 码 | TIGER 表 2 在三个 Amazon 品类报告 RQ-VAE 优于另两种方法。这个对照测 ID 分配方式。[论文 §4.2](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) |
| 候选有用吗 | 固定相同目录、样本切分、候选 K 与同一个精排，只换召回器 | 全体和冷物品分别计 Recall@K，同时数合法、去重后不足 K 件的请求，测两路的 P95。 |
| 新上线的线材能否出现 | 固定训练边界，在上线时才给新商品做编码 | 论文冷启实验用共享前三码的候选扩展和新物品配额。复现时同时记录是否启用扩展及配额大小。[论文 §4.3](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) |

**成本落在哪里。** 双塔要存物品向量索引并更新，在线执行用户塔和索引检索。TIGER 保存码本和目录映射，在线至少串行解出一件商品的多位码。beam 越宽，每步要访问的活动前缀越多。每件物品都需要一个可查的目录条目。两种方案在相同候选数下才有可比的延迟和内存口径。[TIGER §1、附录 E](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

## 练习与参考解答

1. **按形状预测。** 3 个用户各保留 6 件历史，每件 4 个码。采用上面的用户 token 方案，编码器 token 数和目标 logits 是多少？**答**　每人 $1+6\times4=25$ 个输入位置，共 $[3,25]$，预测下一件商品时 logits 为 $[3,4,V]$。增加历史不会增加这件目标的码数。
2. **定位失败。** 某条正确码在目录中能反查，前两步第一名，第三步只排第十九。改正目录去重能否解决这条请求？**答**　发生淘汰的位置在末码竞争。先记录目标累计路径分与更宽 beam 的候选，再比较同一排名预算下的 Top‑5。扩大 beam 可能让目标被探索，但它的最终得分仍可能排在第六之后。
3. **决定下一步。** 采用新召回后，目标在 Top‑50 的覆盖率提高，首页前五件却几乎都是同款不同色。**答**　前者衡量候选能否入围，后者关乎列表组合。用同一候选集对照独立打分、带去重或多样性约束的重排。下一章进一步考察直接学习整组商品顺序的办法。[07　OneRec 与工业版 V1](07-OneRec与工业版V1.md)
