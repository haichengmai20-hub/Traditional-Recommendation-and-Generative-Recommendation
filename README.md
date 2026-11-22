# 搜广推与生成式推荐

同一件商品，走完传统召回 → 粗排精排 → 搜索 → 广告竞价 → 生成式推荐（TIGER / OneRec / MiniOneRec）的全程对照学习包。所有实验用的是同一份 Amazon Reviews 2018 Industrial and Scientific 数据、同一批测试请求，便于逐层对比"候选是在哪一步丢的"。

入口：[00-学习路线.md](00-学习路线.md)（六个学习单元 + 每单元应交的产物）。

## 目录

| 路径 | 内容 |
|---|---|
| `00-学习路线.md` | 学习顺序、每单元的阅读与动手项、按卡点选小红书笔记 |
| `01-传统搜广推链路与项目.md` | 传统链路：请求 → 样本 → 多路召回 → 粗精排 → 重排；搜索与广告两条支线 |
| `02-生成式推荐论文路线.md` | TIGER / OneRec / OneRec-V2 的论文路线与对照点 |
| `03-小红书路线笔记导读.md` | 相关笔记的逐图导读与适用阶段 |
| `搜广推与生成式推荐讲义/` | 主教材 11 篇（01 请求与样本 → 10 怎样决定替换哪个组件），按 01→02→03→06→07→08→09→10 顺序读，04/05 按需查 |
| `traditional-lab/` | 传统基线实验：热门、ItemCF、ItemCF+标题召回；逐请求结果与名次 |
| `mini-lab/` | MiniOneRec 最小可跑实验：作者发布权重推理、逐码 prefix trace、回表核对；`upstream/` 收录上游固定版本代码与许可证 |
| `小红书核对记录/` | 网页浏览记录（jsonl）与逐图 OCR 索引 |
| `VSCode-Codex-本次对话.md` | 本次学习的对话记录存档 |

## 实验口径

- 数据：Amazon Reviews 2018 · Industrial and Scientific；同一份切分与同一批测试行贯穿传统与生成式对照。
- MiniOneRec 评测代码固定在上游 commit `0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`，权重与 SID 词表固定在上游发布的 Hugging Face 版本；运行环境与依赖差异见 `mini-lab/README.md`。
- 文中所有数字都标注了来源文件（如 `traditional-lab/industrial-results.json`、`mini-lab/evidence/target-prefix-trace.json`）；未跑过的结论明确标为待验证。

## 不入库的内容

模型权重（`mini-lab/Industrial_ckpt/`，约 3.4 GB）与推理依赖二进制（`mini-lab/.infer-deps/`）体积过大，已在 `.gitignore` 排除。需要复跑时按 `mini-lab/fetch_weights.py` 的说明获取对应版本。
