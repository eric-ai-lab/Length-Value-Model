# `plot_baseline_vs_centered_exp.py` 命令说明

在 `length_control_generation` 目录下执行（或写全路径）。

```bash
cd /scr/zhenzhang/dir1/Length-Value-Model/eval_lvm/length_control_generation
python3 plot_baseline_vs_centered_exp.py [选项]
```

## 默认读什么、写什么

| 默认行为 | 说明 |
|----------|------|
| 数据源 | `length_vs_passk.csv`（baseline）+ `centered-dir` 下 `centered_exp_*.summary.json` |
| 不写回 | baseline 的 `length_vs_passk.csv` 不会被脚本覆盖 |
| 每次会覆盖 | 导出表 `baseline_vs_centered_exp_pass1.csv`（除非加 `--no-write-export-csv` 或从导出画图且未加 `--write-export-csv`） |
| 输出目录 | 默认 `centered-dir/plots/`：`baseline_vs_centered_exp.png`、`.cache.json`、导出 CSV |

### 导出 CSV 列（`baseline_vs_centered_exp_pass1.csv`）

| 列名 | baseline 行 | centered_exp 行 |
|------|-------------|-----------------|
| `series` | `baseline` | `centered_exp` |
| `token_budget` | 来自 `length_vs_passk.csv` 的 `budget` | 空 |
| `exp_scale` | 空 | 来自文件名 `centered_exp_{scale}.summary.json` 的 scale |
| `run_tag` | 空 | 如 `centered_exp_-10` |
| `avg_length` | `avg_capped_tokens` | 每 choice 平均 completion tokens |
| `pass_at_1` | pass@1（expected） | 同上 |

旧版只有 `series,run_tag,avg_length,pass_at_1` 的文件仍可 `--plot-from-export`：`exp_scale` 会从 `run_tag` 解析；无 `token_budget` 时 baseline 预算列在内存里会当作 0。

## 常用命令

```bash
# 默认：从源解析，写 PNG + JSON 缓存 + 导出 CSV
python3 plot_baseline_vs_centered_exp.py

# 指定三个目录（baseline 目录内需有 length_vs_passk.csv）
python3 plot_baseline_vs_centered_exp.py \
  --baseline-dir path/to/budget_eval_baseline_q500_n64_p1.0_topk-1_minp0.01 \
  --centered-dir path/to/results_lvm_centered_exp_3b-1.5b \
  --output-dir path/to/plots_out

# 直接指定 baseline CSV
python3 plot_baseline_vs_centered_exp.py \
  --baseline-csv /abs/path/length_vs_passk.csv \
  --centered-dir path/to/results_lvm_centered_exp_3b-1.5b

# 从源解析，但不覆盖导出 CSV（保留手改的 baseline_vs_centered_exp_pass1.csv）
python3 plot_baseline_vs_centered_exp.py --no-write-export-csv

# 用手改过的导出 CSV 画图（默认不再写回该 CSV，只更新 PNG）
python3 plot_baseline_vs_centered_exp.py --plot-from-export

# 指定导出 CSV 路径画图
python3 plot_baseline_vs_centered_exp.py --from-export-csv /path/to/baseline_vs_centered_exp_pass1.csv

# 从导出读图但仍规范化写回导出文件
python3 plot_baseline_vs_centered_exp.py --plot-from-export --write-export-csv

# 先试 JSON 缓存（快路径），命中则跳过读 CSV/summary
python3 plot_baseline_vs_centered_exp.py --use-cache

# 不写 JSON 缓存文件（仍从源读，除非用了 --from-export-csv / --plot-from-export）
python3 plot_baseline_vs_centered_exp.py --no-cache

# 指定 PNG 路径
python3 plot_baseline_vs_centered_exp.py -o /path/to/out.png

# 指定 JSON 缓存路径
python3 plot_baseline_vs_centered_exp.py --cache /path/to/baseline_vs_centered_exp_pass1.cache.json

# 查看全部参数
python3 plot_baseline_vs_centered_exp.py --help
```

## 参数一览

| 参数 | 作用 |
|------|------|
| `--baseline-dir DIR` | baseline 所在目录，读取其中 `length_vs_passk.csv`（与 `--baseline-csv` 二选一） |
| `--baseline-csv PATH` | 显式指定 `length_vs_passk.csv`，覆盖 `--baseline-dir` |
| `--centered-dir DIR` | `centered_exp_*.summary.json` 所在目录 |
| `--output-dir DIR` | PNG、JSON 缓存、导出 CSV 的根目录（默认 `centered-dir/plots`） |
| `-o` / `--output PATH` | 仅指定输出 PNG 路径 |
| `--use-cache` | 先尝试读 `.cache.json`，指纹匹配则跳过解析源数据 |
| `--cache PATH` | JSON 缓存文件路径 |
| `--no-cache` | 不写 JSON 缓存 |
| `--plot-from-export` | 从 `output-dir/baseline_vs_centered_exp_pass1.csv` 读数画图 |
| `--from-export-csv PATH` | 从给定导出 CSV 读数画图 |
| `--no-write-export-csv` | 从源解析时**不覆盖**导出 CSV |
| `--write-export-csv` | 与 `--plot-from-export` / `--from-export-csv` 联用时，仍写回导出 CSV |

## 为何改了 CSV 又被覆盖

- 若改的是 **`baseline_vs_centered_exp_pass1.csv`**：默认流程会从 `length_vs_passk.csv` 和 summary **重新计算**，并在结尾 **整文件写回** 导出表，手改会丢。请用 `--plot-from-export` 或 `--from-export-csv`，或配合 `--no-write-export-csv`。
- 若改的是 **`length_vs_passk.csv`**：脚本不会写回该文件；重新跑会用你的修改。
