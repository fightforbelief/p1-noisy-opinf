# 06 远程 Windows 运行手册（hanqing-win）

## 0. 总原则

- **所有代码都在远程机器 hanqing-win 上运行**。项目目录：`%USERPROFILE%\Desktop\projects\p1-noisy-opinf`。
- 远程目录中的仓库是**唯一的真实来源**。如果在本地编辑，每次运行前都要同步到远程（scp 或在远程用 git），并在 `PROGRESS.md` 中记录远程的 git commit 哈希。
- **不修改全局环境**：不升级全局 Python、pip、PyTorch、CUDA 或驱动；不使用、不修改其他项目的虚拟环境；不安装 conda、WSL 或 Docker。
- 本项目**不需要 GPU**。环境检查中要记录 GPU/CUDA/PyTorch 的状况（按负责人要求），但**不要为本项目安装 PyTorch 或 CUDA**。

## 1. 环境检查（第一步，结果写入 `reports/ENV_CHECK.md`）

通过 SSH 登录后（远程默认 shell 很可能是 PowerShell，也可能是 cmd，先确认），依次执行并记录输出：

```powershell
$PSVersionTable.PSVersion
hostname
whoami
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB, 1)
Get-PSDrive C | Select-Object Used, Free

where.exe python
where.exe py
py -0p                         # 列出已安装的 Python 版本（若有 py launcher）
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version

python -c "import torch, sys; print(torch.__version__, torch.cuda.is_available())"   # 失败也照实记录
nvidia-smi                     # 没有也照实记录
nvcc --version                 # 没有也照实记录

git --version
```

`ENV_CHECK.md` 中要写明：选择使用哪个 Python 解释器（完整路径）及理由。优先选 3.11 或 3.12；不要用 Windows Store 版的 Python stub（路径中含 `WindowsApps` 的）。

## 2. 创建项目与虚拟环境

```powershell
cd $env:USERPROFILE\Desktop\projects
if (Test-Path p1-noisy-opinf) { Write-Output "已存在，停止并报告" } else { mkdir p1-noisy-opinf }
cd p1-noisy-opinf
git init
<所选 python 的完整路径> -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip   # 只升级 venv 内的 pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip freeze > reports\pip_freeze.txt
```

- 如果 `p1-noisy-opinf` 目录已经存在，**停下来报告**，不要覆盖。
- 之后所有命令都用 `.\.venv\Scripts\python.exe` 运行，不要依赖 `activate`（SSH 会话里执行策略可能阻止脚本）。
- 若 pip 安装因网络失败，记录错误，尝试一次 `--timeout 120`；仍失败则停下来报告，**不要换用非官方镜像**，除非负责人同意。

## 3. Windows 特有的注意事项

1. **多进程**：joblib（loky 后端）在 Windows 上用 spawn 启动子进程。所有脚本必须有 `if __name__ == "__main__":` 保护；传给子进程的函数必须在模块顶层定义。
2. **BLAS 线程**：在脚本**导入 numpy 之前**设置：
   ```python
   import os
   for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
       os.environ.setdefault(v, "1")
   ```
   并行由 joblib 在进程层面完成。
3. **编码**：所有文件读写都显式 `encoding="utf-8"`；运行时设置环境变量 `PYTHONUTF8=1`（报告中有中文）。
4. **路径**：一律用 `pathlib.Path`，不要硬编码 `/` 或 `\`。路径相对于仓库根目录。
5. **换行符**：在仓库中加 `.gitattributes`，内容为 `* text=auto eol=lf`。
6. **matplotlib**：在脚本开头使用 `matplotlib.use("Agg")`（没有图形界面）。

## 4. 长时间任务

SSH 会话可能断开。超过 10 分钟的任务必须**后台运行并写日志**：

```powershell
$env:PYTHONUTF8 = "1"
Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "scripts\e2_s1_estimators.py --config configs\e2.yaml" `
  -RedirectStandardOutput "logs\e2.out.log" -RedirectStandardError "logs\e2.err.log" `
  -NoNewWindow -PassThru | Select-Object Id
```

然后用以下命令查看进度：

```powershell
Get-Content logs\e2.out.log -Tail 20
Get-Process -Id <PID> -ErrorAction SilentlyContinue
```

若 `Start-Process` 启动的进程会随 SSH 会话一起结束，则改用计划任务（`schtasks /Create /SC ONCE ...`，运行后删除该任务），并在 `PROGRESS.md` 中记录所用方法。

**断点续跑（必须实现）：** 每个实验脚本把已完成的（配置，重复）结果**逐块追加**到 `results/{exp}/runs_partial.csv`。启动时读取它并跳过已完成的块。全部完成后再合并成 `runs.csv`。这样任务中断后可以直接重启。

## 5. 日志

- 每个脚本把进度写入 `logs/{exp}.out.log`：开始时间、估算总耗时、每完成 10% 打印一次、结束时间。
- `logs/` 不提交到 git；但实验结束后，把最后 50 行日志附到 `PROGRESS.md` 对应阶段的汇报中。

## 6. Git

- 在远程仓库中提交（`git add`、`git commit`）。提交信息写清楚做了什么。
- **不要 push 到任何远程仓库**，除非负责人另行明确要求。
- `.gitignore`：`.venv/`、`data/`、`logs/`、`__pycache__/`、`*.npz`、`results/**/runs_partial.csv`。
- `results/` 中的 CSV、JSON 和 `figures/` **要提交**（体积小，便于核对）。如果单个文件 > 50 MB，先报告再决定。
