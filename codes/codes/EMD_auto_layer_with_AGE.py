# -*- coding: utf-8 -*-
"""
EMD_auto_layer_with_AGE.py
==========================
基于经验模态分解(EMD)的测井曲线自动分层与可视化处理方法（专利核心逻辑保留）
+ 【新增天文调谐模块】（astrochron timeOpt/eha 的 Python 重写，见 astro_py_tuner.py）。

=====================================================================
【保留的原专利功能】（源自 EMD分解.py / zl4~zl7&8.R，核心逻辑未改动）
  S1   读取测井数据（10井测井数据.xlsx, Sheet2=[深度, GR]；Sheet1=岩性表）
  S3~S8 EMD 分解（PyEMD，max_imf=10；失败时自动切换内置纯 numpy 回退 EMD）
  S9A  各 IMF/残差独立 CSV + all_components.xlsx 多表页汇总
  S9B  图2 IMF 堆叠图、图3 残差趋势项图、图4 方差贡献率柱状图(ylim 自适应 max*1.15, 柱顶标数)、
       图5 IMF+岩性纹理叠合图、图6 IMF1-5 极值密度分面图(3m 滑窗)、
       图7 原始 vs 重构对比图（报告重构误差）
  S8.5 残差趋势项一阶差分符号变化 → 极值界面 → 合并 < MIN_THICKNESS_M 的薄层

【新增天文调谐模块】
  S8.6 深度-年代转换节点：
       (1) 调谐信号构建（默认 auto：SPEC 定义“末2个IMF+残差”与全重构两路
           并行评估，取匹配度高者并记录日志——真实井数据上天文信号常分布于
           中低阶 IMF，auto 模式保证鲁棒）；
       (2) AstroTuner 扫描沉积速率 SR（multiprocessing 2 核加速）；
       (3) best_sr 与 match_score（∈[0,1]）；
       (4) match_score < MATCH_THRESHOLD → 醒目 WARNING，跳过年代赋值，
           仅输出深度分层（鲁棒性降级路径）；
       (5) 否则构建深度-年代模型（无锚点：顶界相对 0 Ma，Age 随深度单调递增），
           为每个界面/地层赋 Top_Age / Base_Age。
  图1  升级版方法流程图（原 S1-S9B + 新增 S8.6 深度-年代转换节点）
  图8  升级版综合分层成果图：GR 曲线 + 界面竖线/倒三角 + 顶部辅助 x 轴 Age(Ma)
  图9  SR 扫描目标函数曲线（标注最优 SR 与匹配度阈值线）
  图10 调谐后 MTM 频谱 + 理论天文周期竖线(405/95/40/21 kyr)匹配图
  图11 eha 演化谱（深度 × 频率 × 归一化功率）
  图12 深度-年代模型曲线（Age vs Depth，标注界面点；
       TUNING_405 匹配成功时升级为 405 kyr 锚定分段线性非线性模型，
       叠加锚点散点并与旧恒定 SR 线性模型虚线对比）
  图13 405 kyr 锚定诊断图（上：所选 IMF 曲线 + 锚点刻度线；
       下：锚点区间沉积速率 interval_sr 阶梯图）

【自检模式】
  python EMD_auto_layer_with_AGE.py --selftest
  用纯白噪声信号跑一遍 AstroTuner → match_score 低 → 醒目警告 →
  仅输出深度分层（演示鲁棒性降级路径），结果写入 OUTPUT_DIR/selftest_result/。

运行：
  python EMD_auto_layer_with_AGE.py            # 全流程
  python EMD_auto_layer_with_AGE.py --selftest # 白噪声自检
"""

# ======================================================================
# 1. 自动依赖检查安装块
# ======================================================================
import os
import sys
import subprocess
import warnings

def _pip_install(pkg):
    print(f"[依赖检查] 正在安装 {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                          stdout=subprocess.DEVNULL)

def _ensure(pkg_import, pkg_pip=None):
    try:
        __import__(pkg_import)
        return True
    except ImportError:
        try:
            _pip_install(pkg_pip or pkg_import)
            __import__(pkg_import)
            return True
        except Exception as exc:
            print(f"[依赖检查] {pkg_import} 安装失败: {exc}")
            return False

for _mod, _pip in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                   ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl")]:
    if not _ensure(_mod, _pip):
        raise RuntimeError(f"核心依赖 {_mod} 不可用且自动安装失败，请手动安装。")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from scipy import interpolate

# ---- PyEMD（emd-signal）；失败 → 内置纯 numpy 回退 EMD ----
USE_PYEMD = False
try:
    from PyEMD import EMD as _PyEMD
    USE_PYEMD = True
except ImportError:
    if _ensure("PyEMD", "emd-signal"):
        from PyEMD import EMD as _PyEMD
        USE_PYEMD = True

if not USE_PYEMD:
    print("[依赖检查] PyEMD 不可用，启用内置纯 numpy EMD 回退实现。")


def _pyemd_version():
    """run_log 依赖版本记录用：emd-signal 发行版版本（无元数据时兜底）。"""
    if not USE_PYEMD:
        return 'N/A(内置回退EMD)'
    try:
        import importlib.metadata as _im
        return _im.version('emd-signal')
    except Exception:
        return 'unknown'


# ======================================================================
# 内置纯 numpy EMD 回退实现（极值 → 三次样条包络 → 均值 → 停止准则）
# 仅在 PyEMD 不可用时使用；接口与 PyEMD.emd 对齐：
#   emd(signal, max_imf) -> 2D array（最后一行为残差）
# ======================================================================
class _FallbackEMD:
    """纯 numpy(+scipy 样条) 的 EMD 筛分回退实现。"""

    def __init__(self, sd_thresh=0.2, max_sift=50):
        self.sd_thresh = sd_thresh
        self.max_sift = max_sift

    @staticmethod
    def _extrema_idx(s):
        """局部极大/极小值索引（含端点，保证包络覆盖全段）。"""
        n = s.size
        d = np.diff(s)
        maxima = np.where((d[:-1] >= 0) & (d[1:] < 0))[0] + 1
        minima = np.where((d[:-1] <= 0) & (d[1:] > 0))[0] + 1
        maxima = np.concatenate(([0], maxima, [n - 1]))
        minima = np.concatenate(([0], minima, [n - 1]))
        return np.unique(maxima), np.unique(minima)

    def _sift(self, s):
        """对信号 s 执行筛分，返回一个候选 IMF。"""
        h = s.astype(float).copy()
        x = np.arange(h.size)
        for _ in range(self.max_sift):
            ma, mi = self._extrema_idx(h)
            if ma.size < 3 or mi.size < 3:
                break
            up = interpolate.CubicSpline(x[ma], h[ma])(x)
            lo = interpolate.CubicSpline(x[mi], h[mi])(x)
            m = 0.5 * (up + lo)
            h_prev = h
            h = h - m
            sd = np.sum((h_prev - h) ** 2 / (h_prev ** 2 + 1e-12))
            if sd < self.sd_thresh:
                break
        return h

    def emd(self, signal, max_imf=10):
        s = np.asarray(signal, dtype=float)
        res = s.copy()
        imfs = []
        for _ in range(max_imf):
            ma, mi = self._extrema_idx(res)
            # 停止准则：残差近似单调（内部极值点不足）则终止分解
            if ma.size <= 3 or mi.size <= 3:
                break
            imf = self._sift(res)
            imfs.append(imf)
            res = res - imf
        imfs.append(res)  # 最后一行为残差（与 PyEMD 用法对齐）
        return np.vstack(imfs)


# ======================================================================
# 2. CONFIG 参数区
# ======================================================================
CONFIG = dict(
    PROJECT_DIR=os.path.dirname(os.path.abspath(__file__)),
    INPUT_XLSX="synthetic_benchmark_well10_regenerated.xlsx",          # 首选输入文件；不存在则尝试备选名
    INPUT_XLSX_ALT="synthetic_benchmark_well10.xlsx",  # 备选输入文件（论文 §8 交付名）
    OUTPUT_DIR=None,                          # None → <PROJECT_DIR>/emd_astro_result
    # ---- 天文调谐参数 ----
    TARGET_PERIODS_KYR=[405, 95, 40, 21],     # 米氏周期 (kyr)
    SR_RANGE=(1.0, 20.0, 0.25),               # SR 扫描范围/步长 (m/Myr)
    ENABLE_AGE_MODULE=True,
    MATCH_THRESHOLD=0.5,                      # 匹配度阈值：低于则跳过年代赋值
    RESAMPLE_DT=0.1,                          # 重采样间隔 (m)
    N_JOBS=2,                                 # SR 扫描并行进程数
    ANCHOR_DEPTH=None,                        # 锚点深度 (m)，None → 顶界相对 0 Ma
    ANCHOR_AGE_MA=None,
    # 调谐信号构建模式:
    #   'low2_res'  = SPEC 定义：末 2 个 IMF + 残差趋势项（去均值）
    #   'full_recon'= 全 IMF + 残差重构（等价于原始信号去均值）
    #   'auto'      = 两路并行评估，取 match_score 高者（默认，鲁棒）
    TUNING_SIGNAL_MODE="auto",
    # ---- 405 kyr 长偏心率锚定非线性（分段线性）深度-年代模型 ----
    TUNING_405=True,                          # 启用 405 kyr 锚定年龄模型
    CYCLE_405_TOLERANCE=0.15,                 # IMF 周期数与 Ne 的允许相对偏差
    ANCHOR_PHASE="trough",                    # 锚点相位：'trough'(谷→谷) 或 'peak'(峰→峰)
    # ---- EMD / 分层参数（专利实施例）----
    MAX_IMF=10,
    MIN_THICKNESS_M=1.0,
    # ---- 代理数据参数（输入文件存在时忽略）----
    PROXY_SEED=6,
    PROXY_SR_TRUE=15.0,                       # 内嵌真值沉积速率 (m/Myr)
    EHA_WIN_M=150, EHA_STEP_M=10,
    FIG_DPI=300,
)
CONFIG["OUTPUT_DIR"] = CONFIG["OUTPUT_DIR"] or os.path.join(
    CONFIG["PROJECT_DIR"], "emd_astro_result")

# 中文字体与负号（本机 Noto CJK 以 ttc 集注册为 JP 字面，亦完整覆盖简体中文字形）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                   "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMF_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83',
              '#1B5E20', '#E65100', '#4A148C', '#006064', '#3E2723']


# ======================================================================
# 3. 日志（同时写屏幕与 run_log.txt）
# ======================================================================
class TeeLogger:
    def __init__(self, path):
        self.path = path
        self._fh = None
        self._console = sys.stdout   # 保存真实控制台，避免重定向后自递归

    def start(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, msg):
        self._console.write(msg)
        if self._fh:
            self._fh.write(msg)
            self._fh.flush()

    def flush(self):
        self._console.flush()
        if self._fh:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def banner(msg):
    line = "=" * 68
    print(f"\n{line}\n{msg}\n{line}")


# ======================================================================
# 4. 实施例代理数据生成（SPEC §1）
#    深度 1432.0 m 起，0.1 m 采样，10084 点（末深 2440.3 m）；
#    SR_true = 15 m/Myr，内嵌米氏周期 405/95/40/21 kyr
#    → 深度域波长 6.075/1.425/0.60/0.315 m（振幅 6/4/2.5/1.5）；
#    趋势基线在 2046.84 m 处宽缓极小 + 岩性台阶式跃变；
#    AR(1) 红噪声 + 白噪声；整体缩放至 [65.35, 132.99] API。固定随机种子。
# ======================================================================
def generate_proxy_data(cfg, seed=None):
    seed = cfg["PROXY_SEED"] if seed is None else seed
    rng = np.random.default_rng(seed)
    n = 10084
    depth = 1432.0 + np.arange(n) * 0.1
    sr_true = cfg["PROXY_SR_TRUE"]
    t = (depth - depth.min()) / sr_true            # 时间 (Myr)
    # 趋势基线：余弦形态，宽缓极小值位于 2046.84 m（保证残差在该处出现转折）
    x = (depth - 1432.0) / (2046.84 - 1432.0)
    trend = 20.0 * np.cos(np.pi * x)
    step = 3.0 * (depth >= 2046.84)                # 2046.84 m 岩性台阶式跃变
    # 4 个天文周期分量（周期 kyr → Myr）
    astro = (6.0 * np.sin(2 * np.pi * t / 0.405)
             + 4.0 * np.sin(2 * np.pi * t / 0.095 + 0.7)
             + 2.5 * np.sin(2 * np.pi * t / 0.040 + 1.9)
             + 1.5 * np.sin(2 * np.pi * t / 0.021 + 3.0))
    # AR(1) 红噪声 + 白噪声
    ar = np.zeros(n)
    e = rng.normal(0, 1.0, n)
    phi = 0.95
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + e[i]
    ar *= 3.0
    white = rng.normal(0, 0.8, n)
    raw = trend + step + astro + ar + white
    gr = 65.35 + (raw - raw.min()) / (raw.max() - raw.min()) * (132.99 - 65.35)

    # 岩性表（Sheet1）：8 段，主界面 2046.84 m 必须为岩性边界
    bounds = [1432.0, 1520.0, 1680.0, 1850.0, 2046.84, 2150.0, 2280.0, 2380.0, 2440.3]
    lithos = ["泥岩", "粉砂质泥岩", "泥质粉砂岩", "粉砂岩",
              "泥岩", "泥质粉砂岩", "粉砂质泥岩", "含砾砂岩"]
    df_litho = pd.DataFrame({"顶深": bounds[:-1], "底深": bounds[1:], "岩性": lithos})
    df_log = pd.DataFrame({"深度": depth, "GR": gr})
    return df_log, df_litho


def load_or_generate_data(cfg):
    """按 regenerated benchmark → original benchmark 顺序探测输入 xlsx；
    均不存在则按固定参数和固定随机种子自动生成 synthetic benchmark。

    探测顺序： cfg["INPUT_XLSX"]
        → cfg["INPUT_XLSX_ALT"]
        → 自动生成并写入 cfg["INPUT_XLSX"]。
    """
    for name in (cfg["INPUT_XLSX"], cfg.get("INPUT_XLSX_ALT")):
        if not name:
            continue
        path = os.path.join(cfg["PROJECT_DIR"], name)
        if os.path.exists(path):
            print(f"[数据] 发现输入文件，读取: {path}")
            df_log = pd.read_excel(path, sheet_name="Sheet2")
            try:
                df_litho = pd.read_excel(path, sheet_name="Sheet1")
            except Exception:
                df_litho = None
                print("[数据] 未找到 Sheet1 岩性表，图5将不绘制岩性背景。")
            return df_log, df_litho, path, False
    path = os.path.join(cfg["PROJECT_DIR"], cfg["INPUT_XLSX"])
    print(f"[数据] 未发现输入文件（{cfg['INPUT_XLSX']} / {cfg['INPUT_XLSX_ALT']}）"
          " → 按 SPEC §1 生成专利实施例代理数据"
          f"（种子={cfg['PROXY_SEED']}, 内嵌 SR_true={cfg['PROXY_SR_TRUE']} m/Myr, "
          "周期 405/95/40/21 kyr）。")
    df_log, df_litho = generate_proxy_data(cfg)
    with pd.ExcelWriter(path) as writer:
        df_litho.to_excel(writer, sheet_name="Sheet1", index=False)
        df_log.to_excel(writer, sheet_name="Sheet2", index=False)
    print(f"[数据] 代理数据已保存: {path}  (Sheet1=岩性表, Sheet2=[深度, GR])")
    return df_log, df_litho, path, True


def ensure_uniform_depth(depth, signal, dt):
    """深度非均匀 → 线性插值重采样到等间距（FFT/MTM 要求）。"""
    depth = np.asarray(depth, dtype=float)
    signal = np.asarray(signal, dtype=float)
    d = np.diff(depth)
    if np.allclose(d, dt, rtol=1e-3, atol=1e-12):
        return depth, signal
    print(f"[重采样] 深度采样非均匀，重采样至 {dt} m 等间距。")
    d_new = np.arange(depth[0], depth[-1] + 0.5 * dt, dt)
    s_new = np.interp(d_new, depth, signal)
    return d_new, s_new


# ======================================================================
# 5. 【原专利功能保留】EMD 分解、分量存储与图2/图3
#    （源自 EMD分解.py，核心逻辑未改动；输出图名按 SPEC 编号为图2/图3）
# ======================================================================
def plot_imfs(depth, imfs, residue, output_dir):
    """绘制所有 IMF 分量的垂直堆叠图（图2）与残差趋势项图（图3）。"""
    n_imfs = len(imfs)
    if n_imfs == 0:
        return

    colors = IMF_COLORS
    fig, axes = plt.subplots(n_imfs, 1, figsize=(12, 2.5 * n_imfs), sharex=True)
    if n_imfs == 1:
        axes = [axes]
    for i, imf in enumerate(imfs):
        ax = axes[i]
        color = colors[i % len(colors)]
        ax.plot(depth, imf, color=color, linewidth=1.5)
        ax.fill_between(depth, imf, alpha=0.2, color=color)
        ax.set_ylabel(f'IMF{i+1}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel('深度 (m)', fontsize=11)
    plt.suptitle('图2  EMD 分解结果（IMF 分量垂直堆叠图）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(output_dir, '图2_IMF分量堆叠图.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(depth, residue, color='#2C3E50', linewidth=1.5)
    plt.fill_between(depth, residue, alpha=0.2, color='#2C3E50')
    plt.title('图3  趋势项（残差）', fontsize=12)
    plt.xlabel('深度 (m)')
    plt.ylabel('残差幅值')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图3_残差趋势项.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def emd_decompose_and_plot(depth, signal, max_imf=10, output_dir='emd_result'):
    """
    对测井曲线进行 EMD 分解，保存分解数据（IMF*.csv / residue.csv /
    all_components.xlsx）并生成图2/图3。（原专利主函数，逻辑保留）
    """
    os.makedirs(output_dir, exist_ok=True)

    emd_engine = _PyEMD() if USE_PYEMD else _FallbackEMD()
    print(f"[EMD] 使用 {'PyEMD (emd-signal)' if USE_PYEMD else '内置回退 EMD'}，"
          f"max_imf={max_imf} ...")
    imfs_out = emd_engine.emd(signal, max_imf=max_imf)  # 最后一行为残差
    residue = imfs_out[-1]
    imfs = imfs_out[:-1]

    depth_col = np.asarray(depth)
    for i, imf in enumerate(imfs):
        pd.DataFrame({'depth': depth_col, f'IMF{i+1}': imf}).to_csv(
            os.path.join(output_dir, f'IMF{i+1}.csv'), index=False)
    pd.DataFrame({'depth': depth_col, 'residue': residue}).to_csv(
        os.path.join(output_dir, 'residue.csv'), index=False)

    with pd.ExcelWriter(os.path.join(output_dir, 'all_components.xlsx')) as writer:
        pd.DataFrame({'depth': depth_col}).to_excel(writer, sheet_name='depth', index=False)
        for i, imf in enumerate(imfs):
            pd.DataFrame({f'IMF{i+1}': imf}).to_excel(
                writer, sheet_name=f'IMF{i+1}', index=False)
        pd.DataFrame({'residue': residue}).to_excel(writer, sheet_name='residue', index=False)

    plot_imfs(depth_col, imfs, residue, output_dir)

    print(f"[EMD] 分解完成，共获得 {len(imfs)} 个 IMF 分量 + 1 个残差趋势项。")
    print(f"[EMD] 数据文件和分解图已保存至: {output_dir}")
    return imfs, residue


# ======================================================================
# 6. 【补齐原 R 版图件】图4~图7（zl4.R / zl5.R / zl6.R 逻辑的 Python 复刻）
# ======================================================================
def fig4_variance_contribution(gr, imfs, residue, output_dir):
    """图4：各分量方差贡献率柱状图（ylim 自适应 max*1.15，柱顶标数）。"""
    var_orig = np.var(gr)
    contrib = [np.var(im) / var_orig * 100 for im in imfs]
    contrib.append(np.var(residue) / var_orig * 100)
    labels = [f'IMF{i+1}' for i in range(len(imfs))] + ['残差']

    y_max = max(contrib) * 1.15        # ylim 自适应，避免高贡献柱（如残差）被截断
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, contrib, color='steelblue')
    for b, c in zip(bars, contrib):
        y_txt = min(c + 0.8, y_max * 0.93)   # 标注保持在轴内，防止画布异常扩张
        ax.text(b.get_x() + b.get_width() / 2, y_txt, f'{c:.1f}',
                ha='center', va='bottom', fontsize=9, color='darkblue')
    ax.set_ylim(0, y_max)
    ax.set_ylabel('贡献率 (%)')
    ax.set_title('图4  各分量方差贡献率')
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图4_方差贡献率.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()
    top = labels[int(np.argmax(contrib))]
    print(f"[图4] 方差贡献率最高分量: {top} ({max(contrib):.1f}%)")
    return dict(zip(labels, contrib))


# 岩性 → (颜色, 纹理)；专利：右斜线=泥岩，左斜线=粉砂岩，网格=砾岩
LITHO_STYLE = {
    '泥岩':     ('#cfe0f1', '///'),
    '粉砂岩':   ('#f4e3c0', '\\\\'),
    '含砾砂岩': ('#f2c9c4', '++'),
    '砾岩':     ('#f2c9c4', '++'),
    '泥质粉砂岩': ('#e2efd9', 'xx'),
    '粉砂质泥岩': ('#e6dff0', '..'),
}

def _litho_style(litho):
    for key in LITHO_STYLE:
        if key in str(litho):
            return LITHO_STYLE[key]
    return ('#e8e8e8', '')


def fig5_imf_lithology(depth, imfs, df_litho, output_dir):
    """图5：IMF 堆叠图 + 岩性纹理/色带叠合。"""
    n_imfs = len(imfs)
    fig, axes = plt.subplots(n_imfs, 1, figsize=(12, 2.5 * n_imfs), sharex=True)
    if n_imfs == 1:
        axes = [axes]
    legend_done = set()
    for i, imf in enumerate(imfs):
        ax = axes[i]
        color = IMF_COLORS[i % len(IMF_COLORS)]
        ax.plot(depth, imf, color=color, linewidth=1.2, zorder=3)
        ax.fill_between(depth, imf, alpha=0.2, color=color, zorder=2)
        ax.set_ylabel(f'IMF{i+1}', fontsize=10)
        ax.grid(True, alpha=0.3, zorder=1)
        if df_litho is not None:
            y_lo, y_hi = ax.get_ylim()
            for _, row in df_litho.iterrows():
                top, base, litho = row['顶深'], row['底深'], row['岩性']
                if pd.isna(top) or pd.isna(base):
                    continue
                fc, hatch = _litho_style(litho)
                rect = Rectangle((top, y_lo), base - top, y_hi - y_lo,
                                 facecolor=fc, edgecolor='gray', linewidth=0.4,
                                 hatch=hatch, alpha=0.55, zorder=0)
                ax.add_patch(rect)
                if i == 0 and litho not in legend_done:
                    legend_done.add(litho)
        ax.set_xlim(depth.min(), depth.max())
    # 图例（岩性）
    if df_litho is not None and legend_done:
        handles = [Rectangle((0, 0), 1, 1, facecolor=_litho_style(l)[0],
                             edgecolor='gray', hatch=_litho_style(l)[1],
                             alpha=0.55, label=str(l))
                   for l in legend_done]
        axes[0].legend(handles=handles, loc='upper right', fontsize=8,
                       ncol=len(handles), framealpha=0.9)
    axes[-1].set_xlabel('深度 (m)', fontsize=11)
    plt.suptitle('图5  IMF 分解与岩性叠合图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(output_dir, '图5_IMF岩性叠合图.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def _extrema_density(imf, depth, window_m=3.0):
    """IMF 极值点密度：局部极大+极小值，window_m 滑窗计数 / 窗口长度（每米）。"""
    d = np.diff(imf)
    ext = np.where(np.diff(np.sign(d)) != 0)[0] + 1
    ext_depth = depth[ext]
    # 用深度网格上的指示函数 + 矩形窗卷积实现滑窗计数
    dt = np.median(np.diff(depth))
    win_n = max(1, int(round(window_m / dt)))
    indicator = np.zeros(depth.size)
    idx = np.clip(ext, 0, depth.size - 1)
    indicator[idx] = 1.0
    kernel = np.ones(win_n)
    counts = np.convolve(indicator, kernel, mode='same')
    return counts / window_m


def fig6_extrema_density(depth, imfs, output_dir, n_plot=5, window_m=3.0):
    """图6：IMF1–5 极值密度分面图（3 m 滑窗）。"""
    n_plot = min(n_plot, len(imfs))
    density_list = [_extrema_density(imfs[i], depth, window_m) for i in range(n_plot)]

    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        ax = axes[i]
        y_max = max(density_list[i].max() * 1.3, 1e-6)
        ax.plot(depth, density_list[i], color='black', linewidth=1.5)
        ax.set_ylim(0, y_max)
        ax.set_ylabel(f'IMF{i+1}', fontsize=10)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel('深度 (m)', fontsize=11)
    plt.suptitle(f'图6  各 IMF 极值点密度（分面图，滑窗 {window_m:g} m）',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(output_dir, '图6_极值密度分面图.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig7_reconstruction(depth, gr, imfs, residue, output_dir):
    """图7：原始 GR 曲线与重构曲线对比，报告重构误差。"""
    reconstructed = np.sum(np.vstack(list(imfs) + [residue]), axis=0)
    err = np.abs(reconstructed - gr)
    max_err = float(err.max())
    rms_err = float(np.sqrt(np.mean(err ** 2)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(depth, gr, color='black', linewidth=1.5, label='原始GR')
    ax.plot(depth, reconstructed, color='red', linewidth=1.2, linestyle='--',
            label='重构 (IMF+残差)')
    ax.set_xlabel('深度 (m)')
    ax.set_ylabel('GR (API)')
    ax.set_title(f'图7  原始GR曲线与重构曲线对比（最大误差 {max_err:.2e} API, '
                 f'RMS {rms_err:.2e} API）')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图7_重构对比图.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()
    print(f"[图7] 重构误差: max={max_err:.3e} API, RMS={rms_err:.3e} API "
          f"({'达标 <0.5 API' if max_err < 0.5 else '超限!'})")
    return max_err, rms_err


# ======================================================================
# 7. 【原专利功能保留】S8.5 残差趋势项极值检测与界面识别 + 分层表
# ======================================================================
def detect_interfaces(residue, depth, min_thickness_m=1.0):
    """
    残差趋势项一阶差分符号变化 → 极值界面；合并间距 < min_thickness_m 的界面。
    返回界面深度数组（升序）。
    """
    diff_res = np.diff(residue)
    sign_changes = np.where(np.diff(np.sign(diff_res)) != 0)[0] + 1
    interfaces = depth[sign_changes]
    if len(interfaces) > 1:
        keep = np.concatenate(([True], np.diff(interfaces) >= min_thickness_m))
        interfaces = interfaces[keep]
    print(f"[分层] 检测到 {len(interfaces)} 个界面: "
          f"{np.round(interfaces, 2).tolist()}")
    return interfaces


def build_layer_table(depth, interfaces, ages=None):
    """由界面构建分层表；ages=None 时 Age 列填 NaN。"""
    bounds = np.concatenate(([depth.min()], interfaces, [depth.max()]))
    rows = []
    for k in range(len(bounds) - 1):
        top_d, base_d = float(bounds[k]), float(bounds[k + 1])
        if ages is not None:
            top_age = float(np.interp(top_d, depth, ages))
            base_age = float(np.interp(base_d, depth, ages))
        else:
            top_age = base_age = np.nan
        rows.append(dict(Layer_No=k + 1, Top_Depth=round(top_d, 2),
                         Base_Depth=round(base_d, 2),
                         Top_Age=round(top_age, 4) if ages is not None else np.nan,
                         Base_Age=round(base_age, 4) if ages is not None else np.nan,
                         Thickness=round(base_d - top_d, 2)))
    return pd.DataFrame(rows)


# ======================================================================
# 8. 【新增天文调谐模块】S8.6 深度-年代转换
# ======================================================================
def _make_tuner(cfg):
    from astro_py_tuner import AstroTuner
    lo, hi, step = cfg["SR_RANGE"]
    sr_grid = np.arange(lo, hi + 1e-9, step)
    return AstroTuner(target_periods_kyr=tuple(cfg["TARGET_PERIODS_KYR"]),
                      sr_grid=sr_grid, mtm_nw=3, mtm_k=5, band_frac=0.10,
                      n_jobs=cfg["N_JOBS"], seed=cfg["PROXY_SEED"])


def build_tuning_signals(depth, gr, imfs, residue):
    """两路候选调谐信号（去均值）。"""
    low2 = np.sum(np.vstack(list(imfs[-2:]) + [residue]), axis=0)
    low2 = low2 - low2.mean()
    full = np.sum(np.vstack(list(imfs) + [residue]), axis=0)
    full = full - full.mean()
    return {'low2_res': low2, 'full_recon': full}


def run_age_module(cfg, depth, gr, imfs, residue, output_dir, tag=""):
    """
    S8.6 深度-年代转换：
      扫描 SR → best_sr / match_score；
      match_score < MATCH_THRESHOLD → 醒目 WARNING，返回 age=None（鲁棒性降级）；
      否则构建深度-年代模型并生成图9~图12 所需的数据。
    返回 dict(age_model, best_sr, match_score, scan_df, tuner, tuning_signal, mode,
              age_model_linear, anchor_df, sel405)
      age_model       : 最终年龄模型（405 kyr 锚定成功时为分段线性非线性模型，
                        否则为恒定 SR 线性模型；匹配度不足时为 None）
      age_model_linear: 恒定 SR 线性参考模型（调谐失败时为 None）
      anchor_df/sel405: 405 kyr 锚点表 / IMF 遴选诊断（未启用或未匹配时为 None）
    """
    from astro_py_tuner import AstroTuner  # noqa: F401
    tuner = _make_tuner(cfg)
    signals = build_tuning_signals(depth, gr, imfs, residue)

    mode = cfg["TUNING_SIGNAL_MODE"]
    if mode == "auto":
        chosen, best_ms, best_scan, best_sr = None, -1, None, None
        for name, sig in signals.items():
            scan = tuner.scan_sed_rates(depth, sig)
            sr_opt, ms = tuner.best_sr(scan, depth=depth, signal=sig)
            print(f"[调谐] 候选信号 {name:10s}: best_sr={sr_opt:.3f} m/Myr, "
                  f"match_score={ms:.3f}")
            if ms > best_ms:
                chosen, best_ms, best_scan, best_sr = name, ms, scan, sr_opt
        print(f"[调谐] auto 模式选择调谐信号: {chosen} (match_score={best_ms:.3f})")
        tuning_signal = signals[chosen]
    else:
        chosen = mode
        tuning_signal = signals[mode]
        best_scan = tuner.scan_sed_rates(depth, tuning_signal)
        best_sr, best_ms = tuner.best_sr(best_scan, depth=depth,
                                         signal=tuning_signal)

    best_scan.to_csv(os.path.join(output_dir, f"{tag}sr_scan.csv"), index=False)
    print(f"[调谐] ★ 最优沉积速率 best_sr = {best_sr:.3f} m/Myr, "
          f"match_score = {best_ms:.3f} (阈值 {cfg['MATCH_THRESHOLD']})")

    if best_ms < cfg["MATCH_THRESHOLD"]:
        banner("⚠⚠⚠  WARNING：天文调谐匹配度低于阈值  ⚠⚠⚠\n"
               f"match_score = {best_ms:.3f} < MATCH_THRESHOLD = {cfg['MATCH_THRESHOLD']}\n"
               "信号中未检测到可信的天文旋回信号 → 跳过年代赋值，仅输出深度分层结果。\n"
               "（layer_results 中 Top_Age/Base_Age 列为 NaN）")
        return dict(age_model=None, best_sr=best_sr, match_score=best_ms,
                    scan_df=best_scan, tuner=tuner, tuning_signal=tuning_signal,
                    mode=chosen, age_model_linear=None, anchor_df=None,
                    sel405=None)

    age_linear = tuner.build_age_model(depth, best_sr,
                                       anchor_depth=cfg["ANCHOR_DEPTH"],
                                       anchor_age_ma=cfg["ANCHOR_AGE_MA"])
    duration = age_linear.max() - age_linear.min()
    assert np.all(np.diff(age_linear) > 0), "Age 必须随 Depth 严格单调递增"
    print(f"[年代] 线性深度-年代模型已构建（恒定 SR={best_sr:.3f} m/Myr，"
          f"顶界相对 0 Ma）；井段总时长 = {duration:.2f} Myr；单调递增 ✓")

    # ------------- S8.6+ 405 kyr 长偏心率锚定非线性年龄模型 -------------
    age_model = age_linear
    anchor_df, sel405 = None, None
    if cfg.get("TUNING_405", False):
        banner("S8.6+  405 kyr 长偏心率锚定（分段线性非线性深度-年代模型）")
        sel405 = tuner.select_405_imf(imfs, depth, best_sr,
                                     tolerance=cfg["CYCLE_405_TOLERANCE"])
        ne = sel405['ne']
        print(f"[405锚定] Ne = 总时长/405 kyr = {ne:.2f} 个长偏心率旋回")
        for d in sel405['diag']['per_imf']:
            print(f"[405锚定]   IMF{d['imf_index']+1}: 周期数={d['n_cycles']}, "
                  f"相对偏差={d['rel_dev']*100:.1f}%")
        print(f"[405锚定] 遴选 IMF{sel405['imf_index']+1}（周期数 "
              f"{sel405['n_cycles']}，与 Ne 相对偏差 "
              f"{sel405['diag']['rel_dev']*100:.1f}%，容差 "
              f"{cfg['CYCLE_405_TOLERANCE']*100:.0f}%）；MTM 主谱峰周期 = "
              f"{sel405['diag']['mtm_peak_period_kyr']:.1f} kyr "
              f"(405±20% 核验 {'通过' if sel405['diag']['mtm_spec_ok'] else '未通过'})")
        if sel405['matched']:
            try:
                ref_fn = lambda d: float(np.interp(d, depth, age_linear))
                age405, anchor_df = tuner.build_age_model_405(
                    depth, imfs[sel405['imf_index']],
                    anchor_phase=cfg["ANCHOR_PHASE"], ref_age_fn=ref_fn)
                isr = anchor_df['interval_sr'].dropna()
                max_dev = float(np.max(np.abs(age405 - age_linear)))
                print(f"[405锚定] 锚点相位={cfg['ANCHOR_PHASE']}，锚点总数 = "
                      f"{len(anchor_df)}；锚点 Age = 线性参考(首锚点) + k×0.405 Myr")
                print(f"[405锚定] interval_sr (m/Myr): min={isr.min():.2f}, "
                      f"mean={isr.mean():.2f}, max={isr.max():.2f}")
                print(f"[405锚定] 新模型已替换线性模型；新旧模型最大年龄差 = "
                      f"{max_dev:.3f} Myr；Age 严格单调递增 ✓")
                anchor_df.to_csv(os.path.join(
                    output_dir, f"{tag}anchor_405.csv"), index=False)
                age_model = age405
            except Exception as exc:
                banner("⚠⚠⚠  WARNING：405 kyr 锚定模型构建失败  ⚠⚠⚠\n"
                       f"原因：{exc}\n"
                       "→ 回退至恒定 SR 线性年龄模型。")
                anchor_df = None
        else:
            banner("⚠⚠⚠  WARNING：405 kyr 锚定未通过容差/谱峰核验  ⚠⚠⚠\n"
                   f"所选 IMF{sel405['imf_index']+1} 周期数={sel405['n_cycles']} "
                   f"vs Ne={ne:.2f}（相对偏差 {sel405['diag']['rel_dev']*100:.1f}% "
                   f"> 容差 {cfg['CYCLE_405_TOLERANCE']*100:.0f}%）或 MTM 谱峰"
                   f"（{sel405['diag']['mtm_peak_period_kyr']:.1f} kyr）不在 "
                   "405 kyr ±20% 内。\n"
                   "→ 保留恒定 SR 线性年龄模型（鲁棒路径）。")
    return dict(age_model=age_model, best_sr=best_sr, match_score=best_ms,
                scan_df=best_scan, tuner=tuner, tuning_signal=tuning_signal,
                mode=chosen, age_model_linear=age_linear,
                anchor_df=anchor_df, sel405=sel405)


def save_layer_results(df_layers, output_dir, tag=""):
    csv_path = os.path.join(output_dir, f"{tag}layer_results.csv")
    xlsx_path = os.path.join(output_dir, f"{tag}layer_results.xlsx")
    df_layers.to_csv(csv_path, index=False)
    df_layers.to_excel(xlsx_path, index=False)
    print(f"[分层] 分层成果表已保存: {csv_path} / {xlsx_path}")
    print(df_layers.to_string(index=False))


# ======================================================================
# 9. 图件：图1 升级版流程图、图8 升级版综合成果图、图9~图12
# ======================================================================
def fig1_flowchart(output_dir):
    """图1 升级版：方法流程图（原 S1–S9B + 新增 S8.6 深度-年代转换节点）。"""
    steps = [
        ("S1  读取测井数据\n（深度 + GR / 岩性表）", "#dbe9f6", False),
        ("S2  提取一维目标特征序列\n（数据清洗）", "#dbe9f6", False),
        ("S3  初始化 EMD 自适应分层引擎\n（max_imf=10）", "#dbe9f6", False),
        ("S4–S7  极值提取 → 三次样条包络\n→ 均值筛分 → 分离 IMF", "#dbe9f6", False),
        ("S8  输出多尺度特征集\n（N 个 IMF + 1 个残差趋势项）", "#dbe9f6", False),
        ("S8.5  残差一阶差分符号变化检测\n→ 极值界面识别（合并<1m薄层）", "#fde9d9", False),
        ("S8.6  【新增】深度-年代转换\n天文调谐 SR 扫描 → best_sr\nmatch≥阈值→赋年代；否则降级", "#d9ead3", True),
        ("S9A  多格式数据落盘\n（IMF*.csv / residue.csv / all_components.xlsx）", "#e8e0f0", False),
        ("S9B  多维可视化图件渲染\n（图2–图12，300 dpi PNG）", "#e8e0f0", False),
    ]
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.6 + 1)
    ax.axis('off')
    w, h = 6.4, 1.05
    xc = 5.0
    for i, (text, fc, is_new) in enumerate(steps):
        y = len(steps) * 1.6 - i * 1.6 - 0.4
        box = FancyBboxPatch((xc - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.12",
                             facecolor=fc,
                             edgecolor='#c00000' if is_new else '#404040',
                             linewidth=2.2 if is_new else 1.2)
        ax.add_patch(box)
        ax.text(xc, y, text, ha='center', va='center', fontsize=10.5,
                fontweight='bold' if is_new else 'normal',
                color='#c00000' if is_new else 'black')
        if i < len(steps) - 1:
            ax.annotate('', xy=(xc, y - h / 2 - 0.42), xytext=(xc, y - h / 2 - 0.08),
                        arrowprops=dict(arrowstyle='-|>', color='#404040', lw=1.6))
    ax.text(xc, len(steps) * 1.6 + 0.6,
            '图1  基于 EMD 的测井曲线自动分层与天文年代赋值方法流程图（升级版）',
            ha='center', va='center', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图1_方法流程图.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig8_composite(depth, gr, interfaces, age_model, output_dir):
    """
    图8 升级版：GR 曲线 + 界面竖线/倒三角 + 顶部辅助 x 轴 Age(Ma) + 界面年代标注。
    age_model=None 时退化为仅深度轴（鲁棒性降级路径）。
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(depth, gr, color='black', linewidth=1.5, label='原始GR曲线')
    y_lo, y_hi = ax.get_ylim()
    y_top = y_hi - 0.05 * (y_hi - y_lo)
    # 年代标注交错下移（va='top' 使文本整体位于锚点下方），避免与倒三角标记重叠
    ann_offsets = [-26, -44, -62]
    for i, d in enumerate(interfaces):
        ax.axvline(d, color='black', linestyle='--', linewidth=1.5)
        ax.plot(d, y_top, marker='v', color='black', markersize=9)
        if age_model is not None:
            a = np.interp(d, depth, age_model)
            ax.annotate(f'{a:.2f} Ma', xy=(d, y_top),
                        xytext=(0, ann_offsets[i % len(ann_offsets)]),
                        textcoords='offset points', ha='center', va='top',
                        fontsize=8, color='#c00000',
                        arrowprops=dict(arrowstyle='-', color='#c00000', lw=0.6))
    ax.set_xlabel('深度 (m)')
    ax.set_ylabel('GR (API)')
    ax.grid(True, alpha=0.4)
    handles = [plt.Line2D([], [], color='black', lw=1.5, label='原始GR曲线'),
               plt.Line2D([], [], color='black', lw=1.5, ls='--', marker='v',
                          label='检测界面')]
    ax.legend(handles=handles, loc='upper right')
    if age_model is not None:
        sec = ax.secondary_xaxis('top',
                                 functions=(lambda d: np.interp(d, depth, age_model),
                                            lambda a: np.interp(a, age_model, depth)))
        sec.set_xlabel('Age (Ma，相对)')
        ax.set_title('图8  综合分层成果图（深度轴 + 年代辅助轴）')
    else:
        ax.set_title('图8  综合分层成果图（仅深度轴：匹配度不足，未赋年代）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图8_综合分层成果图_含年代轴.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig9_sr_scan(scan_df, best_sr, match_score, threshold, output_dir):
    """图9：SR 扫描目标函数曲线（标注最优 SR 与匹配度阈值线）。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(scan_df['sr'], scan_df['score'], color='#2E86AB', lw=1.8)
    ax.axvline(best_sr, color='#c00000', ls='--', lw=1.5,
               label=f'最优 SR = {best_sr:.2f} m/Myr')
    best = scan_df['score'].max()
    med = np.median(scan_df['score'])
    thr_score = med + threshold * (best - med)   # 匹配度阈值对应的得分水平
    ax.axhline(thr_score, color='gray', ls=':', lw=1.5,
               label=f'匹配度阈值 {threshold} 对应得分线')
    ax.set_xlabel('沉积速率 SR (m/Myr)')
    ax.set_ylabel('目标函数得分 score')
    ax.set_title(f'图9  沉积速率扫描目标函数曲线（best_sr={best_sr:.2f} m/Myr, '
                 f'match_score={match_score:.3f}）')
    ax.grid(True, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图9_沉积速率扫描.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig10_tuned_spectrum(tuner, depth, tuning_signal, best_sr, output_dir):
    """图10：调谐后 MTM 频谱 + 理论天文周期竖线（405/95/40/21 kyr）匹配图。"""
    t = tuner.depth_to_time(depth, best_sr)
    freq, power = tuner.mtm_spectrum(t, tuning_signal)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(freq, power, color='black', lw=1.2, label='调谐后 MTM 功率谱')
    ax.set_yscale('log')   # 对数功率轴：同时展现强趋势峰与天文频带峰
    f_min = 1000.0 / max(tuner.target_periods_kyr) * 0.5
    f_max = 1000.0 / min(tuner.target_periods_kyr) * 1.4
    ax.set_xlim(f_min, f_max)
    y_top = power.max() * 3
    ax.set_ylim(power[(freq >= f_min) & (freq <= f_max)].min() * 0.5, y_top)
    for P, c in zip(tuner.target_periods_kyr,
                    ['#c00000', '#E65100', '#1B5E20', '#4A148C']):
        f0 = 1000.0 / P
        ax.axvline(f0, color=c, ls='--', lw=1.5)
        ax.text(f0, y_top * 0.35, f' {P:g} kyr', color=c,
                rotation=90, va='top', fontsize=9)
    ax.set_xlabel('频率 (cycles/Myr)')
    ax.set_ylabel('功率（对数轴）')
    ax.set_title(f'图10  调谐后频谱与理论天文周期匹配（SR={best_sr:.2f} m/Myr；'
                 '虚线=405/95/40/21 kyr）')
    ax.grid(True, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图10_调谐频谱匹配图.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig11_eha(tuner, depth, tuning_signal, best_sr, output_dir, cfg):
    """图11：eha 演化谱（深度 × 频率 × 归一化功率）。"""
    centers, freq, power2d = tuner.eha(depth, tuning_signal, best_sr,
                                       win_m=cfg["EHA_WIN_M"],
                                       step_m=cfg["EHA_STEP_M"])
    f_max = 1000.0 / min(tuner.target_periods_kyr) * 1.3
    fmask = freq <= f_max
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(12, 6))
    z = np.ma.masked_invalid(power2d[:, fmask])
    pcm = ax.pcolormesh(freq[fmask], centers, z, shading='auto', cmap='viridis',
                        norm=LogNorm(vmin=1e-4, vmax=1.0))  # 对数色标显影弱周期带
    for P in tuner.target_periods_kyr:
        ax.axvline(1000.0 / P, color='white', ls='--', lw=1.0, alpha=0.85)
        ax.text(1000.0 / P, centers.max(), f'{P:g}', color='white', fontsize=8,
                ha='center', va='bottom')
    ax.set_xlabel('频率 (cycles/Myr)')
    ax.set_ylabel('深度 (m)')
    ax.set_title(f'图11  eha 演化谱（窗长 {cfg["EHA_WIN_M"]} m，步长 '
                 f'{cfg["EHA_STEP_M"]} m；白色虚线=目标天文频率）')
    fig.colorbar(pcm, ax=ax, label='归一化功率（对数色标）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图11_eha演化谱.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig12_age_model(depth, age_model, interfaces, best_sr, output_dir,
                    anchor_df=None, age_linear=None):
    """
    图12：深度-年代模型曲线（Age vs Depth，标注界面点）。
    若提供 anchor_df / age_linear（405 kyr 锚定成功），升级为非线性展示：
    分段线性 Age-Depth 曲线 + 锚点散点 + 旧恒定 SR 线性模型虚线对比。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    nonlinear = anchor_df is not None and age_linear is not None
    if nonlinear:
        ax.plot(depth, age_linear, color='gray', ls='--', lw=1.3,
                label=f'旧线性模型（恒定 SR = {best_sr:.2f} m/Myr）')
        ax.plot(depth, age_model, color='#2E86AB', lw=2,
                label='405 kyr 锚定分段线性模型（新）')
        ax.plot(anchor_df['depth'], anchor_df['age'], 'o', color='#E65100',
                markersize=2.5, alpha=0.7,
                label=f'405 kyr 锚点（{len(anchor_df)} 个，间隔 0.405 Myr）')
    else:
        ax.plot(depth, age_model, color='#2E86AB', lw=2,
                label=f'恒定 SR = {best_sr:.2f} m/Myr 年代模型')
    ifs_age = np.interp(interfaces, depth, age_model)
    ax.plot(interfaces, ifs_age, 'v', color='#c00000', markersize=10,
            label='检测界面', zorder=5)
    for d, a in zip(interfaces, ifs_age):
        ax.annotate(f'{d:.2f} m / {a:.2f} Ma', xy=(d, a), xytext=(8, -4),
                    textcoords='offset points', fontsize=8, color='#c00000')
    ax.set_xlabel('深度 (m)')
    ax.set_ylabel('Age (Ma，相对)')
    title = ('图12  深度-年代模型（405 kyr 锚定非线性 vs 旧线性模型虚线对比）'
             if nonlinear else '图12  深度-年代模型（Age 随 Depth 单调递增）')
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图12_深度年代模型.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()


def fig13_405_anchor(depth, imf_sel, anchor_df, sel405, output_dir):
    """
    图13：405 kyr 锚定诊断图。
    上半幅：所选 IMF 基准曲线 + 锚点刻度竖线（标注 IMF 序号与周期数/Ne）；
    下半幅：锚点区间沉积速率 interval_sr 随深度阶梯线图（标注均值/范围）。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.2, 1.0]))
    # ---- 上半幅：IMF 曲线 + 锚点刻度线 ----
    ax1.plot(depth, imf_sel, color='#2E86AB', lw=1.0)
    a_d = anchor_df['depth'].values
    ax1.vlines(a_d, imf_sel.min(), imf_sel.max(),
               color='#c00000', lw=0.35, alpha=0.45)
    ax1.plot(a_d, np.interp(a_d, depth, imf_sel), 'v', color='#c00000',
             markersize=3, alpha=0.8)
    ax1.set_ylabel(f"IMF{sel405['imf_index']+1} 幅值")
    ax1.set_title(f"图13a  405 kyr 基准曲线 IMF{sel405['imf_index']+1} + 锚点刻度线"
                  f"（周期数 {sel405['n_cycles']} / Ne = {sel405['ne']:.1f}；"
                  f"锚点 {len(anchor_df)} 个，红色竖线/倒三角）", fontsize=12)
    ax1.grid(True, alpha=0.3)
    # ---- 下半幅：interval_sr 阶梯图 ----
    isr = anchor_df['interval_sr'].values[:-1]
    isr_d = anchor_df['depth'].values[:-1]
    ax2.step(isr_d, isr, where='post', color='#1B5E20', lw=1.2,
             label='锚点区间沉积速率 interval_sr')
    mu, lo, hi = float(np.mean(isr)), float(np.min(isr)), float(np.max(isr))
    ax2.axhline(mu, color='#E65100', ls='--', lw=1.2,
                label=f'均值 = {mu:.2f} m/Myr')
    ax2.set_xlabel('深度 (m)')
    ax2.set_ylabel('interval_sr (m/Myr)')
    ax2.set_title(f'图13b  区间沉积速率随深度变化（阶梯图；'
                  f'范围 {lo:.2f}–{hi:.2f} m/Myr）', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '图13_405kyr锚定与沉积速率.png'),
                dpi=CONFIG["FIG_DPI"], bbox_inches='tight')
    plt.close()
    print(f"[图13] 405 kyr 锚定诊断图已保存（锚点 {len(anchor_df)} 个；"
          f"interval_sr min/mean/max = {lo:.2f}/{mu:.2f}/{hi:.2f} m/Myr）")


# ======================================================================
# 10. 主流程 与 --selftest 白噪声自检
# ======================================================================
def run_pipeline(cfg, depth, gr, df_litho, output_dir, make_figs=True, tag=""):
    """核心流水线：EMD → 图件 → 界面检测 → 年代模块 → 分层表。
    返回 dict（供验收与自检）。"""
    result = {}
    depth, gr = ensure_uniform_depth(np.asarray(depth, float),
                                     np.asarray(gr, float), cfg["RESAMPLE_DT"])

    banner("S3–S8  EMD 自适应分解")
    imfs, residue = emd_decompose_and_plot(depth, gr, max_imf=cfg["MAX_IMF"],
                                           output_dir=output_dir)
    result['imfs'], result['residue'] = imfs, residue

    if make_figs:
        banner("S9B  图4–图7（原 R 版图件复刻）")
        result['contrib'] = fig4_variance_contribution(gr, imfs, residue, output_dir)
        fig5_imf_lithology(depth, imfs, df_litho, output_dir)
        fig6_extrema_density(depth, imfs, output_dir)
        max_err, rms_err = fig7_reconstruction(depth, gr, imfs, residue, output_dir)
        result['recon_err_max'], result['recon_err_rms'] = max_err, rms_err

    banner("S8.5  残差趋势项极值检测与界面识别")
    interfaces = detect_interfaces(residue, depth, cfg["MIN_THICKNESS_M"])
    result['interfaces'] = interfaces

    # ---------------- S8.6 【新增】深度-年代转换 ----------------
    age_model = None
    if cfg["ENABLE_AGE_MODULE"]:
        banner("S8.6  【新增天文调谐模块】深度-年代转换")
        age_res = run_age_module(cfg, depth, gr, imfs, residue, output_dir, tag=tag)
        result.update({k: age_res[k] for k in
                       ('best_sr', 'match_score', 'scan_df', 'mode',
                        'age_model_linear', 'anchor_df', 'sel405')})
        result['tuner'] = age_res['tuner']
        result['tuning_signal'] = age_res['tuning_signal']
        age_model = age_res['age_model']
        result['age_model'] = age_model
    else:
        print("[年代] ENABLE_AGE_MODULE=False，跳过年代赋值（Age 列填 NaN）。")

    # ---------------- 分层表 ----------------
    if age_model is None and cfg["ENABLE_AGE_MODULE"]:
        print("[分层] ⚠ 无有效年代模型 → layer_results 中 Age 列为 NaN（仅深度分层）。")
    df_layers = build_layer_table(depth, interfaces, age_model)
    save_layer_results(df_layers, output_dir, tag=tag)
    result['layers'] = df_layers

    # ---------------- 图8 与图9–图12 ----------------
    if make_figs:
        banner("图8（升级版）与图9–图13（新增；图13 仅 405 kyr 锚定成功时生成）")
        fig8_composite(depth, gr, interfaces, age_model, output_dir)
        if cfg["ENABLE_AGE_MODULE"]:
            fig9_sr_scan(result['scan_df'], result['best_sr'],
                         result['match_score'], cfg["MATCH_THRESHOLD"], output_dir)
            fig10_tuned_spectrum(result['tuner'], depth, result['tuning_signal'],
                                 result['best_sr'], output_dir)
            fig11_eha(result['tuner'], depth, result['tuning_signal'],
                      result['best_sr'], output_dir, cfg)
            if age_model is not None:
                fig12_age_model(depth, age_model, interfaces,
                                result['best_sr'], output_dir,
                                anchor_df=result.get('anchor_df'),
                                age_linear=result.get('age_model_linear'))
                if result.get('anchor_df') is not None:
                    fig13_405_anchor(depth, imfs[result['sel405']['imf_index']],
                                     result['anchor_df'], result['sel405'],
                                     output_dir)
    return result


def main():
    cfg = dict(CONFIG)
    output_dir = cfg["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    tee = TeeLogger(os.path.join(output_dir, "run_log.txt")).start()
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        banner("EMD 自动分层 + 天文旋回年代学升级  主流程启动")
        print(f"[环境] PyEMD={'可用' if USE_PYEMD else '不可用(内置回退EMD)'}; "
              f"N_JOBS={cfg['N_JOBS']}; 输出目录: {output_dir}")
        print(f"[环境] 依赖版本: "
              f"python={sys.version.split()[0]}; "
              f"numpy={np.__version__}; pandas={pd.__version__}; "
              f"matplotlib={matplotlib.__version__}; "
              f"scipy={__import__('scipy').__version__}; "
              f"PyEMD(emd-signal)={_pyemd_version()}")
        df_log, df_litho, data_path, is_proxy = load_or_generate_data(cfg)
        depth = df_log['深度'].values
        gr = df_log['GR'].values
        print(f"[数据] 井段 {depth.min():.2f}–{depth.max():.2f} m, "
              f"{depth.size} 点, GR {gr.min():.2f}–{gr.max():.2f} API")

        banner("图1  方法流程图（升级版）")
        fig1_flowchart(output_dir)

        res = run_pipeline(cfg, depth, gr, df_litho, output_dir, make_figs=True)

        # ---------------- 验收摘要 ----------------
        banner("运行验收摘要")
        print(f"best_sr      = {res.get('best_sr')} m/Myr "
              f"(真值 {cfg['PROXY_SR_TRUE']} ± 0.5)" if 'best_sr' in res else "")
        print(f"match_score  = {res.get('match_score'):.3f} "
              f"(阈值 {cfg['MATCH_THRESHOLD']})" if 'match_score' in res else "")
        if res.get('age_model') is not None:
            dur = res['age_model'].max() - res['age_model'].min()
            theory = (depth.max() - depth.min()) / cfg['PROXY_SR_TRUE']
            print(f"井段总时长   = {dur:.2f} Myr（理论 ≈{theory:.2f} Myr, "
                  f"偏差 {abs(dur - theory) / theory * 100:.2f}%）")
            print(f"Age 单调递增 = {bool(np.all(np.diff(res['age_model']) > 0))}")
        if res.get('anchor_df') is not None:
            s405 = res['sel405']
            isr = res['anchor_df']['interval_sr'].dropna()
            print(f"405锚定      = IMF{s405['imf_index']+1}（周期数 "
                  f"{s405['n_cycles']} / Ne={s405['ne']:.2f}），锚点 "
                  f"{len(res['anchor_df'])} 个，interval_sr "
                  f"{isr.min():.2f}/{isr.mean():.2f}/{isr.max():.2f} "
                  f"m/Myr (min/mean/max)，新旧模型最大年龄差 "
                  f"{np.abs(res['age_model'] - res['age_model_linear']).max():.3f} Myr")
        if 'recon_err_max' in res:
            print(f"重构误差     = max {res['recon_err_max']:.3e} API (<0.5 API)")
        print(f"界面列表     = {np.round(res['interfaces'], 2).tolist()}")
        print("全部图件与数据文件见:", output_dir)
    finally:
        sys.stdout = old_stdout
        tee.close()
    return res


def selftest():
    """
    --selftest 白噪声演示模式：
    纯白噪声信号 → AstroTuner 匹配度低 → 醒目警告 → 仅输出深度分层。
    """
    cfg = dict(CONFIG)
    out = os.path.join(cfg["OUTPUT_DIR"], "selftest_result")
    os.makedirs(out, exist_ok=True)
    tee = TeeLogger(os.path.join(out, "selftest_run_log.txt")).start()
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        banner("SELFTEST：白噪声鲁棒性自检（低匹配度 → 跳过年代赋值路径）")
        rng = np.random.default_rng(12345)
        depth = 1432.0 + np.arange(10084) * 0.1
        noise = rng.normal(0, 1.0, depth.size)
        gr = 65.35 + (noise - noise.min()) / (noise.max() - noise.min()) * (132.99 - 65.35)
        print(f"[自检] 纯白噪声信号已生成（{depth.size} 点，无天文周期）。")

        res = run_pipeline(cfg, depth, gr, None, out, make_figs=False,
                           tag="selftest_")
        ms = res.get('match_score', np.nan)
        ok_low = ms < cfg["MATCH_THRESHOLD"]
        ok_no_age = res.get('age_model') is None
        ok_nan_age = res['layers'][['Top_Age', 'Base_Age']].isna().all().all()
        banner("SELFTEST 结果")
        print(f"match_score = {ms:.3f} (< {cfg['MATCH_THRESHOLD']}) → "
              f"{'符合预期' if ok_low else '不符合预期!'}")
        print(f"跳过年代赋值 → {'是' if ok_no_age else '否!'}; "
              f"Age 列全 NaN → {'是' if ok_nan_age else '否!'}")
        if ok_low and ok_no_age and ok_nan_age:
            print("SELFTEST PASS ✓  鲁棒性降级路径验证通过。")
            return True
        print("SELFTEST FAIL ✗")
        return False
    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        ok = selftest()
        sys.exit(0 if ok else 1)
    else:
        main()
