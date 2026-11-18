# -*- coding: utf-8 -*-
"""
astro_py_tuner.py
=================
astrochron (R 包) timeOpt / eha 核心逻辑的独立 Python 重写（严禁调用 R）。

功能模块（对应 SPEC §2 接口契约）：
    - Thomson 多窗口谱 (MTM)：基于 scipy.signal.windows.dpss 自行实现
    - 深度域 → 时间域转换（恒定沉积速率假设）
    - 目标函数：目标天文频带能量聚集度 × 峰位匹配系数（timeOpt 思想）
    - 沉积速率 (SR) 网格扫描（支持 multiprocessing 加速）
    - 深度-年代模型构建（相对年代，无锚点时顶界 0 Ma，Age 随深度单调递增）
    - eha 等效：滑动窗 MTM 演化谱（深度 × 频率 × 功率）

接口契约（SPEC §2）：
    class AstroTuner:
        def __init__(self, target_periods_kyr=(405,95,40,21), sr_grid=None,
                     mtm_nw=3, mtm_k=5, band_frac=0.10, n_jobs=1, seed=42)
        def mtm_spectrum(self, t, x) -> (freq, power)
        def depth_to_time(self, depth, sr, ref='top')
        def objective(self, depth, signal, sr) -> float
        def scan_sed_rates(self, depth, signal) -> DataFrame   # 列: sr, score
        def best_sr(self, scan_df) -> (sr_opt, match_score)    # match_score ∈ [0,1]
        def build_age_model(self, depth, sr, anchor_depth=None, anchor_age_ma=None)
        def eha(self, depth, signal, sr, win_m=150, step_m=10)
        def count_cycles(self, x, min_prominence=None)
            -> (n_cycles, peak_idx, trough_idx)
        def select_405_imf(self, imfs, depth, sr_est, tolerance=0.15)
            -> dict(imf_index, n_cycles, ne, matched, diag)
        def build_age_model_405(self, depth, imf, anchor_phase='trough',
                                ref_age_fn=None) -> (age_array, anchor_df)

单位约定：
    深度 m；沉积速率 m/Myr；时间 Myr；频率 cycles/Myr（= 1000/周期kyr）。
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.signal.windows import dpss
from scipy import interpolate

__all__ = ["AstroTuner"]


# ----------------------------------------------------------------------
# multiprocessing worker（必须位于模块顶层以便 pickle）
# ----------------------------------------------------------------------
def _sr_worker(args):
    """SR 网格扫描的并行 worker：计算单个 SR 的目标函数值。"""
    tuner_cfg, depth, signal, sr = args
    tuner = AstroTuner(**tuner_cfg)
    return sr, tuner.objective(depth, signal, sr)


class AstroTuner:
    """TimeOpt-inspired Python reimplementation of the astrochron timeOpt/eha logic
    (非逐行复刻，未经 R 端数值对照 / not a line-by-line replication, no R-side numerical cross-check)."""

    def __init__(self, target_periods_kyr=(405, 95, 40, 21), sr_grid=None,
                 mtm_nw=3, mtm_k=5, band_frac=0.10, n_jobs=1, seed=42):
        self.target_periods_kyr = tuple(float(p) for p in target_periods_kyr)
        # 目标频率 cycles/Myr
        self.target_freqs = np.array([1000.0 / p for p in self.target_periods_kyr])
        # SR 网格 (m/Myr)，默认 1–20 步长 0.25
        self.sr_grid = np.arange(1.0, 20.0 + 1e-9, 0.25) if sr_grid is None \
            else np.asarray(sr_grid, dtype=float)
        self.mtm_nw = float(mtm_nw)      # 时间半带宽积 NW
        self.mtm_k = int(mtm_k)          # DPSS 窗口个数 K
        self.band_frac = float(band_frac)  # 目标频带半宽比例
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)
        # 各波段权重：与实施例振幅包络一致（长周期分量振幅大、信噪比高）
        default_w = {405: 6.0, 95: 4.0, 40: 2.5, 21: 1.5}
        self.band_weights = np.array([default_w.get(int(p), 1.0)
                                      for p in self.target_periods_kyr])
        self.band_weights /= self.band_weights.sum()
        # 峰显著性阈值：目标频带内局部峰功率须 ≥ 邻域中位数 × SIG_THRESH 才算命中。
        # 取 5.0：白噪声频谱在频带内的局部峰/中位数典型值仅 2–3，而真实天文
        # 周期峰通常 ≫10，该阈值可在两类信号间给出干净分离。
        self.sig_thresh = 5.0

    # ------------------------------------------------------------------
    # 工具：非均匀采样 → 等间隔重采样
    # ------------------------------------------------------------------
    @staticmethod
    def _uniform_resample(t, x, dt=None):
        """若采样非均匀，用线性插值重采样到等间距网格。"""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        order = np.argsort(t)
        t, x = t[order], x[order]
        d = np.diff(t)
        if dt is None:
            dt = np.median(d)
        if np.allclose(d, dt, rtol=1e-3, atol=1e-12):
            return t, x, dt
        t_new = np.arange(t[0], t[-1] + 0.5 * dt, dt)
        f = interpolate.interp1d(t, x, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
        return t_new, f(t_new), dt

    # ------------------------------------------------------------------
    # Thomson 多窗口谱（scipy DPSS 实现）
    # ------------------------------------------------------------------
    def mtm_spectrum(self, t, x):
        """
        Thomson multitaper 功率谱估计。
        参数:
            t : 时间坐标 (Myr)，允许非均匀（内部自动重采样）
            x : 信号
        返回:
            freq  : 频率 (cycles/Myr)，仅正频率
            power : 对应功率谱密度
        """
        t, x, dt = self._uniform_resample(t, x)
        n = x.size
        x = x - np.mean(x)
        # K 个 DPSS 离散长球面序列窗口
        tapers = dpss(n, self.mtm_nw, Kmax=self.mtm_k, return_ratios=False)
        # 零填充到 4× 下一 2 的幂，提升峰位定位精度与绘图平滑度
        nfft = 1 << int(np.ceil(np.log2(4 * n)))
        spec = np.abs(np.fft.rfft(tapers * x[None, :], n=nfft, axis=1)) ** 2
        power = spec.mean(axis=0) * dt          # 平均各窗口谱 → 功率
        freq = np.fft.rfftfreq(nfft, d=dt)      # cycles/Myr
        return freq, power

    # ------------------------------------------------------------------
    # 深度域 → 时间域
    # ------------------------------------------------------------------
    def depth_to_time(self, depth, sr, ref='top'):
        """t = (d - d_ref)/sr，单位 Myr。ref='top' 时以顶界为零点。"""
        depth = np.asarray(depth, dtype=float)
        d_ref = depth.min() if ref == 'top' else depth.max()
        return (depth - d_ref) / float(sr)

    # ------------------------------------------------------------------
    # 目标函数（timeOpt 思想：频带能量聚集度 × 峰位匹配系数）
    # ------------------------------------------------------------------
    def _band_diagnostics(self, freq, power):
        """返回 (band_power_ratio, match_coeff, hits, sigs)。"""
        total = np.trapezoid(power, freq)
        if total <= 0:
            return 0.0, 0.0, [], []
        band_power = 0.0
        hits, sigs = [], []
        for f0, w in zip(self.target_freqs, self.band_weights):
            f_lo, f_hi = f0 * (1 - self.band_frac), f0 * (1 + self.band_frac)
            band = (freq >= f_lo) & (freq <= f_hi)
            # 邻域（0.675f0–1.375f0 去掉本频带），用于估计局部背景
            neigh = (freq >= f_lo * 0.75) & (freq <= f_hi * 1.25) & (~band)
            if band.sum() < 3 or neigh.sum() < 5:
                hits.append(0.0)
                sigs.append(0.0)
                continue
            band_power += np.trapezoid(power[band], freq[band])
            i_pk = np.argmax(power[band])
            p_pk = power[band][i_pk]
            f_pk = freq[band][i_pk]
            med = np.median(power[neigh])
            sig = p_pk / med if med > 0 else 0.0
            sigs.append(sig)
            # 软命中：sig >= sig_thresh → 1，线性过渡；下限 1（等于背景）→ 0
            sig_hit = np.clip((sig - 1.0) / (self.sig_thresh - 1.0), 0.0, 1.0)
            # 峰位精度：峰越贴近目标频率越好（锐化目标函数，保证真值 SR 处全局最大）
            precision = 1.0 - abs(f_pk - f0) / (self.band_frac * f0)
            precision = float(np.clip(precision, 0.0, 1.0))
            hits.append(sig_hit * precision * w)
        band_power_ratio = band_power / total
        match_coeff = float(np.sum(hits) / self.band_weights.sum())
        return band_power_ratio, match_coeff, hits, sigs

    def objective(self, depth, signal, sr):
        """
        给定 SR，将深度信号转到时间域并计算目标函数：
            score = 目标频带功率占比 × 峰位匹配系数
        在真值 SR 处取全局最大。
        """
        if sr <= 0:
            return 0.0
        t = self.depth_to_time(depth, sr)
        freq, power = self.mtm_spectrum(t, signal)
        ratio, match, _, _ = self._band_diagnostics(freq, power)
        return float(ratio * match)

    # ------------------------------------------------------------------
    # 逻辑A：SR 网格扫描（multiprocessing 可选）
    # ------------------------------------------------------------------
    def scan_sed_rates(self, depth, signal):
        """扫描 SR 网格，返回 DataFrame[sr, score]。"""
        depth = np.asarray(depth, dtype=float)
        signal = np.asarray(signal, dtype=float)
        scores = None
        if self.n_jobs > 1 and len(self.sr_grid) > 1 and sys.platform.startswith('linux'):
            try:
                import multiprocessing as mp
                cfg = dict(target_periods_kyr=self.target_periods_kyr,
                           sr_grid=self.sr_grid, mtm_nw=self.mtm_nw,
                           mtm_k=self.mtm_k, band_frac=self.band_frac,
                           n_jobs=1, seed=self.seed)
                args = [(cfg, depth, signal, float(sr)) for sr in self.sr_grid]
                ctx = mp.get_context('fork')
                with ctx.Pool(processes=min(self.n_jobs, ctx.cpu_count())) as pool:
                    results = pool.map(_sr_worker, args)
                scores = dict(results)
            except Exception as exc:  # 并行失败 → 退化单进程
                print(f"[AstroTuner] multiprocessing 不可用，退化为单进程: {exc}")
                scores = None
        if scores is None:
            scores = {float(sr): self.objective(depth, signal, float(sr))
                      for sr in self.sr_grid}
        df = pd.DataFrame({'sr': list(scores.keys()),
                           'score': list(scores.values())}).sort_values('sr')
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 最优 SR 与匹配度
    # ------------------------------------------------------------------
    def best_sr(self, scan_df, depth=None, signal=None):
        """
        返回 (sr_opt, match_score)。
        sr_opt      : 扫描得分最大处的 SR（含抛物线亚网格精化）。
        match_score ∈ [0,1]：sr_opt 处的“峰位匹配系数”——4 个目标频带内局部峰
                      显著高于邻域背景的命中度（按波段权重归一，SPEC“命中波段数
                      加权”），真实天文信号 → ~1；白噪声 → ~0。
                      若未提供 depth/signal，则退化为扫描曲线显著度
                      (best-median)/best 的近似度量。
        """
        scan_df = scan_df.sort_values('sr').reset_index(drop=True)
        i_best = int(np.argmax(scan_df['score'].values))
        sr_grid_sr = float(scan_df.loc[i_best, 'sr'])
        sr_opt = sr_grid_sr
        # 抛物线插值亚网格精化
        if 0 < i_best < len(scan_df) - 1:
            y1, y2, y3 = scan_df['score'].values[i_best - 1:i_best + 2]
            denom = (y1 - 2 * y2 + y3)
            if abs(denom) > 1e-15:
                delta = 0.5 * (y1 - y3) / denom
                step = scan_df['sr'].values[1] - scan_df['sr'].values[0]
                if abs(delta) <= 1.0:
                    sr_opt = sr_opt + delta * step
        if depth is not None and signal is not None:
            match_score = self.match_quality(depth, signal, sr_grid_sr)
        else:
            best = float(scan_df['score'].max())
            med = float(np.median(scan_df['score'].values))
            match_score = float(np.clip((best - med) / max(best, 1e-12), 0.0, 1.0))
        return sr_opt, match_score

    def match_quality(self, depth, signal, sr):
        """最优 SR 处的峰位匹配系数（∈[0,1]），供主流程做鲁棒性判断。"""
        t = self.depth_to_time(depth, sr)
        freq, power = self.mtm_spectrum(t, signal)
        _, match, _, _ = self._band_diagnostics(freq, power)
        return float(match)

    # ------------------------------------------------------------------
    # 逻辑D：深度-年代模型
    # ------------------------------------------------------------------
    def build_age_model(self, depth, sr, anchor_depth=None, anchor_age_ma=None):
        """
        Age(Ma) 随深度单调递增。
        无锚点：顶界相对 0 Ma，Age = (d - d_top)/sr。
        有锚点：Age = anchor_age + (d - anchor_depth)/sr。
        """
        depth = np.asarray(depth, dtype=float)
        if anchor_depth is None or anchor_age_ma is None:
            age = (depth - depth.min()) / float(sr)
        else:
            age = float(anchor_age_ma) + (depth - float(anchor_depth)) / float(sr)
        return age

    # ------------------------------------------------------------------
    # 405 kyr 长偏心率锚定：周期计数、IMF 遴选、分段线性深度-年代模型
    # ------------------------------------------------------------------
    def count_cycles(self, x, min_prominence=None):
        """
        统计信号的波动周期数（相邻同相极值峰-峰 / 谷-谷 各计一个周期）。
        参数:
            x : 一维信号
            min_prominence : find_peaks 最小突出度；None → 0.5×std(x)
        返回:
            n_cycles  : 完整周期数 = max(峰数, 谷数) - 1（下限 0）
            peak_idx  : 极大值索引数组
            trough_idx: 极小值索引数组
        """
        from scipy.signal import find_peaks
        x = np.asarray(x, dtype=float)
        if min_prominence is None:
            min_prominence = 0.5 * float(np.std(x))
        peak_idx, _ = find_peaks(x, prominence=min_prominence)
        trough_idx, _ = find_peaks(-x, prominence=min_prominence)
        n_cycles = max(peak_idx.size, trough_idx.size) - 1
        return max(int(n_cycles), 0), peak_idx, trough_idx

    def select_405_imf(self, imfs, depth, sr_est, tolerance=0.15):
        """
        遴选与 405 kyr 长偏心率旋回个数 Ne 最接近的 IMF 作为基准曲线。

            Ne = (depth_span / sr_est) / 0.405     （405 kyr 旋回个数）

        对每个 IMF 用 count_cycles 计周期数，取 |n-Ne| 相对偏差最小者；
        相对偏差 > tolerance → matched=False（调用方回退线性模型）。
        附加核验：所选 IMF 在 sr_est 下的 MTM 主谱峰对应周期须在
        405 kyr ±20% 以内，否则同样 matched=False。

        返回 dict:
            imf_index : 所选 IMF 在 imfs 中的下标（0 基）
            n_cycles  : 所选 IMF 周期数
            ne        : 期望 405 kyr 旋回个数（浮点）
            matched   : 是否通过容差 + 谱峰双重判定
            diag      : dict(per_imf=[{imf_index,n_cycles,rel_dev}...],
                             rel_dev, tolerance,
                             mtm_peak_period_kyr, mtm_spec_ok)
        """
        depth = np.asarray(depth, dtype=float)
        depth_span = float(depth.max() - depth.min())
        ne = (depth_span / float(sr_est)) / 0.405
        per_imf = []
        best = None
        for i, imf in enumerate(imfs):
            n, _, _ = self.count_cycles(imf)
            rel_dev = abs(n - ne) / ne
            per_imf.append(dict(imf_index=i, n_cycles=n, rel_dev=rel_dev))
            if best is None or rel_dev < best[2]:
                best = (i, n, rel_dev)
        i_sel, n_sel, dev_sel = best
        matched = bool(dev_sel <= tolerance)
        # ---- MTM 谱峰核验（sr_est 下的时间域）----
        t = self.depth_to_time(depth, sr_est)
        freq, power = self.mtm_spectrum(t, np.asarray(imfs[i_sel], dtype=float))
        pos = freq > 0
        f_pk = float(freq[pos][np.argmax(power[pos])])
        period_kyr = 1000.0 / f_pk
        mtm_spec_ok = bool(abs(period_kyr - 405.0) / 405.0 <= 0.20)
        if not mtm_spec_ok:
            matched = False
        return dict(imf_index=int(i_sel), n_cycles=int(n_sel), ne=float(ne),
                    matched=matched,
                    diag=dict(per_imf=per_imf, rel_dev=float(dev_sel),
                              tolerance=float(tolerance),
                              mtm_peak_period_kyr=float(period_kyr),
                              mtm_spec_ok=mtm_spec_ok))

    def build_age_model_405(self, depth, imf, anchor_phase='trough',
                            ref_age_fn=None):
        """
        以所选 IMF 的同相极值（峰→峰或谷→谷）为 405 kyr 锚点，
        构建分段线性非线性深度-年代模型（沉积速率随年代变化）。

        第 k 个锚点（k 从 0 起）:
            Age_k = ref_age_fn(d_anchor1) + k × 0.405 Myr
        ref_age_fn 为既有线性年龄模型（d→Age），保证与顶部线性模型衔接；
        None → 首锚点 Age=0。
        锚点间分段线性插值；深度网格两端按最近锚点区间的 SR
        （interval_sr = Δd/0.405）外推。

        返回:
            age_array : 与 depth 等长的 Age(Ma) 数组（严格单调递增，断言）
            anchor_df : DataFrame[anchor_no, depth, age, interval_sr]
                        （interval_sr 为该锚点→下一锚点区间的沉积速率，
                         末锚点为 NaN）
        """
        depth = np.asarray(depth, dtype=float)
        imf = np.asarray(imf, dtype=float)
        _, peaks, troughs = self.count_cycles(imf)
        anchors = troughs if anchor_phase == 'trough' else peaks
        if anchors.size < 2:
            raise ValueError(f"同相极值锚点不足（{anchors.size} 个），"
                             "无法构建 405 kyr 分段线性年龄模型。")
        a_depth = depth[anchors]
        a0 = float(ref_age_fn(a_depth[0])) if ref_age_fn is not None else 0.0
        a_age = a0 + np.arange(anchors.size, dtype=float) * 0.405
        interval_sr = np.diff(a_depth) / 0.405          # m/Myr
        if np.any(interval_sr <= 0):
            raise ValueError("锚点区间 SR 非正，锚点序列异常。")
        # 分段线性插值 + 两端按最近锚点区间 SR 外推
        age = np.interp(depth, a_depth, a_age)
        head = depth < a_depth[0]
        tail = depth > a_depth[-1]
        age[head] = a_age[0] - (a_depth[0] - depth[head]) / interval_sr[0]
        age[tail] = a_age[-1] + (depth[tail] - a_depth[-1]) / interval_sr[-1]
        assert np.all(np.diff(age) > 0), \
            "405 kyr 锚定模型：Age 必须随 Depth 严格单调递增"
        anchor_df = pd.DataFrame({
            'anchor_no': np.arange(1, anchors.size + 1),
            'depth': a_depth,
            'age': a_age,
            'interval_sr': np.append(interval_sr, np.nan)})
        return age, anchor_df

    # ------------------------------------------------------------------
    # eha 等效：滑动窗 MTM 演化谱
    # ------------------------------------------------------------------
    def eha(self, depth, signal, sr, win_m=150, step_m=10):
        """
        滑动窗 MTM 演化谱（深度域开窗，窗内按 SR 转时间域）。
        返回:
            depth_centers : 各窗中心深度 (m)
            freq          : 频率 (cycles/Myr)
            power2d       : shape (n_windows, n_freq) 功率矩阵（已按窗内最大值归一）
        """
        depth = np.asarray(depth, dtype=float)
        signal = np.asarray(signal, dtype=float)
        d_lo, d_hi = depth.min(), depth.max()
        centers = np.arange(d_lo + win_m / 2, d_hi - win_m / 2 + 1e-9, step_m)
        spectra = []
        freq_ref = None
        for c in centers:
            m = (depth >= c - win_m / 2) & (depth <= c + win_m / 2)
            if m.sum() < 64:
                spectra.append(None)
                continue
            t = self.depth_to_time(depth[m], sr)
            freq, power = self.mtm_spectrum(t, signal[m])
            if freq_ref is None:
                freq_ref = freq
            else:
                power = np.interp(freq_ref, freq, power)
            spectra.append(power)
        spectra = [s if s is not None else np.full_like(freq_ref, np.nan)
                   for s in spectra]
        power2d = np.vstack(spectra)
        # 每个窗按自身最大值归一，便于可视化对比
        row_max = np.nanmax(power2d, axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        power2d = power2d / row_max
        return centers, freq_ref, power2d


# ----------------------------------------------------------------------
# 简单自测
# ----------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    sr_true = 15.0
    depth = 1432.0 + np.arange(10084) * 0.1
    t = (depth - depth.min()) / sr_true
    sig = sum(a * np.sin(2 * np.pi * t / (p / 1000.0))
              for p, a in zip((405, 95, 40, 21), (6, 4, 2.5, 1.5)))
    sig += rng.normal(0, 0.5, sig.size)
    tuner = AstroTuner(n_jobs=2)
    scan = tuner.scan_sed_rates(depth, sig)
    sr_opt, ms = tuner.best_sr(scan, depth=depth, signal=sig)
    print(f"self-test: SR_true=15 -> best_sr={sr_opt:.3f}, match_score={ms:.3f}")
    assert abs(sr_opt - 15.0) <= 0.5, "best_sr 偏差超限"
    assert ms >= 0.5, "真实信号匹配度应较高"
    # 白噪声路径：应得低匹配度
    noise = rng.normal(0, 1, sig.size)
    scan_n = tuner.scan_sed_rates(depth, noise)
    _, ms_n = tuner.best_sr(scan_n, depth=depth, signal=noise)
    print(f"white-noise test: match_score={ms_n:.3f} (应 < 0.5)")
    assert ms_n < 0.5, "白噪声匹配度应较低"
    print("AstroTuner 自测通过。")
