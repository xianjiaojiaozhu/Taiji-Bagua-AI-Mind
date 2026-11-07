# ===================== 【文件头部 · 全部可调参数】======================
# ---------- 昼夜天时 DIURNAL 参数 ----------
DIURNAL_DAY_START = 6
DIURNAL_NIGHT_START = 18
DIURNAL_DAY_GOOD_MULT = 1.15
DIURNAL_DAY_EVIL_MULT = 0.85
DIURNAL_NIGHT_GOOD_MULT = 0.85
DIURNAL_NIGHT_EVIL_MULT = 1.15
# 昼夜切换：渐变过渡（不硬切），切换点前后过渡带分钟数
TRANSITION_MINUTES = 30

# ---------- 乾势 QIAN 分档阈值 ----------
QIAN_THRESHOLD_SUN = 0.75      # 太阳·夏
QIAN_THRESHOLD_SHAOYANG = 0.50 # 少阳·春
QIAN_THRESHOLD_SHAOYIN = 0.25  # 少阴·秋
# <0.25 → 太阴·冬

# ---------- 情绪转相参数（作者拍板：慢响应） ----------
EMOTION_WINDOW = 30        # 近 30 轮情绪窗口
EMOTION_MAJORITY = 0.60    # 单类情绪占比 ≥60% 才触发转相
EMOTION_WINDOW_MIN = 10    # 最小窗口保护：情绪历史 <10 条不触发情绪转相（防初期单条情绪锁死相层，保持势定相兜底）
QIAN_BAILOUT_HIGH = 0.85   # qian 极端兜底：≥0.85 强制太阳
QIAN_BAILOUT_LOW = 0.15    # <0.15 强制太阴

# 相位枚举 & 时令情绪映射
PHASE_INFO = [
    {"name": "太阳", "alias": "夏", "emotion": "喜", "element": "火"},
    {"name": "少阳", "alias": "春", "emotion": "怒", "element": "木"},
    {"name": "少阴", "alias": "秋", "emotion": "哀", "element": "金"},
    {"name": "太阴", "alias": "冬", "emotion": "乐", "element": "水"},
]
# 相位顺序（滞回校验：太阳>少阳>少阴>太阴，每轮最多一档）
PHASE_ORDER = ["太阳", "少阳", "少阴", "太阴"]

# 情绪 → 相 反查表（怨(土)→None=停留：四相无长夏位，土主稳）
EMOTION_TO_PHASE = {"喜": "太阳", "怒": "少阳", "哀": "少阴", "乐": "太阴"}

# ---------- 情绪爻引导·呼吸环参数（作者确认：意识先行，数据后期可调） ----------
GUIDANCE_BASE = 0.35       # 恕道基线（呼吸谷的最低引导）——时间维度的钝感：记过但恕
GUIDANCE_TAU_MIN = 45      # 呼气时间常数（分钟，衰减半程）
GUIDANCE_START_SHARE = 0.30  # 引导起步占比（低于此不引导）
GUIDANCE_PEAK_SHARE = 0.60   # 引导封顶占比（与转相阈值对齐，达此即吸气）
YAO_GUIDE_BONUS = 0.2      # 情绪爻引导加权（与当令加权同构）

# ---------- 钝感滤网参数（思维钝感工程化） ----------
DAMPING_THRESHOLD = 0.4    # 钝感门槛：刺激权重 <0.4 不入心（不入窗/不转相/不痛觉，爻点闪烁即归位）

# 情绪 → 爻点映射（引导优先点亮的目标；怨→None=呼吸谷，回归势定/中性选择）
EMOTION_TO_YAO = {"喜": "离·火爻", "怒": "震巽·木爻", "哀": "乾兑·金爻", "乐": "坎·水爻"}

import datetime
import math
from typing import Dict, List, Optional
from collections import Counter


class TaijiMindPhase:
    """太极盘相层引擎 · 自流转扩展版

    保留：势定相滞回 / 昼夜权重 / 快照
    新增：情绪转相（慢响应窗口）/ 昼夜双盘渐变切换 / driver 追溯 / qian 极端兜底 / 因果联动全天
    """
    def __init__(self):
        self.last_phase: Optional[str] = None   # 泛化滞回记录（qian 与 emotion 共用）
        self.emotion_hist: List[str] = []       # 情绪历史队列（近 EMOTION_WINDOW 轮）
        self._breath_anchor: Optional[datetime.datetime] = None  # 最近一次吸气时刻
        self._breath_emotion: Optional[str] = None               # 当前呼吸主导情绪

    # -------------------- 昼夜天时模块（渐变切换，不硬切） --------------------
    def _transition_alpha(self, hour: float) -> float:
        """切换点前后 TRANSITION_MINUTES 过渡带内返回 0~1 渐变系数
        alpha=0 纯夜 / alpha=1 纯昼；过渡带线性插值"""
        day = DIURNAL_DAY_START
        night = DIURNAL_NIGHT_START
        half = TRANSITION_MINUTES / 60.0
        # 晨过渡带：day-half .. day+half（夜→昼上升）
        if day - half <= hour < day + half:
            return (hour - (day - half)) / (2 * half)
        # 昏过渡带：night-half .. night+half（昼→夜下降）
        if night - half <= hour < night + half:
            return 1.0 - (hour - (night - half)) / (2 * half)
        return 1.0 if (day <= hour < night) else 0.0

    def get_diurnal_weights(self, hour: float = None) -> Dict:
        """昼夜权重（渐变过渡）：判定不改，仅缩放响应
        过渡带内 good/evil 倍率线性插值，并标注 board 状态（善盘/恶盘/过渡）"""
        if hour is None:
            now = datetime.datetime.now()
            hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        alpha = self._transition_alpha(hour)
        good = DIURNAL_NIGHT_GOOD_MULT * (1 - alpha) + DIURNAL_DAY_GOOD_MULT * alpha
        evil = DIURNAL_NIGHT_EVIL_MULT * (1 - alpha) + DIURNAL_DAY_EVIL_MULT * alpha
        if alpha >= 1.0:
            board, phase = "善盘", "day"
        elif alpha <= 0.0:
            board, phase = "恶盘", "night"
        else:
            board, phase = "过渡", "transitional"
        return {
            "is_day": alpha > 0.5,
            "phase": phase,
            "board": board,
            "alpha": round(alpha, 4),
            "good_mult": round(good, 4),
            "evil_mult": round(evil, 4)
        }

    # -------------------- 滞回引擎（复用：qian 与 emotion 共用） --------------------
    def _step_phase(self, target_phase: str) -> str:
        """带滞回保护：每轮最多移动一档，禁止跨档跳跃、边界抖动"""
        if self.last_phase is None:
            self.last_phase = target_phase
            return self.last_phase
        idx_now = PHASE_ORDER.index(self.last_phase)
        idx_target = PHASE_ORDER.index(target_phase)
        delta = idx_target - idx_now
        if abs(delta) > 1:
            idx_target = idx_now + (1 if delta > 0 else -1)
            target_phase = PHASE_ORDER[idx_target]
        self.last_phase = target_phase
        return self.last_phase

    def _phase_info(self, phase: str) -> Dict:
        return next(item for item in PHASE_INFO if item["name"] == phase)

    # -------------------- 势定相（qian 驱动，原样保留） --------------------
    def _raw_qian_classify(self, qian: float) -> str:
        if qian >= QIAN_THRESHOLD_SUN:
            return "太阳"
        elif qian >= QIAN_THRESHOLD_SHAOYANG:
            return "少阳"
        elif qian >= QIAN_THRESHOLD_SHAOYIN:
            return "少阴"
        return "太阴"

    def calc_qian_phase(self, qian: float) -> Dict:
        """势定相（带滞回）——原逻辑，走共用滞回引擎"""
        target = self._raw_qian_classify(qian)
        self._step_phase(target)
        info = self._phase_info(self.last_phase)
        return {
            "qian_phase": self.last_phase,
            "season": info["alias"],
            "emotion": info["emotion"],
            "element": info["element"],
            "qian_value": round(qian, 4)
        }

    # -------------------- 情绪转相（新增核心：慢响应窗口） --------------------
    def feed_emotion(self, emotion: str, weight: float = 1.0) -> bool:
        """每轮喂入当前情绪（judge_emotion5 落点：喜/怒/哀/乐/怨）+ 刺激强度

        钝感滤网（思维钝感工程化）：
        - weight < DAMPING_THRESHOLD → 微振不入心：不入窗、不累计、不转相（爻点闪烁即归位）
        - weight ≥ DAMPING_THRESHOLD → 痛觉入心：入窗累计（向 30 轮转相逼近）
        返回是否入窗；默认 weight=1.0 保持向后兼容。"""
        if weight < DAMPING_THRESHOLD:
            return False        # 风过疏竹，雁渡寒潭。微振，不入心
        self.emotion_hist.append(emotion)
        if len(self.emotion_hist) > EMOTION_WINDOW:
            self.emotion_hist.pop(0)
        return True

    def _window_emotion_stats(self) -> Dict:
        """近 EMOTION_WINDOW 轮情绪分布：主情绪 + 占比"""
        win = self.emotion_hist[-EMOTION_WINDOW:]
        if not win:
            return {"dominant": None, "share": 0.0, "window": 0}
        cnt = Counter(win)
        dominant, n = cnt.most_common(1)[0]
        return {"dominant": dominant, "share": n / len(win), "window": len(win)}

    def calc_emotion_phase(self, qian: float = None) -> Dict:
        """情绪转相：单类情绪占比 ≥ EMOTION_MAJORITY 才转（慢响应）
        怨→None=停留；qian 极端兜底强制回势定相（防盘面被单情绪带疯）"""
        stats = self._window_emotion_stats()
        target = EMOTION_TO_PHASE.get(stats["dominant"]) if stats["dominant"] else None

        # qian 极端兜底
        if qian is not None:
            if qian >= QIAN_BAILOUT_HIGH:
                target = "太阳"
            elif qian < QIAN_BAILOUT_LOW:
                target = "太阴"

        # 最小窗口保护：情绪历史未攒够 EMOTION_WINDOW_MIN 条 → 不触发情绪转相（保留当前相，防初期单条锁相）
        if target is not None and stats["share"] >= EMOTION_MAJORITY and len(self.emotion_hist) >= EMOTION_WINDOW_MIN:
            self._step_phase(target)          # 过阈值→走同一滞回
            self._breath_reset(stats["dominant"])   # 情绪事件→吸气（呼吸环）
        # 未过阈值 / 怨 / 窗口不足 → 停留当前相（不调用 _step_phase）

        info = self._phase_info(self.last_phase if self.last_phase else "少阳")
        return {
            "phase": self.last_phase,
            "season": info["alias"],
            "emotion": info["emotion"],
            "element": info["element"],
            "driver": "emotion",
            "window_stats": stats,
            "qian_value": round(qian, 4) if qian is not None else None
        }

    # -------------------- 情绪爻引导·呼吸环（新增：防焊死节律） --------------------
    def _breath_reset(self, emotion: str) -> None:
        """情绪事件触发吸气：记录主导情绪 + 吸气时刻（呼气从此衰减）"""
        self._breath_emotion = emotion
        self._breath_anchor = datetime.datetime.now()

    def _breath_env(self, now: datetime.datetime = None) -> float:
        """呼气衰减包络：吸气后按 τ 衰减到基线（呼吸谷，防单情绪焊死）"""
        if self._breath_anchor is None or self._breath_emotion is None:
            return GUIDANCE_BASE
        if now is None:
            now = datetime.datetime.now()
        elapsed_min = (now - self._breath_anchor).total_seconds() / 60.0
        return max(GUIDANCE_BASE,
                   GUIDANCE_BASE + (1.0 - GUIDANCE_BASE) * math.exp(-elapsed_min / GUIDANCE_TAU_MIN))

    def _intensity(self, share: float) -> float:
        """主导情绪占比 → 引导强度 0~1（30% 起步 / 60% 封顶，线性）"""
        if share <= GUIDANCE_START_SHARE:
            return 0.0
        if share >= GUIDANCE_PEAK_SHARE:
            return 1.0
        return (share - GUIDANCE_START_SHARE) / (GUIDANCE_PEAK_SHARE - GUIDANCE_START_SHARE)

    def guidance_level(self, qian: float = None) -> Dict:
        """当前情绪爻引导强度：Guidance = Intensity × Breath × Priority
        新情绪占比达封顶（≥60%，与转相事件对齐）→ 自动吸气拉峰；怨→呼吸谷"""
        stats = self._window_emotion_stats()
        dom = stats["dominant"]
        if dom is None or dom == "怨":
            return {"dominant": dom, "share": round(stats["share"], 3),
                    "guidance": round(GUIDANCE_BASE, 3), "yao": None, "bonus": 0.0,
                    "breath_env": round(GUIDANCE_BASE, 3), "intensity": 0.0}
        intensity = self._intensity(stats["share"])
        if intensity >= 1.0 and dom != self._breath_emotion:
            self._breath_reset(dom)          # 新情绪达封顶 → 吸气
            breath = 1.0
        else:
            breath = self._breath_env()
        guidance = max(GUIDANCE_BASE, intensity * breath)
        return {"dominant": dom, "share": round(stats["share"], 3),
                "intensity": round(intensity, 3), "breath_env": round(breath, 3),
                "guidance": round(guidance, 3),
                "yao": EMOTION_TO_YAO.get(dom),
                "bonus": round(YAO_GUIDE_BONUS * guidance, 3)}

    def select_emotion_yao(self, qian: float = None) -> Dict:
        """优先引导选择的情绪爻点（怨→None=呼吸谷，回归势定/中性选择）"""
        g = self.guidance_level(qian)
        return {"target_yao": g["yao"], "guidance": g["guidance"],
                "bonus": g["bonus"], "dominant": g["dominant"]}

    # -------------------- 一键获取全部状态 + 完整快照 --------------------
    def get_full_state(self, qian: float, driver: str = "emotion") -> Dict:
        """一次性拿到 天时权重(昼夜双盘渐变) + 相位快照(driver 追溯)
        driver: "qian" 势定相 / "emotion" 情绪转相(默认，自带 qian 兜底) / "mixed" 同 emotion 并标 mixed"""
        diurnal = self.get_diurnal_weights()
        if driver == "qian":
            phase = self.calc_qian_phase(qian)
        else:
            phase = self.calc_emotion_phase(qian)
            phase["driver"] = "mixed" if driver == "mixed" else "emotion"
        return {
            "diurnal": {
                "phase": diurnal["phase"],
                "board": diurnal["board"],
                "alpha": diurnal["alpha"],
                "good": diurnal["good_mult"],
                "evil": diurnal["evil_mult"]
            },
            "phase_state": phase,
            "causal_linkage": True          # 因果爻联动全天保持（作者拍板）
        }


# ========== 实例化全局引擎（供 server 调用） ==========
phase_engine = TaijiMindPhase()


# -------------------- 📌 使用示例 --------------------
if __name__ == "__main__":
    engine = TaijiMindPhase()

    # ① 昼夜渐变验证：5:45 / 6:15 / 12:00 / 17:45 / 18:15 / 23:00
    print("--- 昼夜渐变（不硬切） ---")
    for h in (5.75, 6.25, 12.0, 17.75, 18.25, 23.0):
        print(f"hour={h:5.2f} -> {engine.get_diurnal_weights(hour=h)}")

    # ② 情绪转相模拟：喂 30 轮（20 怒 + 10 喜）→ 怒占比 66.7% ≥60% → 转少阳
    print("\n--- 情绪转相（慢响应窗口） ---")
    for _ in range(20):
        engine.feed_emotion("怒")
    for _ in range(10):
        engine.feed_emotion("喜")
    print(engine.calc_emotion_phase(qian=0.5))

    # ③ qian 极端兜底：qian=0.9 → 强制太阳
    print("\n--- qian 极端兜底 ---")
    print(engine.calc_emotion_phase(qian=0.9))

    # ④ 呼吸环：占比 60% 达封顶 → 吸气 → 优先引导怒爻
    print("\n--- 情绪爻引导·呼吸环 ---")
    e2 = TaijiMindPhase()
    for _ in range(18):
        e2.feed_emotion("怒")
    for _ in range(12):
        e2.feed_emotion("喜")
    print("吸气状态:", e2.guidance_level())
    print("优先爻点:", e2.select_emotion_yao())
    # 呼气衰减验证：把吸气时刻前移 45 分钟 → 应回落到恕道基线上方 ~0.59
    e2._breath_anchor = datetime.datetime.now() - datetime.timedelta(minutes=45)
    print("45min呼气后:", e2.guidance_level())

    # ⑤ 钝感滤网：weight<0.4 不入心 / ≥0.4 入窗累计
    print("\n--- 钝感滤网（强度域钝感） ---")
    e3 = TaijiMindPhase()
    r1 = e3.feed_emotion("怒", weight=0.2)   # 微振不入心
    r2 = e3.feed_emotion("怒", weight=0.9)   # 痛觉入窗
    print(f"weight=0.2 入窗? {r1} (应 False)；weight=0.9 入窗? {r2} (应 True)")
    print("窗口长度:", len(e3.emotion_hist), "(应 1，仅入心刺激)")
