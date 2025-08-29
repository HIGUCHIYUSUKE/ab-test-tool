# app.py
import math
from dataclasses import dataclass

import numpy as np
import streamlit as st
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)

st.set_page_config(page_title="A/Bテスト効果検証ツール（自社用）", layout="wide")


# ------------------------------
# Data model
# ------------------------------
@dataclass
class ABInput:
    left_sessions: int
    left_conversions: int
    right_sessions: int
    right_conversions: int


# ------------------------------
# Core stats helpers
# ------------------------------
def ztest_and_ci(inp: ABInput, alpha: float = 0.05, alternative: str = "two-sided"):
    """
    left と right の二群の比率比較（Z検定）
    返却の diff と diff_ci は「left − right」(＝前者−後者) に注意。
    UIでは「right − left（テスト−コントロール）」に変換して表示する。
    """
    counts = np.array([inp.left_conversions, inp.right_conversions])
    nobs = np.array([inp.left_sessions, inp.right_sessions])

    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative=alternative)

    # CVRとその (Wilson) 95%CI
    p_left = inp.left_conversions / inp.left_sessions
    p_right = inp.right_conversions / inp.right_sessions
    ci_l_low, ci_l_high = proportion_confint(
        inp.left_conversions, inp.left_sessions, alpha=alpha, method="wilson"
    )
    ci_r_low, ci_r_high = proportion_confint(
        inp.right_conversions, inp.right_sessions, alpha=alpha, method="wilson"
    )

    # 差のCI（正規近似、標本SE）
    se = math.sqrt(
        p_left * (1 - p_left) / inp.left_sessions
        + p_right * (1 - p_right) / inp.right_sessions
    )
    z = abs(norm.ppf(alpha / 2)) if alternative == "two-sided" else abs(norm.ppf(alpha))
    diff = p_left - p_right  # ここは「left − right」
    diff_low = diff - z * se
    diff_high = diff + z * se

    return {
        "pval": pval,
        "p_left": p_left,
        "p_right": p_right,
        "ci_left": (ci_l_low, ci_l_high),
        "ci_right": (ci_r_low, ci_r_high),
        "diff_lr": diff,  # left - right
        "diff_ci_lr": (diff_low, diff_high),
        "nobs": nobs,
    }


def quick_required_per_group(
    p_base: float, mde_pt: float, alpha: float, power: float, alternative: str
) -> int:
    """
    簡易サンプルサイズ（必要セッション/群）の近似式：
      n ≈ K * p(1-p) / d^2
      d : MDE（比率）, K = 2 * (z_{α-side} + z_{power})^2
    2側α=0.05, power=0.8 -> K ≈ 15.7 が目安。
    """
    d = max(mde_pt / 100.0, 1e-9)
    z_alpha = norm.ppf(1 - (alpha / 2 if alternative == "two-sided" else alpha))
    z_beta = norm.ppf(power)
    K = 2 * (z_alpha + z_beta) ** 2
    n = K * p_base * (1 - p_base) / (d**2)
    return math.ceil(n)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def yen(x: float) -> str:
    return f"¥{x:,.0f}"


def big_judgement_box(
    title: str, message: str, meta_line: str, detail_text: str, kind: str = "success"
):
    style = """
    <style>
    .judge-box{font-size:20px;font-weight:700;padding:18px 20px;border-radius:12px;margin:8px 0;border:1px solid}
    .judge-success{background:#e6f4ea;color:#135c27;border-color:#b7e1c0}
    .judge-warning{background:#fff4e5;color:#6f3b00;border-color:#ffd699}
    .judge-error{background:#fdecea;color:#611a15;border-color:#f5c6cb}
    .judge-meta{margin-top:6px;font-size:13px;opacity:.9}
    .judge-detail summary{cursor:pointer;color:#334155}
    </style>
    """
    cls = (
        "judge-success"
        if kind == "success"
        else "judge-warning" if kind == "warning" else "judge-error"
    )
    html = f"""{style}
    <div class="judge-box {cls}">
      <div style="font-size:18px;margin-bottom:4px"><strong>{title}</strong></div>
      <div>{message}</div>
      <div class="judge-meta">{meta_line}</div>
      <details class="judge-detail"><summary>詳細を表示</summary>
        <div style="font-size:13px;margin-top:6px">{detail_text}</div>
      </details>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ------------------------------
# Sidebar (inputs)
# ------------------------------
with st.sidebar.expander("検定設定（詳細）", expanded=False):
    alternative_label = st.selectbox("検定方法", ["両側検定", "片側検定（テスト群 > コントロール群）"])
    alternative = "two-sided" if alternative_label == "両側検定" else "larger"
    alpha = st.number_input(
        "有意水準 α", value=0.05, min_value=0.001, max_value=0.2, step=0.01, format="%.3f"
    )
    mde_pt = st.number_input(
        "実務的最小差（MDE, pt）",
        value=0.10,
        min_value=0.0,
        max_value=50.0,
        step=0.05,
        format="%.2f",
        help="この差以上ならビジネス的に意味があるとみなす最小差（pt）。",
    )
    target_power = st.number_input(
        "検出力（Hold判定の目安）", value=0.80, min_value=0.50, max_value=0.99, step=0.05
    )

with st.sidebar.form("inputs"):
    st.markdown("### データ入力")
    a_sessions = st.number_input("A：コントロール群 セッション数", value=31465, min_value=1, step=1)
    a_convs = st.number_input("A：コントロール群 CV数", value=1003, min_value=0, step=1)
    b_sessions = st.number_input("B：テスト群 セッション数", value=11773, min_value=1, step=1)
    b_convs = st.number_input("B：テスト群 CV数", value=313, min_value=0, step=1)

    add_c = st.checkbox("C群（任意）を追加する", value=False)
    if add_c:
        c_sessions = st.number_input("C：追加群 セッション数（任意）", value=0, min_value=0, step=1)
        c_convs = st.number_input("C：追加群 CV数（任意）", value=0, min_value=0, step=1)
        pair = st.selectbox("比較ペアを選択", ["A（コントロール） vs B（テスト）", "A vs C", "B vs C"])
    else:
        pair = "A（コントロール） vs B（テスト）"

    st.markdown("### ビジネス前提")
    monthly_sessions = st.number_input("月間セッション数", value=42000, step=100)
    cv_value = st.number_input(
        "1CVあたりの金額（円）",
        value=20000,
        step=1000,
        help="ECなら平均注文額（または粗利/件）、リードなら LTV×成約率 など。",
    )
    
    submitted = st.form_submit_button("分析開始！")

# ------------------------------
# Main
# ------------------------------
st.title("A/Bテスト効果検証ツール（自社用）")
st.write(
    "Z検定で有意差判定し、CVR差（**テスト − コントロール**）を月間セッション数と1CVあたり金額に外挿してインパクトを推計します。"
)

if not submitted:
    st.info("左の入力を設定し、「分析開始！」を押してください。")
    st.stop()

# 左右の並び（UIの前者＝left、後者＝right）
if pair == "A（コントロール） vs B（テスト）":
    left_label, right_label = "コントロール群 (A)", "テスト群 (B)"
    inp = ABInput(a_sessions, a_convs, b_sessions, b_convs)
elif pair == "A vs C":
    left_label, right_label = "コントロール群 (A)", "追加群 (C)"
    inp = ABInput(a_sessions, a_convs, max(1, c_sessions), c_convs)
else:
    left_label, right_label = "テスト群 (B)", "追加群 (C)"
    inp = ABInput(b_sessions, b_convs, max(1, c_sessions), c_convs)

res = ztest_and_ci(inp, alpha=alpha, alternative=alternative)

# 表示は「right − left（テスト − コントロール）」に統一
uplift_pt = (res["p_right"] - res["p_left"]) * 100
ci_low_rev = -res["diff_ci_lr"][1]  # (left-right)CI を反転して right-left に
ci_high_rev = -res["diff_ci_lr"][0]

# KPI
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(f"{left_label} CVR", fmt_pct(res["p_left"]))
    st.caption(f"95%CI: {fmt_pct(res['ci_left'][0])} – {fmt_pct(res['ci_left'][1])}")
    st.caption("計算: CV数 ÷ セッション数")
with c2:
    st.metric(f"{right_label} CVR", fmt_pct(res["p_right"]))
    st.caption(f"95%CI: {fmt_pct(res['ci_right'][0])} – {fmt_pct(res['ci_right'][1])}")
    st.caption("計算: CV数 ÷ セッション数")
with c3:
    diff_label = "差（テスト − コントロール）" if pair == "A（コントロール） vs B（テスト）" else "差（右 − 左）"
    st.metric(diff_label, f"{uplift_pt:.2f} pt")
    st.caption(f"差のCI: {ci_low_rev*100:.2f} – {ci_high_rev*100:.2f} pt")
    st.caption("計算: 右側のCVR − 左側のCVR")

st.divider()

# 判定（Go / Hold / No-Go）
pval = res["pval"]
is_sig = pval < alpha
meets_mde = abs(uplift_pt) >= mde_pt

# 簡易サンプルサイズ（不足＆日数）
p_base_for_power = res["p_left"]  # 基準は左側（A/Bならコントロール）
n_quick = quick_required_per_group(p_base_for_power, mde_pt, alpha, target_power, alternative)
shortfall = max(0, n_quick - int(min(res["nobs"])))
days = math.ceil(shortfall / daily_rate) if daily_rate > 0 else None

# 信頼度（p値ベースの段階表示）
conf_txt = "High" if pval < 0.01 else ("Medium" if pval < alpha else "Low")

st.subheader("判定")
detail_line = f"(p = {pval:.3g} {'<' if is_sig else '≥'} α = {alpha}). 差 {abs(uplift_pt):.2f}pt は MDE{'≧' if meets_mde else '<'}{mde_pt:.2f}pt。"
meta_line = f"P値：{pval:.3g} ／ 信頼度：{conf_txt}"

if is_sig and meets_mde:
    msg = "おめでとうございます！！今回の施策は効果が出た可能性が高く、再現性が期待できる結果です！単なる偶然ではなく、狙った効果が現れたと考えられます。"
    big_judgement_box("Go", msg, meta_line, detail_line, "success")
elif (not is_sig) and shortfall > 0:
    msg = "データ量が少なく、今回の結果では効果があったと断言できません。"
    extra = f" ／ 目安: 必要セッション/群 ≈ {n_quick:,}、不足 ≈ {shortfall:,}" + (
        f"、日数 ≈ {days}日（/群 {daily_rate:,}/日）" if days else ""
    )
    big_judgement_box("Hold", msg, meta_line + extra, detail_line, "warning")
elif is_sig and (not meets_mde):
    msg = "有意差はあるものの、差が小さく実務的インパクトは限定的です。追加検証を推奨します。"
    big_judgement_box("Hold", msg, meta_line, detail_line, "warning")
else:
    msg = "残念ですが、今回の結果では効果があったと断言できません。効果がゼロとは限りませんが、再現性は不明です。"
    big_judgement_box("No-Go", msg, meta_line, detail_line, "error")

st.divider()

# 事業インパクト（外挿）
st.subheader("月次インパクト推定（外挿）")
diff_for_impact = (res["p_right"] - res["p_left"])  # テスト − コントロール
add_cv = monthly_sessions * diff_for_impact
add_rev = add_cv * cv_value

add_cv_low = monthly_sessions * ci_low_rev
add_cv_high = monthly_sessions * ci_high_rev
add_rev_low = add_cv_low * cv_value
add_rev_high = add_cv_high * cv_value

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("追加CV数（推定）", f"{add_cv:,.0f} 件")
    st.caption(f"区間: {add_cv_low:,.0f} – {add_cv_high:,.0f}")
    st.caption("計算: (CVR差) × 月間セッション数")
with m2:
    st.metric("追加金額（推定）", yen(add_rev))
    st.caption(f"区間: {yen(add_rev_low)} – {yen(add_rev_high)}")
    st.caption("計算: 追加CV × 1CVあたりの金額")
with m3:
    lift_rel = (res["p_right"] / max(res["p_left"], 1e-12) - 1) * 100
    st.metric("リフト率（相対改善率）", f"{lift_rel:.1f}%")
    st.caption("計算: (テストCVR − コントロールCVR) ÷ コントロールCVR")

st.caption(
    "※ 1CVあたりの金額は、ECなら平均注文額（または粗利/件）、リード獲得なら LTV×成約率 など実態に合わせて設定してください。"
)

st.divider()
with st.expander("計算の根拠（かんたん説明）"):
    st.markdown(
        """
- **p値**：二群の比率のZ検定で「偶然この差が出る確率」。小さいほど偶然ではない。  
- **信頼度（表示ルール）**：High : p < 0.01 ／ Medium : 0.01 ≤ p < α（既定 0.05）／ Low : p ≥ α  
- **月次インパクト**：  
  - 追加CV = (CVR差) × 月間セッション数  
  - 追加金額 = 追加CV × 1CVあたりの金額  
- **簡易サンプルサイズ（目安）**：  
  - 必要セッション/群 ≈ `K × p(1−p) / d²`  
  - ここで `p` はベースCVR、`d` は MDE（比率）、`K = 2 × (z_{α-side} + z_{power})²`  
  - 2側・α=0.05・検出力80%なら **K ≈ 15.7**
        """
    )
