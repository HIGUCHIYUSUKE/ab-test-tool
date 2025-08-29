# app.py
import math
from dataclasses import dataclass

import numpy as np
import streamlit as st
from scipy.stats import norm
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
)

st.set_page_config(page_title="A/Bテスト効果検証ツール（自社用）", layout="wide")

# ============== Helpers ==============
@dataclass
class ABInput:
    left_sessions: int
    left_conversions: int
    right_sessions: int
    right_conversions: int

def ztest_and_ci(inp: ABInput, alpha: float = 0.05, alternative: str = "two-sided"):
    """二群の比率比較（Z検定）。返却の差は left-right（前者−後者）。UI側で right-left に変換して使用。"""
    counts = np.array([inp.left_conversions, inp.right_conversions])
    nobs = np.array([inp.left_sessions, inp.right_sessions])
    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative=alternative)

    p_left = inp.left_conversions / inp.left_sessions
    p_right = inp.right_conversions / inp.right_sessions

    # 各群のCI（Wilson）
    ci_l_low, ci_l_high = proportion_confint(
        inp.left_conversions, inp.left_sessions, alpha=alpha, method="wilson"
    )
    ci_r_low, ci_r_high = proportion_confint(
        inp.right_conversions, inp.right_sessions, alpha=alpha, method="wilson"
    )

    # 差(left-right)のCI（正規近似）
    se = math.sqrt(
        p_left*(1-p_left)/inp.left_sessions + p_right*(1-p_right)/inp.right_sessions
    )
    z = abs(norm.ppf(alpha/2)) if alternative == "two-sided" else abs(norm.ppf(alpha))
    diff_lr = p_left - p_right
    diff_low_lr  = diff_lr - z*se
    diff_high_lr = diff_lr + z*se

    return {
        "pval": pval,
        "p_left": p_left,
        "p_right": p_right,
        "ci_left": (ci_l_low, ci_l_high),
        "ci_right": (ci_r_low, ci_r_high),
        "diff_lr": diff_lr,
        "diff_ci_lr": (diff_low_lr, diff_high_lr),
        "nobs": nobs,
    }

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def yen(x: float) -> str:
    return f"¥{x:,.0f}"

def big_judgement_box(title: str, message: str, meta_line: str, detail_text: str, kind: str = "success"):
    style = """
    <style>
    .judge-box{font-size:20px;font-weight:700;padding:18px 20px;border-radius:12px;margin:8px 0;border:1px solid}
    .judge-success{background:#e6f4ea;color:#135c27;border-color:#b7e1c0}
    .judge-warning{background:#fff4e5;color:#6f3b00;border-color:#ffd699}
    .judge-error{background:#fdecea;color:#611a15;border-color:#f5c6cb}
    .judge-meta{margin-top:8px;font-size:13px;opacity:.9}
    .judge-detail summary{cursor:pointer;color:#334155}
    </style>
    """
    cls = "judge-success" if kind=="success" else ("judge-warning" if kind=="warning" else "judge-error")
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

def quick_required_per_group(p_base: float, mde_pt: float, alpha: float, power: float, alternative: str) -> int:
    """簡易サンプルサイズ（必要セッション/群）。 n ≈ K * p(1-p) / d^2, d=MDE(比率), K=2*(z_{α-side}+z_{β})^2"""
    d = max(mde_pt/100.0, 1e-9)
    z_alpha = norm.ppf(1 - (alpha/2 if alternative=="two-sided" else alpha))
    z_beta  = norm.ppf(power)
    K = 2 * (z_alpha + z_beta)**2
    n = K * p_base * (1-p_base) / (d**2)
    return math.ceil(n)

# ============== Sidebar ==============
with st.sidebar.expander("検定設定（詳細）", expanded=False):
    alt_label = st.selectbox("検定方法", ["両側検定", "片側検定（テスト群 > コントロール群）"])
    alternative = "two-sided" if alt_label=="両側検定" else "larger"
    alpha = st.number_input("有意水準 α", value=0.05, min_value=0.001, max_value=0.2, step=0.01, format="%.3f")
    mde_pt = st.number_input("実務的最小差（MDE, pt）", value=0.10, min_value=0.0, max_value=50.0, step=0.05, format="%.2f")
    target_power = st.number_input("検出力（Hold判定の目安）", value=0.80, min_value=0.50, max_value=0.99, step=0.05)

with st.sidebar.form("inputs"):
    st.markdown("### データ入力")
    a_sessions = st.number_input("A：コントロール群 セッション数", value=31465, min_value=1, step=1)
    a_convs    = st.number_input("A：コントロール群 CV数", value=1003, min_value=0, step=1)
    b_sessions = st.number_input("B：テスト群 セッション数", value=11773, min_value=1, step=1)
    b_convs    = st.number_input("B：テスト群 CV数", value=313, min_value=0, step=1)

    add_c = st.checkbox("C群（任意）を追加する", value=False)
    if add_c:
        c_sessions = st.number_input("C：追加群 セッション数（任意）", value=0, min_value=0, step=1)
        c_convs    = st.number_input("C：追加群 CV数（任意）", value=0, min_value=0, step=1)
        pair = st.selectbox("比較ペアを選択", ["A（コントロール） vs B（テスト）", "A vs C", "B vs C"])
    else:
        pair = "A（コントロール） vs B（テスト）"

    st.markdown("### ビジネス前提")
    monthly_sessions = st.number_input("月間セッション数（対象範囲）", value=42000, step=100)
    cv_value = st.number_input("1CVあたりの金額（円）", value=20000, step=1000, help="EC:平均注文額(粗利)/件、リード:LTV×成約率 など実態で設定")
    daily_rate = st.number_input("1日あたりの各群セッション（目安・任意）", value=0, step=100, help="不足セッションを日数に変換するため（任意）")

    submitted = st.form_submit_button("分析開始！")

# ============== Main ==============
st.title("A/Bテスト効果検証ツール（自社用）")
st.write("Z検定で有意差判定し、CVR差（**テスト − コントロール**）を月間セッション数と1CVあたり金額に外挿してインパクトを推計します。")

if not submitted:
    st.info("左の入力を設定し、「分析開始！」を押してください。")
    st.stop()

# 左右（UIの前者＝left、後者＝right）を構成
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

# 表示は right-left（テスト − コントロール）
uplift_pt_
