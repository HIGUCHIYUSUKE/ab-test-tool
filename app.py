
import math
from dataclasses import dataclass
import numpy as np
import streamlit as st
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from scipy.stats import norm

st.set_page_config(page_title="A/Bテスト効果検証ツール（自社用）", layout="wide")

# =====================
# Helpers
# =====================
@dataclass
class ABInput:
    sessions_a: int
    conversions_a: int
    sessions_b: int
    conversions_b: int

def ztest_and_ci(a: ABInput, alpha: float = 0.05, alternative: str = "two-sided"):
    counts = np.array([a.conversions_a, a.conversions_b])
    nobs = np.array([a.sessions_a, a.sessions_b])
    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative=alternative)

    # Wilson CI for each proportion
    ci_a_low, ci_a_high = proportion_confint(a.conversions_a, a.sessions_a, alpha=alpha, method="wilson")
    ci_b_low, ci_b_high = proportion_confint(a.conversions_b, a.sessions_b, alpha=alpha, method="wilson")

    # Difference CI using normal approximation with sample SE
    p1 = a.conversions_a / a.sessions_a
    p2 = a.conversions_b / a.sessions_b
    se = math.sqrt(p1*(1-p1)/a.sessions_a + p2*(1-p2)/a.sessions_b)
    z = abs(norm.ppf(alpha/2)) if alternative == "two-sided" else abs(norm.ppf(alpha))
    diff_low = (p1 - p2) - z*se
    diff_high = (p1 - p2) + z*se

    return {
        "stat": stat,
        "pval": pval,
        "p1": p1,
        "p2": p2,
        "ci_p1": (ci_a_low, ci_a_high),
        "ci_p2": (ci_b_low, ci_b_high),
        "diff": p1 - p2,
        "diff_ci": (diff_low, diff_high),
        "nobs": nobs
    }

def fmt_pct(x): return f"{x*100:.2f}%"
def yen(x: float): return f"¥{x:,.0f}"

def big_judgement_box(title, message, meta_line, detail_text, kind="success"):
    style = """
    <style>
    .judge-box {font-size:20px;font-weight:700;padding:18px 20px;border-radius:12px;margin:8px 0;border:1px solid}
    .judge-success {background:#e6f4ea;color:#135c27;border-color:#b7e1c0}
    .judge-warning {background:#fff4e5;color:#6f3b00;border-color:#ffd699}
    .judge-error {background:#fdecea;color:#611a15;border-color:#f5c6cb}
    .judge-meta {margin-top:6px;font-size:13px;opacity:.9}
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

def compute_needed_sample(p_base: float, mde_pt: float, alpha: float, power: float, alternative: str):
    """Return per-group sample size required to detect MDE with given power.
    MDE is in percentage points (pt). We approximate using p_base vs p_base + delta.
    """
    delta = mde_pt / 100.0
    p2 = min(max(p_base + delta, 1e-9), 1-1e-9)
    eff = proportion_effectsize(p_base, p2)
    if eff == 0:
        return float("inf")
    model = NormalIndPower()
    try:
        n_per_group = model.solve_power(effect_size=eff, power=power, alpha=alpha, alternative='larger' if alternative=='larger' else 'two-sided')
    except Exception:
        n_per_group = float("inf")
    return math.ceil(n_per_group)

# =====================
# Sidebar (inputs)
# =====================
with st.sidebar.expander("検定設定（詳細）", expanded=False):
    alternative_label = st.selectbox("検定方法", ["両側検定", "片側検定（テスト群 > コントロール群）"])
    alternative = "two-sided" if alternative_label == "両側検定" else "larger"
    alpha = st.number_input("有意水準 α", value=0.05, min_value=0.001, max_value=0.2, step=0.01, format="%.3f")
    mde_pct = st.number_input("実務的最小差（MDE, pt）", value=0.10, min_value=0.0, max_value=50.0, step=0.05, format="%.2f")
    target_power = st.number_input("判定の十分データ基準（検出力）", value=0.8, min_value=0.5, max_value=0.99, step=0.05, help="Hold（データ量が少ない）判定で用いる参考値。")

with st.sidebar.form("inputs"):
    st.markdown("### データ入力")
    # A (control)
    sessions_a = st.number_input("A：コントロール群 セッション数", value=31465, step=1, min_value=1)
    conv_a = st.number_input("A：コントロール群 CV数", value=1003, step=1, min_value=0)

    # B (treatment)
    sessions_b = st.number_input("B：テスト群 セッション数", value=11773, step=1, min_value=1)
    conv_b = st.number_input("B：テスト群 CV数", value=313, step=1, min_value=0)

    add_c = st.checkbox("C群（任意）を追加する", value=False)
    if add_c:
        sessions_c = st.number_input("C：追加群 セッション数（任意）", value=0, step=1, min_value=0)
        conv_c = st.number_input("C：追加群 CV数（任意）", value=0, step=1, min_value=0)
        pair = st.selectbox("比較ペアを選択", ["A（コントロール） vs B（テスト）", "A vs C", "B vs C"])
    else:
        pair = "A（コントロール） vs B（テスト）"

    st.markdown("### ビジネス前提")
    monthly_traffic = st.number_input("月間セッション数", value=42000, step=100)
    cv_value = st.number_input("1CVあたりの金額（円）", value=20000, step=1000, help="例：平均注文額（売上）や、粗利ベースでの1件あたり価値")

    submitted = st.form_submit_button("分析開始！")

st.title("A/Bテスト効果検証ツール（自社用）")
st.write("Z検定で有意差判定し、CVR差（uplift）を月間セッション数と1CVあたり金額に外挿してインパクトを推計します。")

if not submitted:
    st.info("左の入力を設定し、「分析開始！」を押してください。")
else:
    # Resolve pair selection
    if pair == "A（コントロール） vs B（テスト）":
        inp = ABInput(sessions_a, conv_a, sessions_b, conv_b)
        labels = ("コントロール群 (A)", "テスト群 (B)")
        p_base_for_power = (conv_a / sessions_a) if sessions_a > 0 else 0.0
    elif pair == "A vs C":
        inp = ABInput(sessions_a, conv_a, max(1, sessions_c), conv_c)
        labels = ("コントロール群 (A)", "追加群 (C)")
        p_base_for_power = (conv_a / sessions_a) if sessions_a > 0 else 0.0
    else:
        inp = ABInput(sessions_b, conv_b, max(1, sessions_c), conv_c)
        labels = ("テスト群 (B)", "追加群 (C)")
        p_base_for_power = (conv_b / sessions_b) if sessions_b > 0 else 0.0

    res = ztest_and_ci(inp, alpha=alpha, alternative=alternative)

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{labels[0]} CVR", fmt_pct(res["p1"]))
        st.caption(f"95%CI: {fmt_pct(res['ci_p1'][0])} – {fmt_pct(res['ci_p1'][1])}")
        st.caption("計算: CV数 ÷ セッション数")
    with col2:
        st.metric(f"{labels[1]} CVR", fmt_pct(res["p2"]))
        st.caption(f"95%CI: {fmt_pct(res['ci_p2'][0])} – {fmt_pct(res['ci_p2'][1])}")
        st.caption("計算: CV数 ÷ セッション数")
    with col3:
        uplift_pt = (res["p1"] - res["p2"]) * 100
        st.metric("差（前者 − 後者）", f"{uplift_pt:.2f} pt")
        st.caption(f"差のCI: {res['diff_ci'][0]*100:.2f} – {res['diff_ci'][1]*100:.2f} pt")
        st.caption("計算: 前者のCVR − 後者のCVR")

    st.divider()

    # 判定ロジック（Go / Hold / No-Go）
    pval = res["pval"]
    is_sig = pval < alpha                          # 統計的有意差
    meets_mde = abs(uplift_pt) >= mde_pct          # 実務的閾値
    n_required = compute_needed_sample(p_base_for_power, mde_pct, alpha, target_power, alternative)  # 目安サンプル
    low_sample = (min(res["nobs"]) < n_required)

    # 信頼度テキスト
    conf_txt = "High" if pval < 0.01 else ("Medium" if pval < alpha else "Low")

    st.subheader("判定  ")
    detail_line = f"(p = {pval:.3g} {'<' if is_sig else '≥'} α = {alpha}). 差 {abs(uplift_pt):.2f}pt は MDE{'≧' if meets_mde else '<'}{mde_pct:.2f}pt。"
    meta_line = f"P値：{pval:.3g} ／ 信頼度：{conf_txt}"

    if is_sig and meets_mde:
        msg = "おめでとうございます！！今回の施策は効果が出た可能性が高く、再現性が期待できる結果です！単なる偶然ではなく、狙った効果が現れたと考えられます。"
        big_judgement_box("Go", msg, meta_line, detail_line, "success")
    elif (not is_sig) and low_sample:
        msg = "データ量が少なく、今回の結果では効果があったと断言できません。"
        big_judgement_box("Hold", msg, meta_line + f" ／ 参考: 必要サンプル/群 ≈ {n_required:,}、実測 min(群) = {int(min(res['nobs'])):,}", detail_line, "warning")
    elif is_sig and (not meets_mde):
        msg = "有意差はあるものの、差が小さく実務的インパクトは限定的です。追加検証を推奨します。"
        big_judgement_box("Hold", msg, meta_line, detail_line, "warning")
    else:
        msg = "残念ですが、今回の結果では効果があったと断言できません。効果がゼロとは限りませんが、再現性は不明です。"
        big_judgement_box("No-Go", msg, meta_line, detail_line, "error")

    st.divider()

    # 事業インパクト（外挿）
    st.subheader("月次インパクト推定（外挿）")
    additional_conversions = monthly_traffic * res["diff"]
    additional_revenue = additional_conversions * cv_value

    # 信頼区間の下限/上限
    add_conv_low = monthly_traffic * res["diff_ci"][0]
    add_conv_high = monthly_traffic * res["diff_ci"][1]
    add_rev_low = add_conv_low * cv_value
    add_rev_high = add_conv_high * cv_value

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("追加CV数（推定）", f"{additional_conversions:,.0f} 件")
        st.caption(f"区間: {add_conv_low:,.0f} – {add_conv_high:,.0f}")
        st.caption("計算: (CVR差) × 月間セッション数")
    with c2:
        st.metric("追加金額（推定）", yen(additional_revenue))
        st.caption(f"区間: {yen(add_rev_low)} – {yen(add_rev_high)}")
        st.caption("計算: 追加CV × 1CVあたりの金額")
    with c3:
        uplift_rel = (res["p1"]/max(res["p2"], 1e-9) - 1) * 100
        st.metric("リフト率（相対改善率）", f"{uplift_rel:.1f}%")
        st.caption("計算: (テストCVR − コントロールCVR) ÷ コントロールCVR")

    st.caption("※ 1CVあたりの金額は、ECなら平均注文額（または粗利/件）、リード獲得なら LTV×成約率 など実態に合わせて設定してください。")

    st.divider()
    with st.expander("計算の根拠（かんたん説明）"):
        st.markdown("""
- **p値**：二群の比率のZ検定で「偶然この差が出る確率」。小さいほど偶然ではない。  
- **MDE**：実務上「意味がある」と判断する最小の差。  
- **Hold（データ量不足）**：MDEを検出するのに必要なサンプル（検出力）に満たない場合の保守的判定。  
- **95%CI**：CVRの推定誤差の範囲。  
- **月次インパクト**：  
  - 追加CV = (CVR差) × 月間セッション数  
  - 追加金額 = 追加CV × 1CVあたりの金額
""")
