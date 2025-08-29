
import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="A/B/Cテスト効果検証ツール（自社用）", layout="wide")

# =====================
# Helpers & Types
# =====================
@dataclass
class Group:
    name: str
    sessions: int
    conv: int

@dataclass
class TestResult:
    group_name: str
    p_ctrl: float
    p_test: float
    diff: float
    diff_ci: Tuple[float, float]
    p_value: float
    decision: str  # "go" | "hold" | "nogo"
    lift: Optional[float]
    add_cv: float
    add_cv_low: float
    add_cv_high: float
    add_rev: float
    add_rev_low: float
    add_rev_high: float

def fmt_pct(x: float) -> str: return f"{x*100:.2f}%"
def fmt_pt(x: float) -> str: return f"{x*100:.2f} pt"
def yen(x: float) -> str: return f"¥{x:,.0f}"

def ztest_and_ci(ctrl: Group, test: Group, alpha: float, alternative: str):
    counts = np.array([test.conv, ctrl.conv])
    nobs   = np.array([test.sessions, ctrl.sessions])
    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative=alternative)

    p1 = ctrl.conv / max(ctrl.sessions, 1)
    p2 = test.conv / max(test.sessions, 1)

    # CI for each proportion (Wilson)
    ci_ctrl = proportion_confint(ctrl.conv, ctrl.sessions, alpha=alpha, method="wilson")
    ci_test = proportion_confint(test.conv, test.sessions, alpha=alpha, method="wilson")

    # CI for difference (normal approx)
    se = math.sqrt(p1*(1-p1)/max(ctrl.sessions,1) + p2*(1-p2)/max(test.sessions,1))
    z = abs(norm.ppf(alpha/2)) if alternative == "two-sided" else abs(norm.ppf(alpha))
    diff_low = (p2 - p1) - z*se
    diff_high = (p2 - p1) + z*se

    return {"p_ctrl": p1, "p_test": p2, "p_value": pval, "diff": p2-p1, "diff_ci": (diff_low, diff_high),
            "ci_ctrl": ci_ctrl, "ci_test": ci_test}

def min_sample_per_arm(p_bar: float, mde_abs: float) -> int:
    if mde_abs <= 0: return 0
    n = 16.0 * p_bar * (1 - p_bar) / (mde_abs ** 2)
    return int(math.ceil(n))

def judge(ctrl: Group, test: Group, alpha: float, mde_pt: float, alternative: str,
          monthly_sessions: int, unit_value: float) -> TestResult:
    res = ztest_and_ci(ctrl, test, alpha, alternative)
    p_ctrl, p_test = res["p_ctrl"], res["p_test"]
    diff, (dl, dh), pval = res["diff"], res["diff_ci"], res["p_value"]
    mde_abs = mde_pt/100.0

    # HOLD rules
    avg_p = (p_ctrl + p_test) / 2.0
    n_min = min_sample_per_arm(avg_p, mde_abs)
    data_low = (ctrl.conv < 100) or (test.conv < 100)
    wide_ci = (abs(dh - dl) > 0.005)  # >0.5pt
    power_low = (ctrl.sessions < n_min) or (test.sessions < n_min)

    if pval < alpha and abs(diff) >= mde_abs:
        decision = "go"
    elif (pval >= alpha) and (data_low or wide_ci or power_low):
        decision = "hold"
    else:
        decision = "nogo"

    lift = None
    if p_ctrl > 0: lift = (p_test - p_ctrl) / p_ctrl

    # Monthly impact
    add_cv = max(0.0, monthly_sessions * diff)
    add_cv_low = max(0.0, monthly_sessions * dl)
    add_cv_high = max(0.0, monthly_sessions * dh)
    add_rev = add_cv * unit_value
    add_rev_low = add_cv_low * unit_value
    add_rev_high = add_cv_high * unit_value

    return TestResult(
        group_name=test.name, p_ctrl=p_ctrl, p_test=p_test, diff=diff, diff_ci=(dl, dh),
        p_value=pval, decision=decision, lift=lift,
        add_cv=add_cv, add_cv_low=add_cv_low, add_cv_high=add_cv_high,
        add_rev=add_rev, add_rev_low=add_rev_low, add_rev_high=add_rev_high
    )

def result_box(title: str, decision: str, message: str):
    if decision == "go": st.success(title); st.markdown(f"### {message}")
    elif decision == "hold": st.warning(title); st.markdown(f"### {message}")
    else: st.error(title); st.markdown(f"### {message}")

def build_summary_df(results: List[TestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "群": r.group_name, "コントロールCVR": r.p_ctrl, "テストCVR": r.p_test,
            "差": r.diff, "差_下限": r.diff_ci[0], "差_上限": r.diff_ci[1],
            "p値": r.p_value, "判定": r.decision, "リフト率": r.lift,
            "追加CV": r.add_cv, "追加CV_下限": r.add_cv_low, "追加CV_上限": r.add_cv_high,
            "追加売上": r.add_rev, "追加売上_下限": r.add_rev_low, "追加売上_上限": r.add_rev_high
        })
    return pd.DataFrame(rows)

def export_summary_png(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4*len(df)))
    ax.axis('off')
    shown = df.copy()
    def pct(x): return f"{x:.2%}"
    def money(x): return f"¥{x:,.0f}"
    fmt = {"コントロールCVR": pct, "テストCVR": pct, "差": pct, "差_下限": pct, "差_上限": pct,
           "リフト率": lambda x: "" if pd.isna(x) else f"{x:.1%}",
           "追加CV": lambda x: f"{x:,.0f}", "追加CV_下限": lambda x: f"{x:,.0f}", "追加CV_上限": lambda x: f"{x:,.0f}",
           "追加売上": money, "追加売上_下限": money, "追加売上_上限": money, "p値": lambda x: f"{x:.3g}"}
    for c, f in fmt.items():
        if c in shown.columns: shown[c] = shown[c].apply(f)
    table = ax.table(cellText=shown.values, colLabels=shown.columns, loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=200, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.read()

# =====================
# UI
# =====================
st.title("A/B/Cテスト効果検証ツール（自社用）")
st.caption("誰でも使えるGo/Hold/No-Go判定と月次インパクト。統計の詳細は折り畳みで確認できます。")

# ---- Sidebar (as requested) ----
with st.sidebar:
    with st.expander("検定設定（通常は開かなくてOK）", expanded=False):
        alt_label = st.selectbox("検定方法", ["両側検定", "片側検定（テスト>A）"])
        alpha = st.number_input("有意水準 α", value=0.05, min_value=0.001, max_value=0.2, step=0.01, format="%.3f")
        mde_pt = st.number_input("MDE（最小検出差, pt）", value=0.10, min_value=0.0, max_value=50.0, step=0.05, format="%.2f")
    alternative = "two-sided" if alt_label == "両側検定" else "larger"

    st.header("データ入力")
    st.caption("A=コントロール群、B/C=テスト群。分母はセッション数です。")
    a_sess = st.number_input("A：コントロール群 セッション数", value=31465, min_value=1, step=1)
    a_cv   = st.number_input("A：コントロール群 CV数", value=1003, min_value=0, step=1)
    b_sess = st.number_input("B：テスト群（かえる層）セッション数", value=11773, min_value=1, step=1)
    b_cv   = st.number_input("B：テスト群（かえる層）CV数", value=313, min_value=0, step=1)
    use_c  = st.checkbox("C群（追加テスト）を入力する", value=False)
    c_sess = st.number_input("C：テスト群 セッション数", value=10000 if use_c else 0, min_value=0, step=1, disabled=not use_c)
    c_cv   = st.number_input("C：テスト群 CV数", value=250 if use_c else 0, min_value=0, step=1, disabled=not use_c)

    st.header("ビジネス前提")
    monthly_sessions = st.number_input("月間セッション数", value=int(a_sess + b_sess + (c_sess if use_c else 0)), step=100)
    unit_value = st.number_input("1件あたりの売上（円）", value=20000, step=1000)
    st.caption("ECなら平均注文額、B2BならLTV×成約率などに置換可。")

    analyze = st.button("分析開始！")

if not analyze:
    st.info("左のサイドバーに数値を入力して『分析開始！』を押してください。")
    st.stop()

# ---- compute ----
ctrl = Group("A：コントロール群", int(a_sess), int(a_cv))
results: List[TestResult] = []

b_group = Group("B：テスト群（かえる層）", int(b_sess), int(b_cv))
b_res = judge(ctrl, b_group, alpha, mde_pt, alternative, int(monthly_sessions), float(unit_value))
results.append(b_res)

if use_c and c_sess > 0:
    c_group = Group("C：テスト群", int(c_sess), int(c_cv))
    c_res = judge(ctrl, c_group, alpha, mde_pt, alternative, int(monthly_sessions), float(unit_value))
    results.append(c_res)

# ---- Render in order: B -> C -> Best ----
def render_one(r: TestResult):
    title = f"{r.group_name} の判定"
    if r.decision == "go":
        result_box(title, "go", "おめでとうございます！！今回の施策は効果が出た可能性が高く、再現性が期待できる結果です！単なる偶然ではなく、狙った効果が現れたと考えられます。")
    elif r.decision == "hold":
        result_box(title, "hold", "データ量が少なく、今回の結果では効果があったと断言できません。")
    else:
        result_box(title, "nogo", "残念ですが、今回の結果では効果があったと断言できません。効果がゼロとは限りませんが、再現性は不明です。")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("コントロール CVR", fmt_pct(r.p_ctrl))
    with c2: st.metric("テスト CVR", fmt_pct(r.p_test))
    with c3:
        st.metric("差（A→テスト）", fmt_pt(r.diff))
        st.caption(f"差の95%CI: {fmt_pt(r.diff_ci[0])} – {fmt_pt(r.diff_ci[1])}")
    with c4:
        if r.lift is not None: st.metric("リフト率", f"{r.lift*100:.1f}%")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric("追加CV数（推定）", f"{r.add_cv:,.0f} 件")
        st.caption(f"区間: {r.add_cv_low:,.0f} – {r.add_cv_high:,.0f}")
    with cc2: st.metric("1件あたりの売上（円）", yen(unit_value))
    with cc3:
        st.metric("追加売上（推定）", yen(r.add_rev))
        st.caption(f"区間: {yen(r.add_rev_low)} – {yen(r.add_rev_high)}")

    with st.expander("詳細（統計の根拠）", expanded=False):
        st.markdown(f"- p値 = **{r.p_value:.6g}**、基準α = **{alpha}**")
        st.markdown(f"- 差 = **{fmt_pt(r.diff)}**、MDE = **{mde_pt:.2f} pt**")
        st.markdown(f"- 差の95%CI: **{fmt_pt(r.diff_ci[0])} – {fmt_pt(r.diff_ci[1])}**")
        st.caption("※ p値は『偶然この差が出る確率』、MDEは『実務的に意味がある最小の差』。本カードはZ検定と信頼区間にもとづく判定です。")

render_one(b_res)
if use_c and c_sess > 0:
    render_one(c_res)

# pick best
best = None
go_list = [r for r in results if r.decision == "go"]
if go_list: best = max(go_list, key=lambda r: r.add_rev)
else: best = max(results, key=lambda r: r.diff)
st.info("🏆 ベスト（最大インパクト）")
render_one(best)

# ---- Summary + export ----
st.divider()
st.subheader("結果サマリー")
df = build_summary_df(results)
st.dataframe(df.style.format({
    "コントロールCVR":"{:.4%}","テストCVR":"{:.4%}","差":"{:.4%}","差_下限":"{:.4%}","差_上限":"{:.4%}",
    "リフト率":"{:.1%}","p値":"{:.3g}",
    "追加CV":"{:.0f}","追加CV_下限":"{:.0f}","追加CV_上限":"{:.0f}",
    "追加売上":"{:,.0f}","追加売上_下限":"{:,.0f}","追加売上_上限":"{:,.0f}"
}))
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("結果をCSVでダウンロード", data=csv, file_name="ab_test_summary.csv", mime="text/csv")

png_bytes = export_summary_png(df)
st.download_button("結果サマリーを画像(PNG)でダウンロード", data=png_bytes, file_name="ab_test_summary.png", mime="image/png")

with st.expander("用語ヘルプ（i）", expanded=False):
    st.markdown(
        """- **p値**: 「偶然でこの差が出る確率」。小さいほど偶然ではない。
- **α（有意水準）**: p値と比べる基準。一般に0.05。
- **MDE**: 実務的に意味がある最小の差（しきい値）。
- **95%CI**: 真の値が入る範囲（推定の誤差幅）。
- **リフト率**: 相対改善率 = (CVR_test - CVR_ctrl) / CVR_ctrl。
- **追加CV/売上**: CVR差を月間セッション数と金額に外挿した推定（区間は差のCIに基づく）。"""
    )
