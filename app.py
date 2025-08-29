
import io
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
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
    go_type: str  # "go" | "hold" | "nogo"
    reason: str
    lift: Optional[float]
    add_cv: float
    add_cv_low: float
    add_cv_high: float
    add_rev: float
    add_rev_low: float
    add_rev_high: float

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"
def fmt_pt(x: float) -> str:
    return f"{x*100:.2f} pt"
def yen(x: float) -> str:
    return f"¥{x:,.0f}"

def ztest_and_ci(ctrl: Group, test: Group, alpha: float, alternative: str) -> Dict[str, Any]:
    counts = np.array([test.conv, ctrl.conv])
    nobs   = np.array([test.sessions, ctrl.sessions])
    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative=alternative)

    # Wilson CI for proportions
    ci_ctrl = proportion_confint(ctrl.conv, ctrl.sessions, alpha=alpha, method="wilson")
    ci_test = proportion_confint(test.conv, test.sessions, alpha=alpha, method="wilson")

    p1 = ctrl.conv / max(ctrl.sessions, 1)
    p2 = test.conv / max(test.sessions, 1)

    # CI for difference (normal approx using sample SE)
    se = math.sqrt(p1*(1-p1)/max(ctrl.sessions,1) + p2*(1-p2)/max(test.sessions,1))
    z = abs(norm.ppf(alpha/2)) if alternative == "two-sided" else abs(norm.ppf(alpha))
    diff_low = (p2 - p1) - z*se
    diff_high = (p2 - p1) + z*se

    return {
        "p_ctrl": p1, "p_test": p2, "p_value": pval,
        "ci_ctrl": ci_ctrl, "ci_test": ci_test,
        "diff": p2 - p1, "diff_ci": (diff_low, diff_high)
    }

def min_sample_per_arm(p_bar: float, mde_abs: float, alpha: float=0.05, power: float=0.8) -> int:
    # Rule-of-thumb sample size per arm
    if mde_abs <= 0:
        return 0
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
    n_min = min_sample_per_arm(avg_p, mde_abs, alpha=alpha)
    data_low = (ctrl.conv < 100) or (test.conv < 100)
    wide_ci = (abs(dh - dl) > 0.005)  # >0.5pt
    power_low = (ctrl.sessions < n_min) or (test.sessions < n_min)

    if pval < alpha and abs(diff) >= mde_abs:
        gtype = "go"
        reason = "有意差あり & 差がMDE以上"
    elif (pval >= alpha) and (data_low or wide_ci or power_low):
        gtype = "hold"
        reason = "データ量／不確実性の観点で保留"
    else:
        gtype = "nogo"
        reason = "有意差なし（または差がMDE未満）"

    lift = None
    if p_ctrl > 0:
        lift = (p_test - p_ctrl) / p_ctrl

    # Monthly impact
    add_cv = max(0.0, monthly_sessions * diff)
    add_cv_low = max(0.0, monthly_sessions * dl)
    add_cv_high = max(0.0, monthly_sessions * dh)
    add_rev = add_cv * unit_value
    add_rev_low = add_cv_low * unit_value
    add_rev_high = add_cv_high * unit_value

    return TestResult(
        group_name=test.name,
        p_ctrl=p_ctrl, p_test=p_test,
        diff=diff, diff_ci=(dl, dh),
        p_value=pval, go_type=gtype, reason=reason, lift=lift,
        add_cv=add_cv, add_cv_low=add_cv_low, add_cv_high=add_cv_high,
        add_rev=add_rev, add_rev_low=add_rev_low, add_rev_high=add_rev_high
    )

def render_card(t: TestResult, alpha: float, mde_pt: float, headline_override: Optional[str]=None, best: bool=False):
    # choose color & headline
    if t.go_type == "go":
        box = st.success
        headline = "おめでとうございます！！今回の施策は効果が出た可能性が高く、再現性が期待できる結果です！単なる偶然ではなく、狙った効果が現れたと考えられます。"
    elif t.go_type == "hold":
        box = st.warning
        headline = "データ量が少なく、今回の結果では効果があったと断言できません。"
    else:
        box = st.error
        headline = "残念ですが、今回の結果では効果があったと断言できません。効果がゼロとは限りませんが、再現性は不明です。"
    if headline_override:
        headline = headline_override

    title = f"{t.group_name} の判定"
    if best:
        title = "🏆 ベスト（最大インパクト）"

    with box(title):
        st.markdown(f"### {headline}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("コントロール CVR", fmt_pct(t.p_ctrl))
        with c2:
            st.metric(f"{t.group_name} CVR", fmt_pct(t.p_test))
        with c3:
            st.metric("差（A→テスト）", fmt_pt(t.diff))
            st.caption(f"差の95%CI: {fmt_pt(t.diff_ci[0])} – {fmt_pt(t.diff_ci[1])}")
        with c4:
            if t.lift is not None:
                st.metric("リフト率", f"{t.lift*100:.1f}%")

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("追加CV数（推定）", f"{t.add_cv:,.0f} 件")
            st.caption(f"区間: {t.add_cv_low:,.0f} – {t.add_cv_high:,.0f}")
        with cc2:
            st.metric("1件あたりの売上（円）", yen(st.session_state.get('unit_value', 0)))
        with cc3:
            st.metric("追加売上（推定）", yen(t.add_rev))
            st.caption(f"区間: {yen(t.add_rev_low)} – {yen(t.add_rev_high)}")

        with st.expander("詳細（統計の根拠）", expanded=False):
            st.markdown(f"- p値 = **{t.p_value:.6g}**、基準α = **{alpha}**")
            st.markdown(f"- 差 = **{fmt_pt(t.diff)}**、MDE = **{mde_pt:.2f} pt**")
            st.markdown(f"- 差の95%CI: **{fmt_pt(t.diff_ci[0])} – {fmt_pt(t.diff_ci[1])}**")
            st.caption("※ p値は『偶然この差が出る確率』、MDEは『実務的に意味がある最小の差』。本カードはZ検定と信頼区間にもとづく判定です。")

def build_summary_df(results: List[TestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "群": r.group_name,
            "コントロールCVR": r.p_ctrl,
            "テストCVR": r.p_test,
            "差": r.diff,
            "差_下限": r.diff_ci[0],
            "差_上限": r.diff_ci[1],
            "p値": r.p_value,
            "判定": r.go_type,
            "リフト率": r.lift,
            "追加CV": r.add_cv,
            "追加CV_下限": r.add_cv_low,
            "追加CV_上限": r.add_cv_high,
            "追加売上": r.add_rev,
            "追加売上_下限": r.add_rev_low,
            "追加売上_上限": r.add_rev_high,
        })
    df = pd.DataFrame(rows)
    return df

def export_summary_png(df: pd.DataFrame) -> bytes:
    # Simple matplotlib table export
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4*len(df)))
    ax.axis('off')
    shown = df.copy()
    # format some columns
    def pct(x): return f"{x:.2%}"
    def money(x): return f"¥{x:,.0f}"
    fmt = {
        "コントロールCVR": pct, "テストCVR": pct, "差": pct, "差_下限": pct, "差_上限": pct,
        "リフト率": lambda x: "" if pd.isna(x) else f"{x:.1%}",
        "追加CV": lambda x: f"{x:,.0f}", "追加CV_下限": lambda x: f"{x:,.0f}", "追加CV_上限": lambda x: f"{x:,.0f}",
        "追加売上": money, "追加売上_下限": money, "追加売上_上限": money,
        "p値": lambda x: f"{x:.3g}",
    }
    for c, f in fmt.items():
        if c in shown.columns:
            shown[c] = shown[c].apply(f)
    table = ax.table(cellText=shown.values, colLabels=shown.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# =====================
# UI
# =====================
st.title("A/B/Cテスト効果検証ツール（自社用）")
st.caption("誰でも使えるGo/Hold/No-Go判定と月次インパクト。統計の詳細は折り畳みで確認できます。")

with st.expander("検定設定（通常は開かなくてOK）", expanded=False):
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        alternative_label = st.selectbox("検定方法 ℹ️", ["両側検定", "片側検定（テスト>A）"])
        st.caption("ℹ️ 両側=上がる/下がる両方を検出。片側=上がる場合だけ検出。")
    with colB:
        alpha = st.number_input("有意水準 α ℹ️", value=0.05, min_value=0.001, max_value=0.2, step=0.01, format="%.3f")
        st.caption("ℹ️ p値がαより小さければ『偶然ではない』と判定。一般に0.05を使います。")
    with colC:
        mde_pt = st.number_input("MDE（最小検出差, pt） ℹ️", value=0.10, min_value=0.0, max_value=50.0, step=0.05, format="%.2f")
        st.caption("ℹ️ 実装判断に必要な最小の差。これ未満なら実務的に小さいとみなします。")

# ---- Input form (no auto update) ----
with st.form("ab_form"):
    st.subheader("データ入力")
    st.caption("A=コントロール群、B/C=テスト群。分母はセッション数です。")
    c1, c2, c3 = st.columns(3)
    with c1:
        a_sess = st.number_input("A：コントロール群 セッション数", value=31465, min_value=1, step=1)
        a_cv   = st.number_input("A：コントロール群 CV数", value=1003, min_value=0, step=1)
    with c2:
        b_sess = st.number_input("B：テスト群（かえる層）セッション数", value=11773, min_value=1, step=1)
        b_cv   = st.number_input("B：テスト群（かえる層）CV数", value=313, min_value=0, step=1)
    with c3:
        use_c = st.checkbox("C群（追加テスト）を入力する", value=False)
        c_sess = st.number_input("C：テスト群 セッション数", value=0 if not use_c else 10000, min_value=0, step=1, disabled=not use_c)
        c_cv   = st.number_input("C：テスト群 CV数", value=0 if not use_c else 250, min_value=0, step=1, disabled=not use_c)

    st.subheader("ビジネス前提")
    d1, d2 = st.columns(2)
    with d1:
        monthly_sessions = st.number_input("月間セッション数", value=int(a_sess + b_sess + (c_sess if use_c else 0)), step=100)
        st.caption("ℹ️ 月に対象施策が当たるセッションの概算。A+B(+C)の実測か、将来見込み。")
    with d2:
        unit_value = st.number_input("1件あたりの売上（円）", value=20000, step=1000)
        st.caption("ℹ️ ECなら平均注文額、B2BならLTV×成約率などに置換可。")

    st.session_state['unit_value'] = unit_value
    submitted = st.form_submit_button("分析開始！")

if not submitted:
    st.info("右側の項目に数字を入れて『分析開始！』を押すと結果が表示されます。")
    st.stop()

# ---- compute ----
ctrl = Group("A：コントロール群", a_sess, a_cv)
results: List[TestResult] = []

b_res = judge(ctrl, Group("B：テスト群（かえる層）", b_sess, b_cv), alpha, mde_pt, alternative, monthly_sessions, unit_value)
results.append(b_res)

if use_c and c_sess > 0:
    c_res = judge(ctrl, Group("C：テスト群", c_sess, c_cv), alpha, mde_pt, alternative, monthly_sessions, unit_value)
    results.append(c_res)

# ---- Render cards in order: B -> C -> Best ----
render_card(b_res, alpha, mde_pt)
if use_c and c_sess > 0:
    render_card(results[1], alpha, mde_pt)

# pick best: prefer GO with max add_rev; else max diff
best = None
go_list = [r for r in results if r.go_type == "go"]
if go_list:
    best = max(go_list, key=lambda r: r.add_rev)
else:
    best = max(results, key=lambda r: r.diff)

render_card(best, alpha, mde_pt, headline_override="このカードが現時点で最もビジネスインパクトが大きい候補です。", best=True)

# ---- Summary table & export ----
st.divider()
st.subheader("結果サマリー")

df = build_summary_df(results)
st.dataframe(df.style.format({
    "コントロールCVR":"{:.4%}","テストCVR":"{:.4%}","差":"{:.4%}","差_下限":"{:.4%}","差_上限":"{:.4%}",
    "リフト率":"{:.1%}","p値":"{:.3g}",
    "追加CV":"{:.0f}","追加CV_下限":"{:.0f}","追加CV_上限":"{:.0f}",
    "追加売上":"{:,.0f}","追加売上_下限":"{:,.0f}","追加売上_上限":"{:,.0f}"
}))

# CSV export
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("結果をCSVでダウンロード", data=csv, file_name="ab_test_summary.csv", mime="text/csv")

# Image export
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

st.caption("※ 本アプリは比率のZ検定と信頼区間に基づく簡易判定ツールです。セグメント差・季節性・重複ユーザーなど外部要因は別途ご確認ください。")
