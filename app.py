import math
from dataclasses import dataclass
import numpy as np
import streamlit as st
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy.stats import norm

st.set_page_config(page_title='A/Bテスト効果検証ツール', layout='wide')

@dataclass
class ABInput:
    sessions_a: int
    conversions_a: int
    sessions_b: int
    conversions_b: int

def ztest_and_ci(a: ABInput, alpha: float = 0.05, alternative: str = 'two-sided'):
    counts = np.array([a.conversions_a, a.conversions_b])
    nobs = np.array([a.sessions_a, a.sessions_b])
    stat, pval = proportions_ztest(count=counts, nobs=nobs, alternative=alternative)
    ci_a_low, ci_a_high = proportion_confint(a.conversions_a, a.sessions_a, alpha=alpha, method='wilson')
    ci_b_low, ci_b_high = proportion_confint(a.conversions_b, a.sessions_b, alpha=alpha, method='wilson')
    p1 = a.conversions_a / a.sessions_a
    p2 = a.conversions_b / a.sessions_b
    se = math.sqrt(p1*(1-p1)/a.sessions_a + p2*(1-p2)/a.sessions_b)
    z = abs(norm.ppf(alpha/2)) if alternative == 'two-sided' else abs(norm.ppf(alpha))
    diff_low = (p1 - p2) - z*se
    diff_high = (p1 - p2) + z*se
    return {
        'stat': stat,
        'pval': pval,
        'p1': p1,
        'p2': p2,
        'ci_p1': (ci_a_low, ci_a_high),
        'ci_p2': (ci_b_low, ci_b_high),
        'diff': p1 - p2,
        'diff_ci': (diff_low, diff_high),
    }

def fmt_pct(x):
    return f'{x*100:.2f}%'

def yen(x: float):
    return f'¥{x:,.0f}'

# Sidebar
st.sidebar.header('検定設定')
alternative_label = st.sidebar.selectbox('検定方法', ['両側検定', '片側検定（A>B）'])
alternative = 'two-sided' if alternative_label == '両側検定' else 'larger'
alpha = st.sidebar.number_input('有意水準 α', value=0.05, min_value=0.001, max_value=0.2, step=0.01, format='%.3f')
mde_pct = st.sidebar.number_input('実務的最小差（MDE, pt）', value=0.10, min_value=0.0, max_value=50.0, step=0.05, format='%.2f')

st.sidebar.header('データ入力')
sessions_a = st.sidebar.number_input('A群：訪問数', value=31465, step=1, min_value=1)
conv_a = st.sidebar.number_input('A群：CV数', value=1003, step=1, min_value=0)
sessions_b = st.sidebar.number_input('B群：訪問数', value=11773, step=1, min_value=1)
conv_b = st.sidebar.number_input('B群：CV数', value=313, step=1, min_value=0)

st.sidebar.header('ビジネス前提（任意）')
monthly_traffic = st.sidebar.number_input('月間トラフィック（人）', value=42000, step=100)
conversion_value = st.sidebar.number_input('コンバージョン価値（円/件）', value=20000, step=1000)
st.sidebar.caption('※ 注意：1CV=売上2万円などの前提は結果を大きく左右します。実売上ベースで設定してください。')

# Main
st.title('A/Bテスト効果検証ツール（自社用）')
st.write('**ロジック**：Z検定で有意差判定 → 効果（uplift）を月間トラフィックとCV価値に外挿して事業インパクトを推計。')

inp = ABInput(sessions_a, conv_a, sessions_b, conv_b)
res = ztest_and_ci(inp, alpha=alpha, alternative=alternative)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('A群 CVR', fmt_pct(res['p1']))
    st.caption(f"95%CI: {fmt_pct(res['ci_p1'][0])} – {fmt_pct(res['ci_p1'][1])}")
with col2:
    st.metric('B群 CVR', fmt_pct(res['p2']))
    st.caption(f"95%CI: {fmt_pct(res['ci_p2'][0])} – {fmt_pct(res['ci_p2'][1])}")
with col3:
    uplift_pt = (res['p1'] - res['p2']) * 100
    st.metric('差（A-B）', f'{uplift_pt:.2f} pt')
    st.caption(f"差のCI: {res['diff_ci'][0]*100:.2f} – {res['diff_ci'][1]*100:.2f} pt")

st.divider()
go = (res['pval'] < alpha) and (abs(uplift_pt) >= mde_pct)
hold = (res['pval'] < alpha) and (abs(uplift_pt) < mde_pct)
st.subheader('判定')
if go:
    st.success(f"Go：有意差あり（p={res['pval']:.3g} < α={alpha}）。差 {uplift_pt:.2f}pt はMDE≧{mde_pct:.2f}pt を満たすため、実装価値あり。")
elif hold:
    st.warning(f"Hold：有意差はある（p={res['pval']:.3g}）が、差 {uplift_pt:.2f}pt はMDE {mde_pct:.2f}pt を満たさず。追加検証を推奨。")
else:
    st.error(f"No-Go：有意差なし（p={res['pval']:.3g} ≥ α={alpha}）。")

st.divider()
st.subheader('月次インパクト推定（外挿）')
additional_conversions = monthly_traffic * res['diff']
additional_revenue = additional_conversions * conversion_value
add_conv_low = monthly_traffic * res['diff_ci'][0]
add_conv_high = monthly_traffic * res['diff_ci'][1]
add_rev_low = add_conv_low * conversion_value
add_rev_high = add_conv_high * conversion_value

c1, c2, c3 = st.columns(3)
with c1:
    st.metric('追加CV数（推定）', f'{additional_conversions:,.0f} 件')
    st.caption(f'区間: {add_conv_low:,.0f} – {add_conv_high:,.0f}')
with c2:
    st.metric('追加売上（推定）', yen(additional_revenue))
    st.caption(f'区間: {yen(add_rev_low)} – {yen(add_rev_high)}')
with c3:
    uplift_rel = (res['p1']/max(res['p2'], 1e-9) - 1) * 100
    st.metric('相対改善率', f'{uplift_rel:.1f}%')

st.info('⚠️ 注意：「コンバージョン価値（円/件）」は事業に合わせて設定してください。ECなら平均注文金額、リード獲得ならその後の平均LTV×成約率など。')

st.divider()
st.markdown('''
**計算式**  
- 有意差検定：二群の比率のZ検定（statsmodels）。  
- CVR差の信頼区間：正規近似。  
- 月次インパクト：  
    - ΔCV = (CVR_A − CVR_B) × 月間トラフィック  
    - Δ売上 = ΔCV × コンバージョン価値  
''')
