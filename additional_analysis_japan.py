# %% [markdown]
# 追加分析：2×3 ソートの規模別分解（効果が中小型株に集中しているか）
# 前提（ノートブック上で定義・実行済み）:
#   - rmw_yhat_japan.py のセル A〜D
#     （関数 assign_bins, vw_ret, ts_reg, align_to_ff, period_info, show, FACTOR_SETS、
#       変数 sigs, rtw, FF, own, MODEL, PROF）
# 各 "# %%" ごとにノートブックのセルへ貼り付けて実行する

# %% 追加分析セルA：関数定義
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIGNALS = {'exp': 'Yhat', 'expprof': 'Yhat_prof', 'act': None}   # None は実績 Y_t


def rmw_legs(sig, rtw, var, bp_mask=None):
    """2×3 ソートの、小型株の群と大型株の群それぞれのスプレッド（高 − 低）を返す"""
    s = sig.dropna(subset=[var, 'ME_jun']).copy()
    s['sz'] = assign_bins(s, 'ME_jun', [0.5], bp_mask)
    s['vb'] = assign_bins(s, var, [0.3, 0.7], bp_mask)
    s = s.dropna(subset=['sz', 'vb'])
    s[['sz', 'vb']] = s[['sz', 'vb']].astype(int)
    p = rtw.merge(s[['security_code', 'form_year', 'sz', 'vb']], on=['security_code', 'form_year'])
    R = vw_ret(p, ['sz', 'vb'])
    return pd.DataFrame({'small': R[(0, 2)] - R[(0, 0)], 'big': R[(1, 2)] - R[(1, 0)]})


def build_legs(sigs, rtw, FF, bp_mask=None):
    """全シグナル・全収益性指標の小型・大型スプレッドを作り、French と共通の月にそろえる"""
    legs = {}
    for y in sigs:
        for lab, var in SIGNALS.items():
            L = rmw_legs(sigs[y], rtw, var or y, bp_mask)
            legs[f'{lab}{y}_small'] = L['small']
            legs[f'{lab}{y}_big'] = L['big']
    return align_to_ff(pd.DataFrame(legs), FF)


def leg_table(legs, FF, y, model='ff5', nw_lags=12):
    """各シグナルについて、小型・大型それぞれの平均と α、および「小型 − 大型」の差の検定"""
    base = FACTOR_SETS[model]
    fmt = lambda e: f"{e.loc['alpha', 'coef'] * 100:.3f} ({e.loc['alpha', 't']:.2f})"
    rows = {}
    for lab in SIGNALS:
        for g in ['small', 'big']:
            x = legs[f'{lab}{y}_{g}']
            m, _, n = ts_reg(x, None, nw_lags)
            a, _, _ = ts_reg(x, FF[base], nw_lags)
            rows[(lab, g)] = {'平均 (%/月)': fmt(m), f'{model.upper()} α (%/月)': fmt(a), '月数': n}
        d = legs[f'{lab}{y}_small'] - legs[f'{lab}{y}_big']
        m, _, n = ts_reg(d, None, nw_lags)
        a, _, _ = ts_reg(d, FF[base], nw_lags)
        rows[(lab, '小型 − 大型')] = {'平均 (%/月)': fmt(m), f'{model.upper()} α (%/月)': fmt(a), '月数': n}
    return pd.DataFrame(rows).T


def plot_legs(legs, ys):
    """期待版と実績版の、小型・大型それぞれの累積リターン"""
    fig, axes = plt.subplots(1, len(ys), figsize=(7 * len(ys), 4.5), squeeze=False)
    for j, y in enumerate(ys):
        for lab, col in [('exp', 'tab:blue'), ('act', 'tab:orange')]:
            for g, ls in [('small', '-'), ('big', '--')]:
                cum = (1 + legs[f'{lab}{y}_{g}']).cumprod()
                axes[0, j].plot(cum.index.to_timestamp(), cum.to_numpy(), color=col, ls=ls, label=f'{lab} {g}')
        axes[0, j].set_yscale('log')
        axes[0, j].axhline(1, color='gray', lw=0.8)
        axes[0, j].set_title(f'{y}: high-minus-low within size group\n({legs.index.min()} - {legs.index.max()})')
        axes[0, j].legend()
    fig.tight_layout()
    plt.show()


# %% 追加分析セルB：実行
legs = build_legs(sigs, rtw, FF)

for y in PROF:
    # 確認：RMW = ½ × 小型 + ½ × 大型 になっているか（0 に近ければ OK）
    chk = (0.5 * (legs[f'exp{y}_small'] + legs[f'exp{y}_big'])
           - own[f'RMW_exp{y}'].reindex(legs.index)).abs().max()
    show(f'[追加分析：規模別の分解] {y}（{MODEL.upper()}）', leg_table(legs, FF, y, MODEL),
         note=f'期間：{period_info(legs)}。括弧内は Newey-West t値。\n'
              'small：小型株の群の中での「収益性が高い − 低い」、big：大型株の群の中での同じスプレッド。\n'
              '「小型 − 大型」の α が有意なら、効果は小型株に偏っている。big の α が有意なら、大型株でも効果がある。\n'
              '小型・大型は各年の全銘柄の時価総額中央値で分割（大型株の群には中型株も含まれる）。\n'
              f'（確認：RMW_exp{y} と ½(small + big) の最大の差 = {chk:.2e}）')

plot_legs(legs, list(PROF))


# %% [markdown]
# 追加分析2：なぜ CBOP か（CP が弱い理由の実証）
# 論理の流れ:
#   1. 事前の理由：CBOP は分子と分母の対応がとれ、レバレッジや自己資本の変動の影響を受けにくい（Ball et al.）
#   2. ノイズ：CP は外れ値が多く、予測式の係数が年によって大きくぶれる       → 追加分析セルC
#   3. 予測の正確さ：CBOP の Ŷ は翌年の実際の収益性の順位をよりよく当てる   → 追加分析セルC
#   4. リターンとの関係：予測が正確な指標ほど、Ŷ とリターンの関係が強く出る → 追加分析セルD
#   5. 原因の特定：CP の分母だけを総資産に替えた CP_ta が CBOP に近ければ、原因は分母
# 前提: table7_japan.py セルA（add_expected, build_sample7, fm_reg, CTRL）、ch, rt を実行・作成済み

# %% 追加分析セルC：関数定義と、ノイズ・安定性・予測の正確さの比較
def add_cp_ta(ch):
    """CP の分子（営業利益 − 支払利息 − Δアクルーアル）を前期総資産で割った CP_ta と、その変化 dCP_ta を追加"""
    ch2 = ch.copy()
    d_accr = ch2['nwc'] - ch2['nwc_lag1']
    ta0 = ch2['total_assets_lag1'].where(ch2['total_assets_lag1'] > 0)
    ch2['CP_ta'] = (ch2['operating_income'] - ch2['interest_expense'] - d_accr) / ta0
    lag = ch2[['stock_id', 'fiscal_year_end_yyyymm', 'CP_ta']].copy()
    lag['fiscal_year_end_yyyymm'] += 100                     # ちょうど1年前の決算期の値を結合
    ch2 = ch2.merge(lag.rename(columns={'CP_ta': 'CP_ta_lag1'}),
                    on=['stock_id', 'fiscal_year_end_yyyymm'], how='left')
    ch2['dCP_ta'] = ch2['CP_ta'] - ch2['CP_ta_lag1']
    return ch2


def noise_profile(ch, cols):
    """年ごとに（中央値・四分位範囲で）標準化し、極端な値の割合と尖度を比べる"""
    rows = {}
    for c in cols:
        x = ch[['form_year', c]].dropna()
        g = x.groupby('form_year')[c]
        z = (x[c] - g.transform('median')) / (g.transform(lambda s: s.quantile(.75) - s.quantile(.25)))
        rows[c] = {'尖度（年平均）': g.apply(lambda s: s.kurt()).mean(),
                   '|z|>5 の割合 (%)': (z.abs() > 5).mean() * 100,
                   '観測数': len(x)}
    return pd.DataFrame(rows).T.round(2)


def coef_stability(B):
    """予測式の係数の年ごとのばらつき（変動係数 = 標準偏差 ÷ |平均|）"""
    cols = [c for c in B.columns if c not in ('adj_r2', 'n', 'const')]
    out = pd.DataFrame({'平均': B[cols].mean(), '標準偏差': B[cols].std(),
                        '変動係数': B[cols].std() / B[cols].mean().abs()})
    out.loc['（参考）Adj-R2', '平均'] = B['adj_r2'].mean()
    return out.round(3)


def forecast_accuracy(ch, y, inv, **kw):
    """
    Ŷ（期待）と Y_t（今の値をそのまま使う予測）が、翌年の実際の収益性の順位をどれだけ当てるか
    返り値: 要約、年ごとの順位相関、1段目の係数
    """
    yhat, B = add_expected(ch, y, inv, **kw)
    nxt = ch[['security_code', 'form_year', y]].rename(columns={y: 'y_next'})
    nxt['form_year'] -= 1                                   # 翌年の行の Y を「1年後の実際の値」とする
    d = (yhat.merge(nxt, on=['security_code', 'form_year'])
             .merge(ch[['security_code', 'form_year', y]], on=['security_code', 'form_year'])
             .dropna())
    ic = d.groupby('form_year').apply(lambda g: pd.Series({
        'Ŷ の順位相関': g['Yhat'].corr(g['y_next'], method='spearman'),
        'Y_t の順位相関': g[y].corr(g['y_next'], method='spearman')}))
    summary = pd.Series({'Ŷ の順位相関（年平均）': ic['Ŷ の順位相関'].mean(),
                         'Y_t の順位相関（年平均）': ic['Y_t の順位相関'].mean(),
                         'Ŷ が Y_t を上回った年の割合 (%)': (ic['Ŷ の順位相関'] > ic['Y_t の順位相関']).mean() * 100,
                         '年数': len(ic)})
    return summary.round(3), ic, B


ch2 = add_cp_ta(ch)
MEASURES = [('CBOP', 'dA', {}),
            ('CP', 'dB', {}),
            ('CP', 'dB', {'winsor_dep': True}),
            ('CP_ta', 'dA', {})]


def _label(y, kw):
    return y + (' (winsor_dep)' if kw.get('winsor_dep') else '')


show('[追加分析2：ノイズ] 分布の裾の重さ', noise_profile(ch2, ['CBOP', 'CP', 'CP_ta']),
     note='年ごとに中央値と四分位範囲で標準化。尖度が大きく、|z|>5 の割合が高いほど外れ値が多い。\n'
          'CP_ta：CP の分子（営業利益 − 支払利息 − Δアクルーアル）を前期総資産で割った指標。')

acc_rows, ic_all = {}, {}
for y, inv, kw in MEASURES:
    summary, ic, B = forecast_accuracy(ch2, y, inv, **kw)
    acc_rows[_label(y, kw)] = summary
    ic_all[_label(y, kw)] = ic['Ŷ の順位相関']
    show(f'[追加分析2：安定性] {_label(y, kw)}：予測式の係数の年ごとのばらつき', coef_stability(B),
         note='変動係数が大きいほど、係数が年によって大きくぶれる（予測式が不安定）。')

show('[追加分析2：予測の正確さ] 翌年の実際の収益性との順位相関', pd.DataFrame(acc_rows).T,
     note='Ŷ の順位相関が高いほど、将来の収益性の順位をよく当てている（尺度や外れ値に左右されない指標）。\n'
          'Y_t の順位相関は、今の値をそのまま予測に使った場合。\n'
          'CP_ta が CBOP に近ければ、CP が弱い原因は分母（自己資本）にある。CP に近ければ、原因は分子。')

ic_all = pd.DataFrame(ic_all)
fig, ax = plt.subplots(figsize=(10, 4))
for c in ic_all.columns:
    ax.plot(ic_all.index, ic_all[c], marker='o', ms=3, label=c)
ax.set_title('Rank correlation between Yhat and next-year realized profitability')
ax.set_xlabel('formation year')
ax.legend()
fig.tight_layout()
plt.show()


# %% 追加分析セルD：予測の正確さとリターンとの関係（Table 7 の列(1)・列(3) を各指標で比較）
def table7_compact(ch, rt, y, inv, nw_lags=12, **kw):
    """列(1)：Y_t のみ、列(3)：Ŷ のみ（コントロールは Table 7 と同じ）の係数×100 と t値"""
    yhat, _ = add_expected(ch, y, inv, **kw)
    s = build_sample7(ch, rt, y, inv, yhat)
    out = {}
    for col, x in [('(1) Y_t', y), ('(3) Ŷ', 'Yhat')]:
        res, C = fm_reg(s, [x] + CTRL + [inv], nw_lags=nw_lags)
        m, t = res[x]
        out[col] = f'{m * 100:.2f} ({abs(t):.2f})'
    out['Avg. N'] = f"{C['n'].mean():,.0f}"
    out['Months'] = len(C)
    return pd.Series(out)


rows = {_label(y, kw): table7_compact(ch2, rt, y, inv, **kw) for y, inv, kw in MEASURES}
tab = pd.DataFrame(rows).T
tab['Ŷ の順位相関（年平均）'] = pd.DataFrame(acc_rows).T['Ŷ の順位相関（年平均）']
show('[追加分析2：リターンとの関係] 予測の正確さと、Ŷ のリターン予測力', tab,
     note='Fama-MacBeth（全銘柄・等ウェイト）。係数×100、括弧内は Newey-West t値（絶対値）。\n'
          '予測の正確さ（順位相関）が高い指標ほど、列(3)の Ŷ の t値が高ければ、\n'
          '「将来をより正確に予測できる指標ほど、期待収益性とリターンの関係を検出しやすい」ことを支持する。')
