# %% ミスプライシング戦略（規模 2 × B/M 3 × 収益性 3 = 18 ポートフォリオ）
# 戦略 = 1/2 × (大型・高B/M・高収益 − 大型・低B/M・低収益) + 1/2 × (小型・高B/M・高収益 − 小型・低B/M・低収益)
#
# 前提：ノートブック上に df_bs, df_pl, df_mkt_cap, df_rtn_features があること（他のコードと同じ形式）
# このセルだけで、データの作成 → ポートフォリオ → パフォーマンスの表と図 まで出力する
#
# 先読みバイアスを防ぐためのタイミング（t 年 6 月末にポートフォリオを組む場合）
#   - 会計データ：決算期末が「t−1 年 4 月 〜 t 年 3 月」の本決算（12か月決算）のうち最新のもの
#                 （t 年 3 月期までに終わった本決算。3 月決算の短信は通常 5 月中に公表済み）
#   - 規模・ウェイト：t 年 6 月末の時価総額
#   - B/M：自己資本 ÷ 時価総額（既定は決算期末の時価総額。BM_ME='june' で 6 月末）
#   - リターン：t 年 7 月 〜 t+1 年 6 月（ウェイトは 6 月末の時価総額で 1 年間固定）

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ===================== 設定 =====================
PROF_VAR = 'CBOP'        # 収益性指標：'CBOP'（営業利益+R&D−Δアクルーアル）/前期総資産 ／ 'CP'（営業利益−支払利息−Δアクルーアル）/前期自己資本
BIG_CUM_SHARE = 0.90     # 時価総額の大きい順に累積して、この割合に達するまでを「大型」
BP_UNIVERSE = 'all'      # B/M・収益性の 30/70% 分位点を計算する銘柄：'all'（全銘柄）／ 'big'（大型株のみ。FF の国際版に近い）
BM_ME = 'fye'            # B/M の分母：'fye'（決算期末の時価総額）／ 'june'（t 年 6 月末の時価総額）
EXCLUDE_IND = globals().get('EXCLUDE_IND', [])   # 除外する業種（金融・公益）。ノートブックに定義済みならそれを使う
NW_LAGS = 12             # Newey-West のラグ
ALPHA_LEVEL = 0.05       # 有意水準


def _show(title, obj=None, note=None):
    print('\n' + '=' * 80)
    print('■ ' + title)
    if note:
        for line in note.strip().split('\n'):
            print('  ' + line.strip())
    print('=' * 80)
    if obj is not None:
        display(obj)


# ===================== 1. 会計データ（企業 × 決算期） =====================
KEY = ['stock_id', 'accounting_type', 'fiscal_year_end_yyyymm']
acc = df_bs.merge(
    df_pl[KEY + ['industry', 'operating_income', 'interest_expense', 'research_and_development_expense']],
    on=KEY, how='inner')
acc = acc[acc['fiscal_period_months'] == 12].copy()          # 本決算の12か月決算のみ

acc['nwc'] = (acc['receivables'] + acc['inventories'] + acc['prepaid_expenses']
              - acc['deferred_revenue_current'] - acc['deferred_revenue_noncurrent']
              - acc['accounts_payable'] - acc['accrued_expenses_proxy'])

lag = acc[KEY + ['total_assets', 'book_equity', 'nwc']].copy()
lag['fiscal_year_end_yyyymm'] += 100                         # ちょうど1年前の決算期（同一会計基準）
acc = acc.merge(lag.rename(columns={'total_assets': 'ta_lag', 'book_equity': 'be_lag', 'nwc': 'nwc_lag'}),
                on=KEY, how='left')
d_accr = acc['nwc'] - acc['nwc_lag']
ta0 = acc['ta_lag'].where(acc['ta_lag'] > 0)
be0 = acc['be_lag'].where(acc['be_lag'] > 0)
acc['CBOP'] = (acc['operating_income'] + acc['research_and_development_expense'] - d_accr) / ta0
acc['CP'] = (acc['operating_income'] - acc['interest_expense'] - d_accr) / be0

# 会計基準の変更年で2行ある場合：収益性が計算できる行 → 新しい基準の行 を優先
last_std = acc.sort_values('fiscal_year_end_yyyymm').groupby('stock_id')['accounting_type'].last()
acc['_ok'] = acc[PROF_VAR].notna().astype(int)
acc['_new'] = acc['accounting_type'].eq(acc['stock_id'].map(last_std)).astype(int)
acc = (acc.sort_values(['stock_id', 'fiscal_year_end_yyyymm', '_ok', '_new'], ascending=[True, True, False, False])
          .drop_duplicates(['stock_id', 'fiscal_year_end_yyyymm']))

# ===================== 2. 形成年（t 年 6 月）への割り当て =====================
fy = acc['fiscal_year_end_yyyymm'] // 100
fm = acc['fiscal_year_end_yyyymm'] % 100
acc['form_year'] = np.where(fm <= 3, fy, fy + 1)            # 決算期末が t−1年4月〜t年3月 → t 年 6 月
acc = (acc.dropna(subset=['security_code'])
          .sort_values('fiscal_year_end_yyyymm')
          .drop_duplicates(['security_code', 'form_year'], keep='last'))   # 同じ形成年なら最新の決算期
acc = acc[acc['industry'].notna() & ~acc['industry'].isin(list(EXCLUDE_IND))]

# 先読みの確認：使う決算期末は必ず t 年 3 月以前
assert (acc['fiscal_year_end_yyyymm'] <= acc['form_year'] * 100 + 3).all(), '先読み：t年3月より後の決算期を使っています'

# ===================== 3. 時価総額・B/M =====================
mcap = (df_mkt_cap.rename(columns={'ticker': 'security_code'})
                  .drop_duplicates(['security_code', 'date']))
me_jun = (mcap[mcap['date'] % 100 == 6]
          .assign(form_year=lambda d: d['date'] // 100)
          .rename(columns={'market_cap': 'ME_jun'})[['security_code', 'form_year', 'ME_jun']])
me_fye = mcap.rename(columns={'date': 'fiscal_year_end_yyyymm', 'market_cap': 'ME_fye'})[
    ['security_code', 'fiscal_year_end_yyyymm', 'ME_fye']]

pf = acc.merge(me_jun, on=['security_code', 'form_year'], how='inner')        # t 年 6 月末の時価総額
pf = pf.merge(me_fye, on=['security_code', 'fiscal_year_end_yyyymm'], how='left')
me_bm = pf['ME_fye'] if BM_ME == 'fye' else pf['ME_jun']
pf['BM'] = pf['book_equity'].where(pf['book_equity'] > 0) / me_bm.where(me_bm > 0)
pf = pf[(pf['ME_jun'] > 0)].dropna(subset=['BM', PROF_VAR]).copy()

# ===================== 4. 規模（累積 90%）・B/M・収益性の分類 =====================
def _size_group(g):
    g = g.sort_values('ME_jun', ascending=False)
    share = g['ME_jun'] / g['ME_jun'].sum()
    big = (share.cumsum() - share) < BIG_CUM_SHARE           # その銘柄を加える前の累積が 90% 未満なら大型
    return pd.Series(np.where(big, 'B', 'S'), index=g.index)

pf['size'] = pf.groupby('form_year', group_keys=False).apply(_size_group)


def _tercile(pf, var):
    out = pd.Series(index=pf.index, dtype=object)
    for t, g in pf.groupby('form_year'):
        ref = g if BP_UNIVERSE == 'all' else g[g['size'] == 'B']
        q30, q70 = ref[var].quantile([0.3, 0.7])
        out.loc[g.index] = np.where(g[var] <= q30, 'L', np.where(g[var] > q70, 'H', 'M'))
    return out

pf['bm_g'] = _tercile(pf, 'BM')
pf['pr_g'] = _tercile(pf, PROF_VAR)
pf['port'] = pf['size'] + '/' + pf['bm_g'] + '/' + pf['pr_g']   # 例：'B/H/H' = 大型・高B/M・高収益

# ===================== 5. 月次リターン（t 年 7 月〜 t+1 年 6 月） =====================
rt = df_rtn_features.rename(columns={'TICKER': 'security_code'})[['security_code', 'month', 'ret']].copy()
rt['form_year'] = np.where(rt['month'].dt.month >= 7, rt['month'].dt.year, rt['month'].dt.year - 1)
pan = rt.merge(pf[['security_code', 'form_year', 'port', 'ME_jun']], on=['security_code', 'form_year'], how='inner')
pan = pan.dropna(subset=['ret'])
_mnum = pan['month'].dt.year * 12 + pan['month'].dt.month
assert (_mnum >= pan['form_year'] * 12 + 7).all() and (_mnum <= (pan['form_year'] + 1) * 12 + 6).all(), \
    '先読み：保有期間（t年7月〜t+1年6月）の外のリターンが含まれています'

pan['rw'] = pan['ret'] * pan['ME_jun']
g = pan.groupby(['month', 'port'])[['rw', 'ME_jun']].sum()
R = (g['rw'] / g['ME_jun']).unstack('port')                    # 18 ポートフォリオの時価総額加重リターン
NUM = pan.groupby(['month', 'port']).size().unstack('port')    # 各月の銘柄数

LONG_B, SHORT_B, LONG_S, SHORT_S = 'B/H/H', 'B/L/L', 'S/H/H', 'S/L/L'
S = pd.DataFrame({
    'Big leg（大型 高高−低低）': R[LONG_B] - R[SHORT_B],
    'Small leg（小型 高高−低低）': R[LONG_S] - R[SHORT_S],
})
S['Strategy'] = 0.5 * S.iloc[:, 0] + 0.5 * S.iloc[:, 1]
S = S.dropna()


# ===================== 6. パフォーマンス =====================
def _perf(x):
    x = x.dropna()
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    m, t, p2 = res.params[0], res.tvalues[0], res.pvalues[0]
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2                        # 片側検定（H1：平均 > 0）
    cum = (1 + x).cumprod()
    return {'平均 (%/月)': round(m * 100, 3), 't値 (NW)': round(t, 2), '片側p値 (H1: >0)': round(p1, 4),
            '年率平均 (%)': round(m * 12 * 100, 2), '年率ボラ (%)': round(x.std() * np.sqrt(12) * 100, 2),
            '年率Sharpe': round(m / x.std() * np.sqrt(12), 2),
            '最大ドローダウン (%)': round((cum / cum.cummax() - 1).min() * 100, 1),
            '勝率 (%)': round((x > 0).mean() * 100, 1),
            '期間': f'{x.index.min()}〜{x.index.max()}', '月数': len(x)}

perf = pd.DataFrame({c: _perf(S[c]) for c in S.columns}).T
st = perf.loc['Strategy']
verdict = (f"戦略の月次平均スプレッドは {st['平均 (%/月)']:.3f}%（年率 {st['年率平均 (%)']:.2f}%）、"
           f"Newey-West t値 {st['t値 (NW)']:.2f}、片側p値 {st['片側p値 (H1: >0)']:.4f}（{st['月数']}か月）。"
           f"→ {int(ALPHA_LEVEL * 100)}%水準で、スプレッドは"
           f"{'有意に正である' if st['片側p値 (H1: >0)'] < ALPHA_LEVEL else '有意に正とはいえない'}。")

_show(f'[ミスプライシング戦略] パフォーマンス（収益性 = {PROF_VAR}）', perf,
      note=f'戦略 = ½(大型 高B/M高収益 − 大型 低B/M低収益) + ½(小型 高B/M高収益 − 小型 低B/M低収益)。\n'
           f'大型 = 6月末時価総額の累積 {int(BIG_CUM_SHARE * 100)}% まで。分位点の計算対象：{BP_UNIVERSE}。B/M の分母：{BM_ME}。\n'
           'ポートフォリオ内は 6月末時価総額で加重（1年間固定）。t値は Newey-West、p値は片側（平均 > 0）。\n'
           + verdict)

# 銘柄数の確認（18 ポートフォリオ）
num_tab = NUM.mean().round(1).rename('平均銘柄数').to_frame()
num_tab['最小銘柄数'] = NUM.min()
num_tab = num_tab.reindex(sorted(num_tab.index))
_show('[確認] 18 ポートフォリオの銘柄数（月平均・最小）', num_tab.T,
      note='表記は「規模/B/M/収益性」（B=大型・S=小型、H=高・M=中・L=低）。\n'
           '戦略で使うのは B/H/H, B/L/L, S/H/H, S/L/L。銘柄数が極端に少ない月があると、リターンが不安定になる。')
for p in [LONG_B, SHORT_B, LONG_S, SHORT_S]:
    if NUM[p].min() < 5:
        print(f'  ※ 注意：{p} は銘柄数が5未満の月があります（最小 {int(NUM[p].min())}）')

# 4 つのコーナーポートフォリオの平均リターン
corner = pd.DataFrame({p: {'平均 (%/月)': round(R[p].mean() * 100, 3), '月数': R[p].count()}
                       for p in [LONG_B, SHORT_B, LONG_S, SHORT_S]}).T
_show('[参考] ロング・ショートに使う 4 ポートフォリオの平均リターン', corner)

# 外部ファクターに対するアルファ（ノートブックに French ファクター FF があれば）
if isinstance(globals().get('FF'), pd.DataFrame):
    FFx = globals()['FF']
    rows = {}
    for name, cols in [('FF3', ['MKT', 'SMB', 'HML']), ('FF5', ['MKT', 'SMB', 'HML', 'RMW', 'CMA'])]:
        if not set(cols).issubset(FFx.columns):
            continue
        d = pd.concat([S['Strategy'], FFx[cols]], axis=1).dropna()
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
        res = sm.OLS(d['Strategy'].to_numpy(), X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        rows[name] = {'α (%/月)': round(res.params[0] * 100, 3), 't値': round(res.tvalues[0], 2),
                      'Adj-R2': round(res.rsquared_adj, 2), '期間': f'{d.index.min()}〜{d.index.max()}', '月数': len(d)}
    if rows:
        _show('[参考] French ファクターに対する戦略のアルファ', pd.DataFrame(rows).T,
              note='ノートブック上の FF（French ファクター、小数表記）と共通の月で推定。')

# ===================== 7. 描画 =====================
fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
for c, col in zip(S.columns, ['tab:blue', 'tab:orange', 'black']):
    cum = (1 + S[c]).cumprod()
    axes[0].plot(cum.index.to_timestamp(), cum.to_numpy(), color=col, lw=2 if c == 'Strategy' else 1.2,
                 label=c.split('（')[0])
axes[0].set_yscale('log')
axes[0].axhline(1, color='gray', lw=0.8)
axes[0].set_title(f'Cumulative return ({PROF_VAR}) | {S.index.min()} - {S.index.max()}')
axes[0].legend()

roll = S['Strategy'].rolling(36).mean() * 12 * 100
axes[1].plot(roll.index.to_timestamp(), roll.to_numpy(), color='black')
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set_title('Strategy: 36-month rolling mean (% p.a.)')

ann = ((1 + S['Strategy']).groupby(S.index.year).prod() - 1) * 100
axes[2].bar(ann.index.astype(str), ann.to_numpy(), color=np.where(ann >= 0, 'tab:blue', 'tab:red'))
axes[2].axhline(0, color='gray', lw=0.8)
axes[2].set_title('Strategy: calendar-year return (%)')
axes[2].tick_params(axis='x', rotation=90)

fig.tight_layout()
plt.show()

print('\n' + verdict)
