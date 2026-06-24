"""
训练BP模拟器专用模型 —— 全局统计特征
关键: 仅用训练赛季构建统计, 在未见赛季上评估(无数据泄漏)
"""

import pymysql
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DB = {"host": "localhost", "port": 3306, "user": "root",
      "password": "auba7956", "database": "kpl", "charset": "utf8mb4"}

TRAIN_LEAGUES = ['20250002', '20250003', '20250004', '20260001', '20260002']
TEST_LEAGUES = ['20260003']

FEATURES = ['hero_wr_avg_a', 'hero_wr_avg_b', 'hero_wr_diff',
            'synergy_avg_a', 'synergy_avg_b', 'synergy_diff',
            'counter_mean', 'team_wr_a', 'team_wr_b', 'team_wr_diff']


def load_game_data():
    """加载所有比赛的基础数据"""
    conn = pymysql.connect(**DB)
    # 每局的A/B队 + 英雄列表 + 胜负
    games = pd.read_sql("""
        SELECT g.battle_id, m.league_id, m.team_a_id, m.team_b_id,
               m.start_time, gts_a.is_win AS label
        FROM games g
        JOIN matches m ON g.match_id = m.match_id
        JOIN game_team_stats gts_a ON g.battle_id = gts_a.battle_id AND gts_a.team_id = m.team_a_id
        WHERE g.status = 2 AND m.start_time IS NOT NULL
        ORDER BY m.start_time, g.battle_seq
    """, conn, parse_dates=['start_time'])

    players = pd.read_sql("""
        SELECT gps.battle_id, gps.team_id, gps.hero_id
        FROM game_player_stats gps
        JOIN games g ON gps.battle_id = g.battle_id
        WHERE g.status = 2
    """, conn)
    conn.close()

    print(f"比赛: {len(games)}, 选手记录: {len(players)}")
    return games, players


def compute_features(games_df, players_df, hero_wr, synergy, counter, team_wr):
    """为所有比赛构建特征矩阵"""
    samples = []
    skipped = 0

    for _, game in games_df.iterrows():
        bid = game['battle_id']
        ta, tb = game['team_a_id'], game['team_b_id']
        lid = game['league_id']

        a_rows = players_df[(players_df['battle_id'] == bid) & (players_df['team_id'] == ta)]
        b_rows = players_df[(players_df['battle_id'] == bid) & (players_df['team_id'] == tb)]

        if len(a_rows) < 5 or len(b_rows) < 5:
            skipped += 1
            continue

        ah = list(a_rows['hero_id'].astype(int))
        bh = list(b_rows['hero_id'].astype(int))

        f = {'battle_id': bid, 'league_id': lid, 'start_time': game['start_time'],
             'label': int(game['label'])}

        # 英雄胜率
        a_hwr = [hero_wr.get(h, 0.5) for h in ah]
        b_hwr = [hero_wr.get(h, 0.5) for h in bh]
        f['hero_wr_avg_a'] = np.mean(a_hwr)
        f['hero_wr_avg_b'] = np.mean(b_hwr)
        f['hero_wr_diff'] = f['hero_wr_avg_a'] - f['hero_wr_avg_b']

        # 协同
        sa, sb = [], []
        for i in range(5):
            for j in range(i+1, 5):
                sa.append(synergy.get((min(ah[i], ah[j]), max(ah[i], ah[j])), 0.5))
                sb.append(synergy.get((min(bh[i], bh[j]), max(bh[i], bh[j])), 0.5))
        f['synergy_avg_a'] = np.mean(sa) if sa else 0.5
        f['synergy_avg_b'] = np.mean(sb) if sb else 0.5
        f['synergy_diff'] = f['synergy_avg_a'] - f['synergy_avg_b']

        # 克制
        cs = [counter.get((ha, hb), 0.5) for ha in ah for hb in bh]
        f['counter_mean'] = np.mean(cs) if cs else 0.5

        # 队伍
        f['team_wr_a'] = team_wr.get(ta, 0.5)
        f['team_wr_b'] = team_wr.get(tb, 0.5)
        f['team_wr_diff'] = f['team_wr_a'] - f['team_wr_b']

        samples.append(f)

    print(f"有效样本: {len(samples)}, 跳过: {skipped}")
    return pd.DataFrame(samples)


def build_stats_from_games(games_subset, players_df):
    """从指定比赛子集构建全局统计(英雄/协同/克制/队伍)"""
    bids = set(games_subset['battle_id'])

    # 英雄胜率
    hero_wr = {}
    hero_counts = {}
    for _, game in games_subset.iterrows():
        bid = game['battle_id']
        ta, tb = game['team_a_id'], game['team_b_id']
        a_win = bool(game['label'])

        for _, pl in players_df[players_df['battle_id'] == bid].iterrows():
            h = int(pl['hero_id'])
            win = a_win if pl['team_id'] == ta else (not a_win)
            hero_wr[h] = hero_wr.get(h, 0) + (1 if win else 0)
            hero_counts[h] = hero_counts.get(h, 0) + 1
    hero_wr = {h: hero_wr[h] / hero_counts[h] for h in hero_wr}

    # 协同 (同队英雄对)
    syn_pairs = {}
    for _, game in games_subset.iterrows():
        bid = game['battle_id']
        ta, tb = game['team_a_id'], game['team_b_id']
        a_win = bool(game['label'])

        for team_id, win in [(ta, a_win), (tb, not a_win)]:
            team_heroes = list(players_df[(players_df['battle_id'] == bid) &
                                          (players_df['team_id'] == team_id)]['hero_id'].astype(int))
            for i in range(len(team_heroes)):
                for j in range(i+1, len(team_heroes)):
                    key = (min(team_heroes[i], team_heroes[j]), max(team_heroes[i], team_heroes[j]))
                    if key not in syn_pairs:
                        syn_pairs[key] = {'wins': 0, 'total': 0}
                    syn_pairs[key]['total'] += 1
                    if win:
                        syn_pairs[key]['wins'] += 1

    synergy = {k: v['wins'] / v['total'] for k, v in syn_pairs.items() if v['total'] >= 3}

    # 克制 (A方英雄 vs B方英雄)
    ctr_pairs = {}
    for _, game in games_subset.iterrows():
        bid = game['battle_id']
        ta, tb = game['team_a_id'], game['team_b_id']
        a_win = bool(game['label'])

        a_heroes = list(players_df[(players_df['battle_id'] == bid) &
                                   (players_df['team_id'] == ta)]['hero_id'].astype(int))
        b_heroes = list(players_df[(players_df['battle_id'] == bid) &
                                   (players_df['team_id'] == tb)]['hero_id'].astype(int))
        for ha in a_heroes:
            for hb in b_heroes:
                key = (ha, hb)
                if key not in ctr_pairs:
                    ctr_pairs[key] = {'wins': 0, 'total': 0}
                ctr_pairs[key]['total'] += 1
                if a_win:
                    ctr_pairs[key]['wins'] += 1

    counter = {k: v['wins'] / v['total'] for k, v in ctr_pairs.items() if v['total'] >= 3}

    # 队伍胜率
    team_wr = {}
    for _, game in games_subset.iterrows():
        for tid, win in [(game['team_a_id'], bool(game['label'])),
                         (game['team_b_id'], not bool(game['label']))]:
            if tid not in team_wr:
                team_wr[tid] = {'wins': 0, 'total': 0}
            team_wr[tid]['total'] += 1
            if win:
                team_wr[tid]['wins'] += 1
    team_wr = {k: v['wins'] / v['total'] for k, v in team_wr.items()}

    print(f"  英雄: {len(hero_wr)}, 协同: {len(synergy)}, 克制: {len(counter)}, 队伍: {len(team_wr)}")
    return hero_wr, synergy, counter, team_wr


def main():
    print("=" * 56)
    print("BP模拟器模型训练 (全局统计 · 无泄漏评估)")
    print("=" * 56)

    games, players = load_game_data()

    # 划分训练/测试赛季
    train_games = games[games['league_id'].isin(TRAIN_LEAGUES)]
    test_games = games[games['league_id'].isin(TEST_LEAGUES)]

    # 🚫 仅用训练赛季构建统计
    print(f"\n仅用 {TRAIN_LEAGUES} 构建全局统计...")
    hwr, syn, ctr, twr = build_stats_from_games(train_games, players)

    # 对全部数据计算特征
    print("\n计算特征...")
    full_df = compute_features(games, players, hwr, syn, ctr, twr)
    full_df = full_df.sort_values('start_time')

    train_df = full_df[full_df['league_id'].isin(TRAIN_LEAGUES)]
    test_df = full_df[full_df['league_id'].isin(TEST_LEAGUES)]

    print(f"\n训练: {len(train_df)}局 | 测试: {len(test_df)}局")
    print(f"训练正样本率: {train_df['label'].mean():.3f}")

    X_train = train_df[FEATURES].fillna(0.5)
    y_train = train_df['label']
    X_test = test_df[FEATURES].fillna(0.5)
    y_test = test_df['label']

    # LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {TEST_LEAGUES} 测试结果 (无泄漏) ===")
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {acc:.4f}  AUC: {auc:.4f}")

    print("\n特征重要性:")
    for f, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        bar = '#' * int(imp / 5)
        print(f"  {f:25s} {imp:6.0f} {bar}")

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nBaseline(多数类 = {baseline:.3f})")
    print(f"提升: +{(acc - baseline)*100:.1f}pp")

    # 5折CV
    from sklearn.model_selection import TimeSeriesSplit
    cv = TimeSeriesSplit(3)
    scores = []
    train_sorted = train_df.sort_values('start_time')
    X_cv = train_sorted[FEATURES].fillna(0.5)
    y_cv = train_sorted['label']
    for i, (tr, val) in enumerate(cv.split(X_cv)):
        m = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                               random_state=42, verbose=-1)
        m.fit(X_cv.iloc[tr], y_cv.iloc[tr])
        yp = m.predict(X_cv.iloc[val])
        scores.append(accuracy_score(y_cv.iloc[val], yp))
        print(f"  CV Fold{i+1}: {scores[-1]:.4f}")
    print(f"  CV Avg: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # 生产模型: 全量数据训练(不用测试赛季统计，但用全量统计表)
    print("\n训练生产模型...")
    # 使用MySQL中的全量统计表(hero_stats/hero_synergy/hero_counter/team_stats)
    conn = pymysql.connect(**DB)

    hero_wr_all = pd.read_sql("SELECT hero_id, win_rate FROM hero_stats", conn)
    hwr_all = dict(zip(hero_wr_all['hero_id'], hero_wr_all['win_rate'].fillna(0.5)))

    syn_all = pd.read_sql("SELECT hero_a_id, hero_b_id, win_rate FROM hero_synergy", conn)
    syn_dict = {}
    for _, r in syn_all.iterrows():
        syn_dict[(int(r['hero_a_id']), int(r['hero_b_id']))] = float(r['win_rate']) if r['win_rate'] else 0.5

    ctr_all = pd.read_sql("SELECT hero_a_id, hero_b_id, win_rate_a FROM hero_counter", conn)
    ctr_dict = {}
    for _, r in ctr_all.iterrows():
        ctr_dict[(int(r['hero_a_id']), int(r['hero_b_id']))] = float(r['win_rate_a']) if r['win_rate_a'] else 0.5

    twr_all = pd.read_sql("SELECT team_id, win_rate FROM team_stats", conn)
    twr_dict = dict(zip(twr_all['team_id'], twr_all['win_rate'].fillna(0.5)))
    conn.close()

    # 重建全量特征
    all_df = compute_features(games, players, hwr_all, syn_dict, ctr_dict, twr_dict)
    X_all = all_df[FEATURES].fillna(0.5)

    product = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        num_leaves=31, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )
    product.fit(X_all, all_df['label'])

    product.booster_.save_model('e:/gediandata/kpl/bp_model.txt')
    with open('e:/gediandata/kpl/bp_model_features.json', 'w') as f:
        json.dump(FEATURES, f)

    print(f"生产模型已保存 (全量{len(X_all)}局训练)")
    print("文件: bp_model.txt / bp_model_features.json")


if __name__ == "__main__":
    main()
