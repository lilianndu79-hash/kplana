"""
KPL V2 完整预测模型 (V1特征 + 英雄阵容特征)
"""

import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DB = {
    "host": "localhost", "port": 3306, "user": "root",
    "password": "auba7956", "database": "kpl", "charset": "utf8mb4",
}

WINDOW = 60  # 滚动窗口天数


def load_all():
    conn = pymysql.connect(**DB)
    games = pd.read_sql("""
        SELECT g.battle_id, g.match_id, g.battle_seq, g.win_camp, g.game_duration,
               m.league_id, m.start_time, m.team_a_id, m.team_b_id
        FROM games g JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 ORDER BY m.start_time, g.battle_seq
    """, conn, parse_dates=['start_time'])

    team_stats = pd.read_sql("""
        SELECT gts.*, m.start_time
        FROM game_team_stats gts
        JOIN games g ON gts.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 ORDER BY m.start_time
    """, conn, parse_dates=['start_time'])

    player_stats = pd.read_sql("""
        SELECT gps.*, m.start_time
        FROM game_player_stats gps
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 ORDER BY m.start_time
    """, conn, parse_dates=['start_time'])

    conn.close()
    return games, team_stats, player_stats


class FullCache:
    """维护所有维度的滚动统计"""

    def __init__(self, days=60):
        self.days = days
        self.data = {}  # key -> list of dicts with 'time' and metrics

    def _get(self, key):
        return self.data.get(key, [])

    def _prune_add(self, key, record):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(record)
        cutoff = record['time'] - timedelta(days=self.days)
        self.data[key] = [r for r in self.data[key] if r['time'] >= cutoff]

    def _query(self, key, current_time):
        cutoff = current_time - timedelta(days=self.days)
        return [r for r in self._get(key) if r['time'] < current_time and r['time'] >= cutoff]

    def _safe_mean(self, records, field, default):
        if not records:
            return default
        vals = [r[field] for r in records if field in r]
        return float(np.mean(vals)) if vals else default

    def _safe_count(self, records):
        return len(records)

    # --- 更新方法 ---
    def update_team(self, team_id, battle_id, t, kills, deaths, assists, kda,
                    gold, tower, tyrant_cnt, big_dragon, is_win):
        key = f"team_{team_id}"
        self._prune_add(key, {
            'time': t, 'bid': battle_id, 'is_win': is_win,
            'kills': kills, 'deaths': deaths, 'assists': assists,
            'kda': kda, 'gold': gold, 'tower': tower,
            'tyrant': tyrant_cnt, 'big_dragon': big_dragon,
        })

    def update_player(self, player_name, team_id, hero_id, position, t, battle_id,
                      kills, deaths, assists, kda, mvp, gold, hurt_hero, part_rate):
        key = f"player_{player_name}"
        self._prune_add(key, {
            'time': t, 'bid': battle_id, 'team_id': team_id,
            'hero_id': hero_id, 'position': position,
            'kills': kills, 'deaths': deaths, 'assists': assists,
            'kda': kda, 'mvp': mvp, 'gold': gold,
            'hurt_hero': hurt_hero, 'part_rate': part_rate,
        })

    def update_hero(self, hero_id, t, is_win):
        key = f"hero_{hero_id}"
        self._prune_add(key, {'time': t, 'is_win': is_win})

    def update_player_hero(self, player_name, hero_id, t, is_win):
        key = f"ph_{player_name}_{hero_id}"
        self._prune_add(key, {'time': t, 'is_win': is_win})

    def update_synergy(self, ha, hb, t, is_win):
        a, b = (ha, hb) if ha < hb else (hb, ha)
        key = f"syn_{a}_{b}"
        self._prune_add(key, {'time': t, 'is_win': is_win})

    def update_counter(self, ha, hb, t, ha_win):
        key = f"ctr_{ha}_{hb}"
        self._prune_add(key, {'time': t, 'is_win': ha_win})

    # --- 查询方法 ---
    def get_team(self, team_id, t):
        recs = self._query(f"team_{team_id}", t)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
            'avg_kda': self._safe_mean(recs, 'kda', 1.0),
            'avg_gold': self._safe_mean(recs, 'gold', 0),
            'avg_tower': self._safe_mean(recs, 'tower', 0),
            'avg_tyrant': self._safe_mean(recs, 'tyrant', 0),
            'avg_big_dragon': self._safe_mean(recs, 'big_dragon', 0),
        }

    def get_player(self, player_name, t):
        recs = self._query(f"player_{player_name}", t)
        return {
            'avg_kda': self._safe_mean(recs, 'kda', 1.0),
            'avg_mvp': self._safe_mean(recs, 'mvp', 5.0),
            'avg_part_rate': self._safe_mean(recs, 'part_rate', 0.5),
            'avg_gold': self._safe_mean(recs, 'gold', 5000),
            'avg_hurt_hero': self._safe_mean(recs, 'hurt_hero', 30000),
            'games': self._safe_count(recs),
        }

    def get_hero(self, hero_id, t):
        recs = self._query(f"hero_{hero_id}", t)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_player_hero(self, pn, hid, t):
        recs = self._query(f"ph_{pn}_{hid}", t)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_synergy(self, ha, hb, t):
        a, b = (ha, hb) if ha < hb else (hb, ha)
        recs = self._query(f"syn_{a}_{b}", t)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_counter(self, ha, hb, t):
        recs = self._query(f"ctr_{ha}_{hb}", t)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_global_hero_wr(self, hero_id):
        """全历史英雄胜率（冷启动回退）"""
        recs = self._get(f"hero_{hero_id}")
        if not recs:
            return 0.5
        return float(np.mean([r['is_win'] for r in recs]))


def build_features(games_df, team_stats_df, player_stats_df):
    cache = FullCache(WINDOW)
    all_games = games_df.sort_values('start_time').reset_index(drop=True)
    samples = []

    # 预填充缓存（用最早一批比赛做Warmup）
    warmup = all_games.iloc[:len(all_games)//6]
    print(f"  预热: 前 {len(warmup)} 局导入缓存...")
    for _, game in warmup.iterrows():
        bid = game['battle_id']
        t = game['start_time']
        ta = game['team_a_id']
        tb = game['team_b_id']
        pa = player_stats_df[player_stats_df['battle_id'] == bid]

        # 队伍A
        gts_a = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == ta)]
        if len(gts_a) > 0:
            r = gts_a.iloc[0]
            tyrant = int(r['tyrant_kills'] or 0) + int(r['dark_tyrant_kills'] or 0)
            cache.update_team(ta, bid, t, int(r['kills']), int(r['deaths']),
                              int(r['assists']), float(r['kda']), int(r['gold']),
                              int(r['tower_kills']), tyrant, int(r['big_dragon_kills']),
                              bool(r['is_win']))

        # 队伍B
        gts_b = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == tb)]
        if len(gts_b) > 0:
            r = gts_b.iloc[0]
            tyrant = int(r['tyrant_kills'] or 0) + int(r['dark_tyrant_kills'] or 0)
            cache.update_team(tb, bid, t, int(r['kills']), int(r['deaths']),
                              int(r['assists']), float(r['kda']), int(r['gold']),
                              int(r['tower_kills']), tyrant, int(r['big_dragon_kills']),
                              bool(r['is_win']))

        # 选手、英雄、协同
        for _, pl in pa.iterrows():
            is_win = bool(gts_a.iloc[0]['is_win']) if pl['team_id'] == ta and len(gts_a) > 0 else (
                     not bool(gts_b.iloc[0]['is_win']) if pl['team_id'] == tb and len(gts_b) > 0 else False)
            cache.update_player(pl['player_name'], pl['team_id'], pl['hero_id'],
                                pl['position'], t, bid, int(pl['kills']), int(pl['deaths']),
                                int(pl['assists']), float(pl['kda'] if pl['kda'] else pl['kda'] or 1),
                                float(pl['mvp_score']), int(pl['gold']),
                                int(pl['hurt_to_hero']), float(pl['participation_rate']))
            cache.update_hero(pl['hero_id'], t, is_win)
            cache.update_player_hero(pl['player_name'], pl['hero_id'], t, is_win)

        # 队内协同
        a_heroes = [(pl['hero_id'], pl['player_name']) for _, pl in pa.iterrows()
                     if pl['team_id'] == ta]
        b_heroes = [(pl['hero_id'], pl['player_name']) for _, pl in pa.iterrows()
                     if pl['team_id'] == tb]
        for i in range(len(a_heroes)):
            for j in range(i+1, len(a_heroes)):
                cache.update_synergy(a_heroes[i][0], a_heroes[j][0], t,
                                     bool(gts_a.iloc[0]['is_win']) if len(gts_a) > 0 else False)
        for i in range(len(b_heroes)):
            for j in range(i+1, len(b_heroes)):
                cache.update_synergy(b_heroes[i][0], b_heroes[j][0], t,
                                     bool(gts_b.iloc[0]['is_win']) if len(gts_b) > 0 else False)
        # 跨队克制
        for ha, _ in a_heroes:
            for hb, _ in b_heroes:
                cache.update_counter(ha, hb, t, bool(gts_a.iloc[0]['is_win']) if len(gts_a) > 0 else False)

    # --- 从warmup之后开始构建特征 ---
    print(f"  构建特征 (从第{len(warmup)+1}局起)...")
    for idx in range(len(warmup), len(all_games)):
        game = all_games.iloc[idx]
        bid = game['battle_id']
        t = game['start_time']
        lid = game['league_id']
        ta_id = game['team_a_id']
        tb_id = game['team_b_id']

        pa = player_stats_df[player_stats_df['battle_id'] == bid]
        team_a_players = pa[pa['team_id'] == ta_id]
        team_b_players = pa[pa['team_id'] == tb_id]
        if len(team_a_players) < 5 or len(team_b_players) < 5:
            # 赛后更新后才能继续
            continue

        a_heroes = list(zip(team_a_players['player_name'],
                            team_a_players['hero_id'],
                            team_a_players['position']))
        b_heroes = list(zip(team_b_players['player_name'],
                            team_b_players['hero_id'],
                            team_b_players['position']))
        a_hero_ids = [h[1] for h in a_heroes]
        b_hero_ids = [h[1] for h in b_heroes]

        gts_a = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == ta_id)]
        if len(gts_a) == 0:
            continue
        a_win = bool(gts_a.iloc[0]['is_win'])
        label = 1 if a_win else 0

        f = {}
        f['battle_id'] = bid
        f['league_id'] = lid
        f['start_time'] = t
        f['label'] = label

        # ======== V1特征: 队伍+选手 ========
        ta = cache.get_team(ta_id, t)
        tb = cache.get_team(tb_id, t)
        f['v1_team_wr_diff'] = ta['wr'] - tb['wr']
        f['v1_team_kda_diff'] = ta['avg_kda'] - tb['avg_kda']
        f['v1_team_gold_diff'] = ta['avg_gold'] - tb['avg_gold']
        f['v1_team_tower_diff'] = ta['avg_tower'] - tb['avg_tower']
        f['v1_team_dragon_diff'] = ta['avg_tyrant'] - tb['avg_tyrant']
        f['v1_team_baron_diff'] = ta['avg_big_dragon'] - tb['avg_big_dragon']
        f['v1_team_games_diff'] = ta['games'] - tb['games']

        # 选手对位 (5个位置)
        pos_map = {6: 'top', 5: 'jgl', 2: 'mid', 7: 'adc', 4: 'sup'}
        for pos, tag in pos_map.items():
            pa_pos = team_a_players[team_a_players['position'] == pos]
            pb_pos = team_b_players[team_b_players['position'] == pos]
            a_stat = cache.get_player(pa_pos.iloc[0]['player_name'], t) if len(pa_pos) > 0 else {'avg_kda': 1, 'avg_mvp': 5, 'avg_part_rate': 0.5}
            b_stat = cache.get_player(pb_pos.iloc[0]['player_name'], t) if len(pb_pos) > 0 else {'avg_kda': 1, 'avg_mvp': 5, 'avg_part_rate': 0.5}
            f[f'v1_{tag}_kda_diff'] = a_stat['avg_kda'] - b_stat['avg_kda']
            f[f'v1_{tag}_mvp_diff'] = a_stat['avg_mvp'] - b_stat['avg_mvp']
            f[f'v1_{tag}_part_diff'] = a_stat['avg_part_rate'] - b_stat['avg_part_rate']

        # ======== V2特征: 英雄+阵容 ========
        a_hero_wrs = []
        for h in a_hero_ids:
            wr = cache.get_hero(h, t)['wr']
            if cache.get_hero(h, t)['games'] < 3:
                wr = wr * 0.5 + cache.get_global_hero_wr(h) * 0.5
            a_hero_wrs.append(wr)
        b_hero_wrs = []
        for h in b_hero_ids:
            wr = cache.get_hero(h, t)['wr']
            if cache.get_hero(h, t)['games'] < 3:
                wr = wr * 0.5 + cache.get_global_hero_wr(h) * 0.5
            b_hero_wrs.append(wr)
        f['v2_hero_wr_avg_a'] = np.mean(a_hero_wrs)
        f['v2_hero_wr_avg_b'] = np.mean(b_hero_wrs)
        f['v2_hero_wr_diff'] = f['v2_hero_wr_avg_a'] - f['v2_hero_wr_avg_b']

        # 选手英雄熟练度
        a_ph_wrs = [cache.get_player_hero(pn, hid, t)['wr'] for pn, hid, _ in a_heroes]
        b_ph_wrs = [cache.get_player_hero(pn, hid, t)['wr'] for pn, hid, _ in b_heroes]
        f['v2_ph_wr_avg_a'] = np.mean(a_ph_wrs)
        f['v2_ph_wr_avg_b'] = np.mean(b_ph_wrs)
        f['v2_ph_wr_diff'] = f['v2_ph_wr_avg_a'] - f['v2_ph_wr_avg_b']

        # 阵容协同
        syn_a, syn_b = [], []
        for i in range(len(a_hero_ids)):
            for j in range(i+1, len(a_hero_ids)):
                syn_a.append(cache.get_synergy(a_hero_ids[i], a_hero_ids[j], t)['wr'])
        for i in range(len(b_hero_ids)):
            for j in range(i+1, len(b_hero_ids)):
                syn_b.append(cache.get_synergy(b_hero_ids[i], b_hero_ids[j], t)['wr'])
        f['v2_synergy_mean_a'] = np.mean(syn_a) if syn_a else 0.5
        f['v2_synergy_mean_b'] = np.mean(syn_b) if syn_b else 0.5
        f['v2_synergy_diff'] = f['v2_synergy_mean_a'] - f['v2_synergy_mean_b']
        f['v2_synergy_min_a'] = np.min(syn_a) if syn_a else 0.5

        # 克制
        ctrs = []
        for ha in a_hero_ids:
            for hb in b_hero_ids:
                ctrs.append(cache.get_counter(ha, hb, t)['wr'])
        f['v2_counter_mean'] = np.mean(ctrs) if ctrs else 0.5
        f['v2_counter_max'] = np.max(ctrs) if ctrs else 0.5
        f['v2_counter_min'] = np.min(ctrs) if ctrs else 0.5

        # 核心位强度
        core_a = [h for pn, h, pos in a_heroes if pos in (5, 7)]
        core_b = [h for pn, h, pos in b_heroes if pos in (5, 7)]
        f['v2_core_wr_a'] = np.mean([cache.get_hero(h, t)['wr'] for h in core_a]) if core_a else 0.5
        f['v2_core_wr_b'] = np.mean([cache.get_hero(h, t)['wr'] for h in core_b]) if core_b else 0.5
        f['v2_core_wr_diff'] = f['v2_core_wr_a'] - f['v2_core_wr_b']

        # 英雄多样性 (对方不知道你要玩什么)
        played_heroes_a = len(set(h for _, h, _ in a_heroes))
        played_heroes_b = len(set(h for _, h, _ in b_heroes))

        samples.append(f)

        # --- 赛后更新缓存 (与warmup相同逻辑) ---
        gts_a2 = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == ta_id)]
        gts_b2 = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == tb_id)]
        if len(gts_a2) > 0:
            r = gts_a2.iloc[0]
            tyrant = int(r['tyrant_kills'] or 0) + int(r['dark_tyrant_kills'] or 0)
            cache.update_team(ta_id, bid, t, int(r['kills']), int(r['deaths']),
                              int(r['assists']), float(r['kda']), int(r['gold']),
                              int(r['tower_kills']), tyrant, int(r['big_dragon_kills']), bool(r['is_win']))
        if len(gts_b2) > 0:
            r = gts_b2.iloc[0]
            tyrant = int(r['tyrant_kills'] or 0) + int(r['dark_tyrant_kills'] or 0)
            cache.update_team(tb_id, bid, t, int(r['kills']), int(r['deaths']),
                              int(r['assists']), float(r['kda']), int(r['gold']),
                              int(r['tower_kills']), tyrant, int(r['big_dragon_kills']), bool(r['is_win']))

        for _, pl in pa.iterrows():
            team_win = a_win if pl['team_id'] == ta_id else (not a_win)
            cache.update_player(pl['player_name'], pl['team_id'], pl['hero_id'],
                                pl['position'], t, bid, int(pl['kills']), int(pl['deaths']),
                                int(pl['assists']), float(pl['kda'] or 1),
                                float(pl['mvp_score']), int(pl['gold']),
                                int(pl['hurt_to_hero']), float(pl['participation_rate']))
            cache.update_hero(pl['hero_id'], t, team_win)
            cache.update_player_hero(pl['player_name'], pl['hero_id'], t, team_win)

        for i in range(len(a_heroes)):
            for j in range(i+1, len(a_heroes)):
                cache.update_synergy(a_heroes[i][1], a_heroes[j][1], t, a_win)
        for i in range(len(b_heroes)):
            for j in range(i+1, len(b_heroes)):
                cache.update_synergy(b_heroes[i][1], b_heroes[j][1], t, not a_win)
        for ha, _, _ in a_heroes:
            for hb, _, _ in b_heroes:
                cache.update_counter(ha, hb, t, a_win)

        if (idx + 1) % 400 == 0:
            print(f"  已处理 {idx+1} 局...")

    return pd.DataFrame(samples)


def train_model(feature_df):
    exclude = ['battle_id', 'league_id', 'start_time', 'label']
    feats = [c for c in feature_df.columns if c not in exclude]

    # 按时序排序
    df_sorted = feature_df.sort_values('start_time').reset_index(drop=True)
    N = len(df_sorted)

    # 用前80%时间训练, 后20%时间测试 (所有赛季)
    split_idx = int(N * 0.8)
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    print(f"\n全量数据: {N} 局")
    print(f"训练集: {len(train)} 局 (前80%时间)")
    print(f"测试集: {len(test)} 局 (后20%时间)")
    print(f"特征维度: {len(feats)}")

    X_train = train[feats].fillna(0)
    y_train = train['label']
    X_test = test[feats].fillna(0)
    y_test = test['label']

    print(f"训练正样本率: {y_train.mean():.3f} | 测试正样本率: {y_test.mean():.3f}")

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        num_leaves=31, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_weight, random_state=42, verbose=-1,
    )

    # 滚动交叉验证 (更可靠的评估)
    from sklearn.model_selection import TimeSeriesSplit
    print("\n滚动时间序列交叉验证 (5折):")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_acc, cv_auc = [], []
    X_cv = df_sorted[feats].fillna(0)
    y_cv = df_sorted['label']

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, X_val = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]

        m = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                               random_state=42, verbose=-1)
        m.fit(X_tr, y_tr)
        yp = m.predict(X_val)
        ypb = m.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, yp)
        auc = roc_auc_score(y_val, ypb)
        cv_acc.append(acc)
        cv_auc.append(auc)
        print(f"  Fold {fold+1}: train={len(y_tr)} val={len(y_val)}, Acc={acc:.4f}, AUC={auc:.4f}")

    print(f"\nCV平均: Acc={np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}, "
          f"AUC={np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

    # 最终在80/20划分上训练评估
    print("\n最终模型 (80/20划分):")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['B队胜','A队胜'])}")

    imp = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})\
             .sort_values('importance', ascending=False)
    print("Top 15 特征:")
    for _, r in imp.head(15).iterrows():
        tag = 'V1' if r['feature'].startswith('v1_') else 'V2'
        print(f"  [{tag}] {r['feature']:35s} {r['importance']:.1f}")

    test = test.copy()
    test['pred'] = y_proba
    test['correct'] = (y_pred == y_test)
    test['bin'] = pd.cut(test['pred'], [0, 0.35, 0.45, 0.55, 0.65, 1.0])
    print("\n置信度分档:")
    for b, g in test.groupby('bin', observed=False):
        if len(g):
            print(f"  {b}: {len(g)}局, Acc={g['correct'].mean():.3f}")

    return model, feats, imp, test


def main():
    print("=" * 60)
    print("KPL V2 完整模型 (V1特征 + 英雄阵容)")
    print("=" * 60)

    print("\n[1/3] 加载数据...")
    games, team_stats, player_stats = load_all()
    print(f"  {len(games)} 小局")

    print("\n[2/3] 特征工程...")
    df = build_features(games, team_stats, player_stats)
    print(f"  样本: {len(df)}")
    df.to_csv('e:/gediandata/kpl/features_v2.csv', index=False, encoding='utf-8-sig')

    print("\n[3/3] 训练...")
    model, feats, imp, preds = train_model(df)

    model.booster_.save_model('e:/gediandata/kpl/v2_model.txt')
    imp.to_csv('e:/gediandata/kpl/feature_importance_v2.csv', index=False, encoding='utf-8-sig')
    preds.to_csv('e:/gediandata/kpl/predictions_v2.csv', index=False, encoding='utf-8-sig')

    print("\n模型已保存: v2_model.txt")


if __name__ == "__main__":
    main()
