"""
KPL V3 优化预测模型
= V2全部特征 + 红蓝方 + 交手历史 + 连胜趋势 + 比赛阶段 - 噪声特征
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

WINDOW = 60  # 基本滚动窗口
LONG_WINDOW = 180  # 长窗口用于英雄统计


def load_all():
    conn = pymysql.connect(**DB)
    games = pd.read_sql("""
        SELECT g.battle_id, g.match_id, g.battle_seq, g.win_camp, g.game_duration,
               m.league_id, m.start_time, m.team_a_id, m.team_b_id,
               m.stage_name, m.stage_desc
        FROM games g JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 ORDER BY m.start_time, g.battle_seq
    """, conn, parse_dates=['start_time'])

    team_stats = pd.read_sql("""
        SELECT gts.*, g.win_camp, m.start_time
        FROM game_team_stats gts
        JOIN games g ON gts.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 ORDER BY m.start_time
    """, conn, parse_dates=['start_time'])

    player_stats = pd.read_sql("""
        SELECT gps.*, g.win_camp, m.start_time
        FROM game_player_stats gps
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 ORDER BY m.start_time
    """, conn, parse_dates=['start_time'])

    conn.close()
    return games, team_stats, player_stats


class V3Cache:
    """V3缓存: 保留V2能力 + 新增交手/连胜/红蓝方"""

    def __init__(self):
        self.data = {}  # key -> list of dicts
        self.window = WINDOW
        self.long_window = LONG_WINDOW

    def _get(self, key):
        return self.data.get(key, [])

    def _prune(self, key, t, window=None):
        w = window or self.window
        cutoff = t - timedelta(days=w)
        if key in self.data:
            self.data[key] = [r for r in self.data[key] if r['time'] >= cutoff]

    def _add(self, key, record):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(record)

    def _query(self, key, t, window=None):
        w = window or self.window
        cutoff = t - timedelta(days=w)
        return [r for r in self._get(key) if r['time'] < t and r['time'] >= cutoff]

    def _safe_mean(self, recs, field, default):
        if not recs: return default
        vals = [r[field] for r in recs if field in r]
        return float(np.mean(vals)) if vals else default

    def _safe_count(self, recs):
        return len(recs)

    def _safe_streak(self, recs):
        """计算最近连胜/连败"""
        if not recs: return 0
        sorted_recs = sorted(recs, key=lambda r: r['time'], reverse=True)
        streak = 0
        target = sorted_recs[0].get('is_win')
        for r in sorted_recs:
            if r.get('is_win') == target:
                streak += 1 if target else -1
            else:
                break
        return streak

    # ======== 更新 ========
    def update_team(self, tid, bid, t, kills, deaths, assists, kda,
                    gold, tower, tyrant, big_dragon, is_win):
        key = f"T_{tid}"
        self._add(key, {'time': t, 'bid': bid, 'is_win': is_win,
                        'kills': kills, 'deaths': deaths, 'assists': assists,
                        'kda': kda, 'gold': gold, 'tower': tower,
                        'tyrant': tyrant, 'big_dragon': big_dragon})
        self._prune(key, t)

    def update_player(self, pn, tid, hid, pos, t, bid,
                      kills, deaths, assists, kda, mvp, gold, hurt_hero, part_rate):
        key = f"P_{pn}"
        self._add(key, {'time': t, 'bid': bid, 'tid': tid, 'hid': hid, 'pos': pos,
                        'kills': kills, 'deaths': deaths, 'assists': assists,
                        'kda': kda, 'mvp': mvp, 'gold': gold,
                        'hurt_hero': hurt_hero, 'part_rate': part_rate})
        self._prune(key, t)

    def update_hero(self, hid, t, is_win):
        key = f"H_{hid}"
        self._add(key, {'time': t, 'is_win': is_win})
        self._prune(key, t, self.long_window)

    def update_player_hero(self, pn, hid, t, is_win):
        key = f"PH_{pn}_{hid}"
        self._add(key, {'time': t, 'is_win': is_win})
        self._prune(key, t, self.long_window)

    def update_synergy(self, ha, hb, t, is_win):
        a, b = (ha, hb) if ha < hb else (hb, ha)
        key = f"SYN_{a}_{b}"
        self._add(key, {'time': t, 'is_win': is_win})
        self._prune(key, t, self.long_window)

    def update_counter(self, ha, hb, t, ha_win):
        key = f"CTR_{ha}_{hb}"
        self._add(key, {'time': t, 'is_win': ha_win})
        self._prune(key, t, self.long_window)

    def update_h2h(self, ta, tb, ta_win, t):
        """交手历史: 记录A_vs_B的结果"""
        key = f"H2H_{min(ta,tb)}_{max(ta,tb)}"
        self._add(key, {'time': t, 'ta': ta, 'tb': tb, 'ta_win': ta_win})

    def update_side_win(self, camp, is_win, t):
        """红蓝方胜率"""
        key = f"SIDE_{camp}"
        self._add(key, {'time': t, 'is_win': is_win})
        self._prune(key, t, self.long_window)

    # ======== 查询 ========
    def get_team(self, tid, t):
        recs = self._query(f"T_{tid}", t)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
            'avg_kda': self._safe_mean(recs, 'kda', 1.0),
            'avg_gold': self._safe_mean(recs, 'gold', 0),
            'avg_tower': self._safe_mean(recs, 'tower', 0),
            'avg_dragon': self._safe_mean(recs, 'tyrant', 0),
            'avg_baron': self._safe_mean(recs, 'big_dragon', 0),
            'streak': self._safe_streak(recs),
        }

    def get_player(self, pn, t):
        recs = self._query(f"P_{pn}", t)
        recent5 = self._query(f"P_{pn}", t)[:5] if recs else []
        return {
            'avg_kda': self._safe_mean(recs, 'kda', 1.0),
            'avg_mvp': self._safe_mean(recs, 'mvp', 5.0),
            'avg_part_rate': self._safe_mean(recs, 'part_rate', 0.5),
            'avg_gold': self._safe_mean(recs, 'gold', 5000),
            'avg_hurt_hero': self._safe_mean(recs, 'hurt_hero', 30000),
            'games': self._safe_count(recs),
            # 近期趋势: 最近5场 vs 全部
            'kda_trend': self._safe_mean(recent5, 'kda', 1.0)
                         - self._safe_mean(recs, 'kda', 1.0),
        }

    def get_hero(self, hid, t):
        recs = self._query(f"H_{hid}", t, self.long_window)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_player_hero(self, pn, hid, t):
        recs = self._query(f"PH_{pn}_{hid}", t, self.long_window)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_synergy(self, ha, hb, t):
        a, b = (ha, hb) if ha < hb else (hb, ha)
        recs = self._query(f"SYN_{a}_{b}", t, self.long_window)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_counter(self, ha, hb, t):
        recs = self._query(f"CTR_{ha}_{hb}", t, self.long_window)
        return {
            'wr': self._safe_mean(recs, 'is_win', 0.5),
            'games': self._safe_count(recs),
        }

    def get_h2h(self, ta, tb, t):
        """A对B的历史胜率"""
        key = f"H2H_{min(ta,tb)}_{max(ta,tb)}"
        all_recs = self._get(key)
        recs = [r for r in all_recs if r['time'] < t]
        if not recs: return {'ta_wr': 0.5, 'games': 0}
        ta_wins = sum(1 for r in recs if (r['ta'] == ta and r['ta_win']) or
                                          (r['tb'] == ta and not r['ta_win']))
        return {'ta_wr': ta_wins / len(recs), 'games': len(recs)}

    def get_side_win_rate(self, camp, t):
        """获取某阵营(camp)的历史胜率"""
        recs = self._query(f"SIDE_{camp}", t, self.long_window)
        return self._safe_mean(recs, 'is_win', 0.5)

    def get_global_hero_wr(self, hid):
        """全局英雄胜率(冷启动回退)"""
        recs = self._get(f"H_{hid}")
        if not recs:
            return 0.5
        return float(np.mean([r['is_win'] for r in recs]))

    def get_hero_ban_rate(self, hid, t):
        """英雄被Ban率 - 用全局hero_stats表近似"""
        # 简化: 用近期games中该英雄的pick率反推
        # 实际应该从draft_picks统计,这里用100 - pick_rate近似
        recs = self._query(f"H_{hid}", t, self.long_window)
        total_games = self._safe_count(self._query(f"T_0", t))  # 用team数据估计总场次
        if total_games > 0:
            pick_rate = self._safe_count(recs) / (total_games * 10)  # 每局10个英雄
            return max(0, 1 - pick_rate * 20)  # 近似: ban率 ≈ 1 - 20×pick率
        return 0.5


def build_features(games_df, team_stats_df, player_stats_df):
    cache = V3Cache()
    all_games = games_df.sort_values('start_time').reset_index(drop=True)

    # Warmup: 前1/6的数据
    warmup_size = len(all_games) // 6
    warmup = all_games.iloc[:warmup_size]
    print(f"  预热: 前 {warmup_size} 局导入缓存...")

    def update_cache_from_game(game):
        bid = game['battle_id']
        t = game['start_time']
        ta = game['team_a_id']
        tb = game['team_b_id']
        pa = player_stats_df[player_stats_df['battle_id'] == bid]
        gts_a = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == ta)]
        gts_b = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == tb)]

        a_win = bool(gts_a.iloc[0]['is_win']) if len(gts_a) > 0 else False
        b_win = bool(gts_b.iloc[0]['is_win']) if len(gts_b) > 0 else False
        a_camp = int(pa[pa['team_id'] == ta].iloc[0]['camp']) if len(pa[pa['team_id'] == ta]) > 0 else 1
        b_camp = 3 - a_camp

        for gts, tid, win in [(gts_a, ta, a_win), (gts_b, tb, b_win)]:
            if len(gts) > 0:
                r = gts.iloc[0]
                tyrant = int(r.get('tyrant_kills', 0) or 0) + int(r.get('dark_tyrant_kills', 0) or 0)
                cache.update_team(tid, bid, t,
                                  int(r.get('kills', 0)), int(r.get('deaths', 0)),
                                  int(r.get('assists', 0)), float(r.get('kda', 1)),
                                  int(r.get('gold', 0)), int(r.get('tower_kills', 0)),
                                  tyrant, int(r.get('big_dragon_kills', 0)), win)

        for _, pl in pa.iterrows():
            is_win = a_win if pl['team_id'] == ta else b_win
            cache.update_player(str(pl['player_name']), int(pl['team_id']),
                                int(pl['hero_id']), int(pl['position']), t, bid,
                                int(pl['kills']), int(pl['deaths']), int(pl['assists']),
                                float(pl['kda'] or 1), float(pl['mvp_score']),
                                int(pl['gold']), int(pl['hurt_to_hero']),
                                float(pl['participation_rate']))
            cache.update_hero(int(pl['hero_id']), t, is_win)
            cache.update_player_hero(str(pl['player_name']), int(pl['hero_id']), t, is_win)

        # 协同
        a_heroes = [(int(pl['hero_id']), str(pl['player_name'])) for _, pl in pa.iterrows()
                     if pl['team_id'] == ta]
        b_heroes = [(int(pl['hero_id']), str(pl['player_name'])) for _, pl in pa.iterrows()
                     if pl['team_id'] == tb]
        for i in range(len(a_heroes)):
            for j in range(i+1, len(a_heroes)):
                cache.update_synergy(a_heroes[i][0], a_heroes[j][0], t, a_win)
        for i in range(len(b_heroes)):
            for j in range(i+1, len(b_heroes)):
                cache.update_synergy(b_heroes[i][0], b_heroes[j][0], t, b_win)

        # 克制
        for ha, _ in a_heroes:
            for hb, _ in b_heroes:
                cache.update_counter(ha, hb, t, a_win)
                cache.update_counter(hb, ha, t, b_win)

        # 交手
        cache.update_h2h(ta, tb, a_win, t)

        # 阵营
        cache.update_side_win(a_camp, a_win, t)
        cache.update_side_win(b_camp, b_win, t)

    for _, game in warmup.iterrows():
        update_cache_from_game(game)

    # 构建特征
    print(f"  构建特征 (从第{warmup_size+1}局起)...")
    samples = []

    for idx in range(warmup_size, len(all_games)):
        game = all_games.iloc[idx]
        bid = game['battle_id']
        t = game['start_time']
        lid = game['league_id']
        ta_id = game['team_a_id']
        tb_id = game['team_b_id']
        stage = game.get('stage_name', '') or ''

        pa = player_stats_df[player_stats_df['battle_id'] == bid]
        team_a_players = pa[pa['team_id'] == ta_id]
        team_b_players = pa[pa['team_id'] == tb_id]
        if len(team_a_players) < 5 or len(team_b_players) < 5:
            update_cache_from_game(game)
            continue

        a_heroes = list(zip(team_a_players['player_name'],
                            team_a_players['hero_id'].astype(int),
                            team_a_players['position'].astype(int)))
        b_heroes = list(zip(team_b_players['player_name'],
                            team_b_players['hero_id'].astype(int),
                            team_b_players['position'].astype(int)))
        a_hero_ids = [h[1] for h in a_heroes]
        b_hero_ids = [h[1] for h in b_heroes]
        a_camp_val = int(team_a_players.iloc[0]['camp']) if 'camp' in team_a_players.columns else 1

        gts_a = team_stats_df[(team_stats_df['battle_id'] == bid) & (team_stats_df['team_id'] == ta_id)]
        if len(gts_a) == 0:
            update_cache_from_game(game)
            continue
        a_win = bool(gts_a.iloc[0]['is_win'])
        label = 1 if a_win else 0

        f = {}
        f['battle_id'] = bid
        f['league_id'] = lid
        f['start_time'] = t
        f['label'] = label

        # ============ V1: 队伍 + 选手 ============
        ta = cache.get_team(ta_id, t)
        tb = cache.get_team(tb_id, t)
        f['team_wr_diff'] = ta['wr'] - tb['wr']
        f['team_kda_diff'] = ta['avg_kda'] - tb['avg_kda']
        f['team_baron_diff'] = ta['avg_baron'] - tb['avg_baron']
        f['team_dragon_diff'] = ta['avg_dragon'] - tb['avg_dragon']
        f['team_games_diff'] = ta['games'] - tb['games']

        # 连胜
        f['team_streak_a'] = ta['streak']
        f['team_streak_b'] = tb['streak']
        f['team_streak_diff'] = ta['streak'] - tb['streak']

        # 选手对位(5位置×3指标)
        pos_map = {6: 'top', 5: 'jgl', 2: 'mid', 7: 'adc', 4: 'sup'}
        for pos, tag in pos_map.items():
            pa_pos = team_a_players[team_a_players['position'] == pos]
            pb_pos = team_b_players[team_b_players['position'] == pos]
            a_stat = cache.get_player(pa_pos.iloc[0]['player_name'], t) if len(pa_pos) > 0 else {}
            b_stat = cache.get_player(pb_pos.iloc[0]['player_name'], t) if len(pb_pos) > 0 else {}
            f[f'{tag}_kda_diff'] = a_stat.get('avg_kda', 1) - b_stat.get('avg_kda', 1)
            f[f'{tag}_mvp_diff'] = a_stat.get('avg_mvp', 5) - b_stat.get('avg_mvp', 5)
            f[f'{tag}_part_diff'] = a_stat.get('avg_part_rate', 0.5) - b_stat.get('avg_part_rate', 0.5)
            # 趋势
            f[f'{tag}_kda_trend'] = a_stat.get('kda_trend', 0) - b_stat.get('kda_trend', 0)

        # ============ V2: 英雄 + 阵容 ============
        a_hero_wrs, b_hero_wrs = [], []
        for h in a_hero_ids:
            wr = cache.get_hero(h, t)['wr']
            if cache.get_hero(h, t)['games'] < 3:
                wr = wr * 0.3 + cache.get_global_hero_wr(h) * 0.7
            a_hero_wrs.append(wr)
        for h in b_hero_ids:
            wr = cache.get_hero(h, t)['wr']
            if cache.get_hero(h, t)['games'] < 3:
                wr = wr * 0.3 + cache.get_global_hero_wr(h) * 0.7
            b_hero_wrs.append(wr)

        f['hero_wr_avg_a'] = np.mean(a_hero_wrs)
        f['hero_wr_avg_b'] = np.mean(b_hero_wrs)
        f['hero_wr_diff'] = f['hero_wr_avg_a'] - f['hero_wr_avg_b']

        # 选手-英雄熟练度
        a_ph = [cache.get_player_hero(pn, hid, t)['wr'] for pn, hid, _ in a_heroes]
        b_ph = [cache.get_player_hero(pn, hid, t)['wr'] for pn, hid, _ in b_heroes]
        f['ph_wr_avg_a'] = np.mean(a_ph)
        f['ph_wr_avg_b'] = np.mean(b_ph)
        f['ph_wr_diff'] = f['ph_wr_avg_a'] - f['ph_wr_avg_b']

        # 阵容协同
        syn_a, syn_b = [], []
        for i in range(len(a_hero_ids)):
            for j in range(i+1, len(a_hero_ids)):
                syn_a.append(cache.get_synergy(a_hero_ids[i], a_hero_ids[j], t)['wr'])
        for i in range(len(b_hero_ids)):
            for j in range(i+1, len(b_hero_ids)):
                syn_b.append(cache.get_synergy(b_hero_ids[i], b_hero_ids[j], t)['wr'])
        f['synergy_mean_a'] = np.mean(syn_a) if syn_a else 0.5
        f['synergy_mean_b'] = np.mean(syn_b) if syn_b else 0.5
        f['synergy_diff'] = f['synergy_mean_a'] - f['synergy_mean_b']

        # 克制 (只保留mean, 去掉max/min)
        ctrs = []
        for ha in a_hero_ids:
            for hb in b_hero_ids:
                ctrs.append(cache.get_counter(ha, hb, t)['wr'])
        f['counter_mean'] = np.mean(ctrs) if ctrs else 0.5

        # 核心位
        core_a = [h for _, h, pos in a_heroes if pos in (5, 7)]
        core_b = [h for _, h, pos in b_heroes if pos in (5, 7)]
        f['core_wr_a'] = np.mean([cache.get_hero(h, t)['wr'] for h in core_a]) if core_a else 0.5
        f['core_wr_b'] = np.mean([cache.get_hero(h, t)['wr'] for h in core_b]) if core_b else 0.5
        f['core_wr_diff'] = f['core_wr_a'] - f['core_wr_b']

        # ============ V3 新增特征 ============

        # 红蓝方 (只保留胜率，不保留阵营标签本身)
        f['side_win_rate'] = cache.get_side_win_rate(a_camp_val, t)

        # 交手历史
        h2h = cache.get_h2h(ta_id, tb_id, t)
        f['h2h_wr'] = h2h['ta_wr']
        f['h2h_games'] = h2h['games']

        samples.append(f)

        # 赛后更新
        update_cache_from_game(game)

        if (idx + 1) % 400 == 0:
            print(f"  已处理 {idx+1} 局...")

    return pd.DataFrame(samples)


def train_model(feature_df):
    exclude = ['battle_id', 'league_id', 'start_time', 'label']
    feats = [c for c in feature_df.columns if c not in exclude]

    df_sorted = feature_df.sort_values('start_time').reset_index(drop=True)
    N = len(df_sorted)
    split_idx = int(N * 0.8)
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    print(f"\n全量: {N}局 | 训练: {len(train)}局 | 测试: {len(test)}局")
    print(f"特征: {len(feats)}维")
    print(f"训练正样本率: {train['label'].mean():.3f} | 测试: {test['label'].mean():.3f}")

    X_train = train[feats].fillna(0)
    y_train = train['label']
    X_test = test[feats].fillna(0)
    y_test = test['label']

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        num_leaves=31, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_weight, random_state=42, verbose=-1,
    )

    # 5折时间序列CV
    from sklearn.model_selection import TimeSeriesSplit
    print("\n5折CV:")
    tscv = TimeSeriesSplit(n_splits=5)
    X_cv = df_sorted[feats].fillna(0)
    y_cv = df_sorted['label']
    cv_acc, cv_auc = [], []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        m = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                               random_state=42, verbose=-1)
        m.fit(X_cv.iloc[tr_idx], y_cv.iloc[tr_idx])
        yp = m.predict(X_cv.iloc[val_idx])
        ypb = m.predict_proba(X_cv.iloc[val_idx])[:, 1]
        cv_acc.append(accuracy_score(y_cv.iloc[val_idx], yp))
        cv_auc.append(roc_auc_score(y_cv.iloc[val_idx], ypb))
        print(f"  Fold {fold+1}: Acc={cv_acc[-1]:.4f}, AUC={cv_auc[-1]:.4f}")
    print(f"  CV平均: Acc={np.mean(cv_acc):.4f}±{np.std(cv_acc):.4f}, AUC={np.mean(cv_auc):.4f}±{np.std(cv_auc):.4f}")

    # 最终80/20评估
    print("\n=== V3 测试集 ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {acc:.4f}  AUC: {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['B队胜','A队胜'])}")

    imp = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})\
             .sort_values('importance', ascending=False)
    print("Top 20 特征重要性:")
    for _, r in imp.head(20).iterrows():
        bar = '█' * int(r['importance'] / 5)
        print(f"  {r['feature']:35s} {r['importance']:6.1f} {bar}")

    # 新增特征单独看
    new_feats = ['side_win_rate', 'h2h_wr', 'h2h_games',
                 'team_streak_a', 'team_streak_b', 'team_streak_diff']
    print("\n=== V3新增特征表现 ===")
    for nf in new_feats:
        if nf in imp['feature'].values:
            val = imp[imp['feature'] == nf]['importance'].values[0]
            print(f"  {nf:25s} 重要性={val:.1f}")

    # 置信度
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
    print("KPL V3 优化模型 (V2 + 红蓝方/交手/连胜/阶段)")
    print("=" * 60)

    print("\n[1/3] 加载...")
    games, team_stats, player_stats = load_all()

    print("\n[2/3] 特征工程...")
    df = build_features(games, team_stats, player_stats)
    print(f"  样本: {len(df)}")
    df.to_csv('e:/gediandata/kpl/features_v3.csv', index=False, encoding='utf-8-sig')

    print("\n[3/3] 训练...")
    model, feats, imp, preds = train_model(df)

    model.booster_.save_model('e:/gediandata/kpl/v3_model.txt')
    imp.to_csv('e:/gediandata/kpl/feature_importance_v3.csv', index=False, encoding='utf-8-sig')
    preds.to_csv('e:/gediandata/kpl/predictions_v3.csv', index=False, encoding='utf-8-sig')

    # 对比V2
    print("\n" + "=" * 60)
    print("V2 vs V3 对比")
    print("=" * 60)
    try:
        v2 = pd.read_csv('e:/gediandata/kpl/features_v2_archive.csv')
        v2 = v2.sort_values('start_time').reset_index(drop=True)
        split = int(len(v2) * 0.8)
        v2_test = v2.iloc[split:]
        print(f"  V2 测试集: {len(v2_test)}局, 正样本率={v2_test['label'].mean():.3f}")
    except Exception as e:
        print(f"  V2对比失败: {e}")

    print(f"\n  V3 测试集: {len(preds)}局, Acc={preds['correct'].mean():.4f}  ← 最终结果")


if __name__ == "__main__":
    main()
