"""
KPL V1 胜负预测模型
LightGBM 二分类: 预测大场中每小局的胜负
特征: 赛前滚动统计 + 队伍差 + 选手对位差
训练/验证: 时间切分 (前5赛季train, 最新赛季test)
"""

import pymysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DB = {
    "host": "localhost", "port": 3306, "user": "root",
    "password": "auba7956", "database": "kpl", "charset": "utf8mb4",
}

# ============================================================
# 1. 从MySQL加载原始数据
# ============================================================

def load_data():
    conn = pymysql.connect(**DB)

    # 大场 + 小局, 按时间排序
    games = pd.read_sql("""
        SELECT g.battle_id, g.match_id, g.battle_seq, g.game_duration, g.win_camp,
               m.league_id, m.start_time, m.team_a_id, m.team_b_id,
               m.team_a_score, m.team_b_score
        FROM games g
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2 AND m.start_time IS NOT NULL
        ORDER BY m.start_time, g.battle_seq
    """, conn)

    # 小局队伍数据
    team_stats = pd.read_sql("""
        SELECT gts.*, g.match_id, m.league_id, m.start_time
        FROM game_team_stats gts
        JOIN games g ON gts.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2
        ORDER BY m.start_time
    """, conn)

    # 小局选手数据
    player_stats = pd.read_sql("""
        SELECT gps.*, g.match_id, m.league_id, m.start_time
        FROM game_player_stats gps
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE g.status = 2
        ORDER BY m.start_time
    """, conn)

    # 队伍信息
    teams = pd.read_sql("SELECT * FROM teams", conn)
    conn.close()

    # 统一时间格式
    for df in [games, team_stats, player_stats]:
        df['start_time'] = pd.to_datetime(df['start_time'])

    return games, team_stats, player_stats, teams


# ============================================================
# 2. 特征工程: 为每场比赛计算赛前滚动统计
# ============================================================

def compute_team_rolling_features(team_stats_df, window_days=30):
    """
    为每场比赛的每个队伍计算赛前滚动特征
    返回: DataFrame, index=(battle_id, team_id), columns=特征
    """
    df = team_stats_df.copy()

    # 按时间排序
    df = df.sort_values('start_time')

    features_list = []

    for idx, row in df.iterrows():
        battle_id = row['battle_id']
        team_id = row['team_id']
        current_time = row['start_time']
        cutoff = current_time - timedelta(days=window_days)

        # 找到该队伍在该时间之前的所有比赛
        hist = df[(df['team_id'] == team_id) &
                  (df['start_time'] < current_time) &
                  (df['start_time'] >= cutoff) &
                  (df['battle_id'] != battle_id)]

        # 也计算全历史的 (不限制时间窗口)
        hist_all = df[(df['team_id'] == team_id) &
                      (df['start_time'] < current_time) &
                      (df['battle_id'] != battle_id)]

        feats = {'battle_id': battle_id, 'team_id': team_id}

        # 近期窗口特征 (反映当前状态)
        if len(hist) > 0:
            feats['team_win_rate_30d'] = hist['is_win'].mean()
            feats['team_avg_kda_30d'] = hist['kda'].mean()
            feats['team_avg_kills_30d'] = hist['kills'].mean()
            feats['team_avg_deaths_30d'] = hist['deaths'].mean()
            feats['team_avg_gold_30d'] = hist['gold'].mean()
            feats['team_avg_tower_30d'] = hist['tower_kills'].mean()
            feats['team_avg_big_dragon_30d'] = hist['big_dragon_kills'].mean()
            feats['team_tyrant_cnt_30d'] = (hist['tyrant_kills'] + hist['dark_tyrant_kills']).mean()
        else:
            feats['team_win_rate_30d'] = 0.5
            feats['team_avg_kda_30d'] = 1.0
            feats['team_avg_kills_30d'] = 0
            feats['team_avg_deaths_30d'] = 0
            feats['team_avg_gold_30d'] = 0
            feats['team_avg_tower_30d'] = 0
            feats['team_avg_big_dragon_30d'] = 0
            feats['team_tyrant_cnt_30d'] = 0

        # 全历史特征 (反映长期实力)
        if len(hist_all) > 0:
            feats['team_win_rate_all'] = hist_all['is_win'].mean()
            feats['team_avg_kda_all'] = hist_all['kda'].mean()
            feats['team_avg_gold_all'] = hist_all['gold'].mean()
            feats['team_games_all'] = len(hist_all)
        else:
            feats['team_win_rate_all'] = 0.5
            feats['team_avg_kda_all'] = 1.0
            feats['team_avg_gold_all'] = 0
            feats['team_games_all'] = 0

        features_list.append(feats)

    feat_df = pd.DataFrame(features_list)
    return feat_df


def compute_player_rolling_features(player_stats_df, window_days=30):
    """
    为每场比赛的每个选手计算赛前滚动特征
    """
    df = player_stats_df.copy()
    df = df.sort_values('start_time')

    features_list = []

    for idx, row in df.iterrows():
        battle_id = row['battle_id']
        player_name = row['player_name']
        current_time = row['start_time']
        cutoff = current_time - timedelta(days=window_days)

        hist = df[(df['player_name'] == player_name) &
                  (df['start_time'] < current_time) &
                  (df['start_time'] >= cutoff) &
                  (df['battle_id'] != battle_id)]

        feats = {'battle_id': battle_id, 'player_name': player_name,
                 'team_id': row['team_id']}

        if len(hist) > 0:
            feats['player_avg_kda_30d'] = hist['kda'].mean()
            feats['player_avg_mvp_30d'] = hist['mvp_score'].mean()
            feats['player_avg_participation_30d'] = hist['participation_rate'].mean()
            feats['player_avg_gold_30d'] = hist['gold'].mean()
            feats['player_avg_hurt_hero_30d'] = hist['hurt_to_hero'].mean()
            feats['player_games_30d'] = len(hist)
        else:
            feats['player_avg_kda_30d'] = 1.0
            feats['player_avg_mvp_30d'] = 5.0
            feats['player_avg_participation_30d'] = 0.5
            feats['player_avg_gold_30d'] = 5000
            feats['player_avg_hurt_hero_30d'] = 30000
            feats['player_games_30d'] = 0

        features_list.append(feats)

    return pd.DataFrame(features_list)


def build_dataset(games, team_stats, player_stats, teams):
    """
    构建训练数据集
    每个样本 = 一小局, 特征是两队赛前统计的差值
    """
    print("计算队伍滚动特征...")
    team_feat = compute_team_rolling_features(team_stats)

    print("计算选手滚动特征...")
    player_feat = compute_player_rolling_features(player_stats)

    # 合并到 games 表
    games = games.merge(teams, left_on='team_a_id', right_on='team_id')
    team_a_name = teams.rename(columns={c: f'team_a_{c}' for c in teams.columns})
    team_b_name = teams.rename(columns={c: f'team_b_{c}' for c in teams.columns})

    print("构建特征矩阵...")
    samples = []

    for idx, game in games.iterrows():
        bid = game['battle_id']
        start_time = game['start_time']
        league_id = game['league_id']

        # 找该场次的队伍特征
        ta_feat = team_feat[(team_feat['battle_id'] == bid) &
                            (team_feat['team_id'] == game['team_a_id'])]
        tb_feat = team_feat[(team_feat['battle_id'] == bid) &
                            (team_feat['team_id'] == game['team_b_id'])]

        if len(ta_feat) == 0 or len(tb_feat) == 0:
            continue

        ta = ta_feat.iloc[0]
        tb = tb_feat.iloc[0]

        # 队伍差值特征 (按照V1设计)
        f = {}

        f['battle_id'] = bid
        f['league_id'] = league_id
        f['team_a_id'] = int(game['team_a_id'])
        f['team_b_id'] = int(game['team_b_id'])
        f['start_time'] = start_time

        # V1.1: 队伍强度差 (近30天胜率差)
        f['winrate_diff'] = ta['team_win_rate_30d'] - tb['team_win_rate_30d']

        # V1.2: 经济压制能力 (近30天场均经济差)
        f['gold_diff'] = ta['team_avg_gold_30d'] - tb['team_avg_gold_30d']

        # V1.3: 资源控制能力
        f['tower_diff'] = ta['team_avg_tower_30d'] - tb['team_avg_tower_30d']
        f['big_dragon_diff'] = ta['team_avg_big_dragon_30d'] - tb['team_avg_big_dragon_30d']
        f['tyrant_diff'] = ta['team_tyrant_cnt_30d'] - tb['team_tyrant_cnt_30d']

        # KDA差
        f['kda_diff_30d'] = ta['team_avg_kda_30d'] - tb['team_avg_kda_30d']
        f['kda_diff_all'] = ta['team_avg_kda_all'] - tb['team_avg_kda_all']

        # 全历史胜率差
        f['winrate_diff_all'] = ta['team_win_rate_all'] - tb['team_win_rate_all']

        # 经验差 (比赛场数)
        f['games_diff'] = ta['team_games_all'] - tb['team_games_all']

        # V1.4: 选手对位差 (按位置汇总)
        pos_map = {6: 'top', 5: 'jungle', 2: 'mid', 7: 'adc', 4: 'support'}
        for pos, name in pos_map.items():
            pa = player_feat[(player_feat['battle_id'] == bid) &
                             (player_feat['team_id'] == game['team_a_id']) &
                             (player_feat['player_name'].isin(
                                 player_stats[player_stats['battle_id'] == bid]['player_name']))]
            pb = player_feat[(player_feat['battle_id'] == bid) &
                             (player_feat['team_id'] == game['team_b_id']) &
                             (player_feat['player_name'].isin(
                                 player_stats[player_stats['battle_id'] == bid]['player_name']))]

            # 从player_stats找该场次该位置的选手
            pa_pos = player_stats[(player_stats['battle_id'] == bid) &
                                  (player_stats['team_id'] == game['team_a_id']) &
                                  (player_stats['position'] == pos)]
            pb_pos = player_stats[(player_stats['battle_id'] == bid) &
                                  (player_stats['team_id'] == game['team_b_id']) &
                                  (player_stats['position'] == pos)]

            if len(pa_pos) > 0:
                pn_a = pa_pos.iloc[0]['player_name']
                pfa = player_feat[(player_feat['battle_id'] == bid) &
                                  (player_feat['player_name'] == pn_a)]
                if len(pfa) > 0:
                    pfa = pfa.iloc[0]
                    f[f'{name}_kda_diff'] = pfa['player_avg_kda_30d']
                    f[f'{name}_mvp_diff'] = pfa['player_avg_mvp_30d']
                else:
                    f[f'{name}_kda_diff'] = 1.0
                    f[f'{name}_mvp_diff'] = 5.0
            else:
                f[f'{name}_kda_diff'] = 1.0
                f[f'{name}_mvp_diff'] = 5.0

            if len(pb_pos) > 0:
                pn_b = pb_pos.iloc[0]['player_name']
                pfb = player_feat[(player_feat['battle_id'] == bid) &
                                  (player_feat['player_name'] == pn_b)]
                if len(pfb) > 0:
                    pfb = pfb.iloc[0]
                    f[f'{name}_kda_diff'] -= pfb['player_avg_kda_30d']
                    f[f'{name}_mvp_diff'] -= pfb['player_avg_mvp_30d']
                else:
                    f[f'{name}_kda_diff'] -= 1.0
                    f[f'{name}_mvp_diff'] -= 5.0

        # Label: A队是否获胜
        gts_a = team_stats[(team_stats['battle_id'] == bid) &
                           (team_stats['team_id'] == game['team_a_id'])]
        if len(gts_a) > 0:
            f['label'] = int(gts_a.iloc[0]['is_win'])
        else:
            continue

        samples.append(f)

    df = pd.DataFrame(samples)
    return df


# ============================================================
# 3. 训练模型
# ============================================================

def train_model(feature_df):
    # 按时序分割: 前5赛季训练, 最新赛季测试
    # league_id 顺序: 20250002 -> 20250003 -> 20250004 -> 20260001 -> 20260002 -> 20260003
    train_leagues = ['20250002', '20250003', '20250004', '20260001', '20260002']
    test_leagues = ['20260003']

    train_df = feature_df[feature_df['league_id'].isin(train_leagues)]
    test_df = feature_df[feature_df['league_id'].isin(test_leagues)]

    print(f"\n训练集: {len(train_df)} 小局 (赛季 {train_leagues})")
    print(f"测试集: {len(test_df)} 小局 (赛季 {test_leagues})")

    # 排除非特征列
    exclude = ['battle_id', 'league_id', 'team_a_id', 'team_b_id', 'start_time', 'label']
    feature_cols = [c for c in feature_df.columns if c not in exclude]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['label']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['label']

    print(f"特征维度: {len(feature_cols)}")
    print(f"训练集正样本率: {y_train.mean():.3f}")
    print(f"测试集正样本率: {y_test.mean():.3f}")

    # 类别平衡处理
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # LightGBM 模型
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )

    # 用时间序列交叉验证选最佳迭代次数
    print("\n时间序列交叉验证...")
    tscv = TimeSeriesSplit(n_splits=3)
    train_sorted = train_df.sort_values('start_time')
    X_cv = train_sorted[feature_cols].fillna(0)
    y_cv = train_sorted['label']

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        cv_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
        print(f"  Fold {fold+1}: Accuracy={acc:.4f}, AUC={auc:.4f}")

    # 最终在全量训练集上训练
    print("\n最终训练...")
    model.fit(X_train, y_train)

    # 测试集评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== 测试集结果 (赛季 {test_leagues}) ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC:      {test_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['B队胜', 'A队胜'])}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 特征重要性:")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # 预测分布
    test_df = test_df.copy()
    test_df['pred_proba'] = y_proba
    test_df['pred_label'] = y_pred
    test_df['correct'] = (test_df['label'] == y_pred)

    # 按置信度分档看准确率
    test_df['confidence_bin'] = pd.cut(test_df['pred_proba'], bins=[0, 0.4, 0.45, 0.55, 0.6, 1.0])
    print("\n置信度分档准确率:")
    for bin_val, grp in test_df.groupby('confidence_bin', observed=False):
        if len(grp) > 0:
            print(f"  {bin_val}: {len(grp)}局, 准确率={grp['correct'].mean():.3f}")

    return model, importance, test_df


def main():
    print("=" * 60)
    print("KPL V1 胜负预测模型")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    games, team_stats, player_stats, teams = load_data()
    print(f"  {len(games)} 小局, {len(team_stats)} 队伍面板, "
          f"{len(player_stats)} 选手记录")

    # 特征工程
    print("\n[2/4] 特征工程...")
    feature_df = build_dataset(games, team_stats, player_stats, teams)
    print(f"  有效样本: {len(feature_df)}")

    if len(feature_df) == 0:
        print("ERROR: 没有有效样本, 检查数据完整性")
        return

    # 保存特征
    feature_df.to_csv('e:/gediandata/kpl/features_v1.csv', index=False, encoding='utf-8-sig')
    print("  特征已保存到 features_v1.csv")

    # 训练
    print("\n[3/4] 训练模型...")
    model, importance, predictions = train_model(feature_df)

    # 保存结果
    predictions.to_csv('e:/gediandata/kpl/predictions_v1.csv', index=False, encoding='utf-8-sig')

    # 保存模型
    model.booster_.save_model('e:/gediandata/kpl/v1_model.txt')
    importance.to_csv('e:/gediandata/kpl/feature_importance.csv', index=False, encoding='utf-8-sig')

    print("\n[4/4] 模型已保存:")
    print("  v1_model.txt           — 模型文件")
    print("  features_v1.csv        — 特征数据")
    print("  predictions_v1.csv     — 预测结果")
    print("  feature_importance.csv — 特征重要性")


if __name__ == "__main__":
    main()
