"""
KPL 数据分析面板 - Flask 后端
BP模拟 / 选手分析 / 战队分析 / 英雄分析 / 选手对比 / 队伍H2H
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, jsonify, request, render_template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import query_df, query_one, query_all

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ============================================================
# 加载 BP 预测模型
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bp_model = None
bp_features = None
try:
    bp_model = lgb.Booster(model_file=os.path.join(BASE_DIR, 'bp_model.txt'))
    with open(os.path.join(BASE_DIR, 'bp_model_features.json'), 'r') as f:
        bp_features = json.load(f)
    print(f"[OK] BP模型已加载: {len(bp_features)}维特征")
except Exception as e:
    print(f"[WARN] BP模型加载失败, 使用回退方案: {e}")


# ============================================================
# 工具函数
# ============================================================

def clamp_winrate(val):
    """修正数据库中的协同胜率异常值(>1.0)"""
    if val is None:
        return None
    return round(min(max(float(val), 0.0), 1.0), 3)


# ============================================================
# API: 基础数据
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/heroes")
def api_heroes():
    df = query_df("SELECT hero_id, hero_name FROM heroes ORDER BY hero_name")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/teams")
def api_teams():
    df = query_df("SELECT team_id, team_name, team_abbr FROM teams ORDER BY team_name")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/players")
def api_players():
    league = request.args.get("league_id", "20260003")
    df = query_df("""
        SELECT DISTINCT p.player_name, p.full_name, p.player_icon
        FROM game_player_stats gps
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN players p ON gps.player_name = p.player_name
        WHERE g.match_id IN (SELECT match_id FROM matches WHERE league_id = %s)
          AND gps.player_name IS NOT NULL
        ORDER BY p.player_name
    """, (league,))
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/leagues")
def api_leagues():
    df = query_df("SELECT league_id, league_name FROM leagues ORDER BY start_time DESC")
    return jsonify(df.to_dict(orient="records"))


# ============================================================
# API: BP预测 (LightGBM 模型)
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    ta = data.get("team_a", {})
    tb = data.get("team_b", {})
    a_heroes = ta.get("heroes", [])
    b_heroes = tb.get("heroes", [])
    a_team = ta.get("name", "")
    b_team = tb.get("name", "")

    if len(a_heroes) != 5 or len(b_heroes) != 5:
        return jsonify({"error": "每队需要5个英雄"}), 400

    features = compute_bp_features(a_heroes, b_heroes, a_team, b_team)

    # LightGBM 模型预测
    global bp_model, bp_features
    if bp_model is not None and bp_features is not None:
        X = pd.DataFrame([features])[bp_features].fillna(0.5)
        raw = float(bp_model.predict(X)[0])
        win_rate_a = 1.0 / (1.0 + np.exp(-raw))
    else:
        win_rate_a = fallback_predict(features)

    confidence = round(abs(win_rate_a - 0.5) * 2, 4)

    # 协同/克制详情
    synergy_a = get_synergy_details(a_heroes)
    synergy_b = get_synergy_details(b_heroes)
    counter_detail = get_counter_details(a_heroes, b_heroes)

    hero_names = {}
    all_hids = set(a_heroes + b_heroes)
    if all_hids:
        df_hn = query_df(
            f"SELECT hero_id, hero_name FROM heroes WHERE hero_id IN ({','.join(map(str, all_hids))})")
        hero_names = dict(zip(df_hn['hero_id'], df_hn['hero_name']))

    return jsonify({
        "prediction": {
            "win_rate_a": round(win_rate_a, 4),
            "confidence": confidence,
            "model": "LightGBM" if bp_model else "fallback",
        },
        "breakdown": {
            "英雄强度差": round(features.get('hero_wr_diff', 0), 3),
            "阵容协同差": round(features.get('synergy_diff', 0), 3),
            "平均克制值": round(features.get('counter_mean', 0), 3),
            "队伍实力差": round(features.get('team_wr_diff', 0), 3),
        },
        "heroes_a": [{"id": h, "name": hero_names.get(h, str(h))} for h in a_heroes],
        "heroes_b": [{"id": h, "name": hero_names.get(h, str(h))} for h in b_heroes],
        "synergy": {"team_a": synergy_a, "team_b": synergy_b},
        "counter": counter_detail,
    })


def fallback_predict(features):
    """模型加载失败时的简易回退预测"""
    score = 0.5
    score += features.get('hero_wr_diff', 0) * 1.5
    score += features.get('synergy_diff', 0) * 2.0
    score += (features.get('counter_mean', 0.5) - 0.5) * 1.5
    score += features.get('team_wr_diff', 0) * 1.5
    return max(0.05, min(0.95, score))


def compute_bp_features(a_heroes, b_heroes, a_team, b_team):
    feats = {}
    all_hids = set(a_heroes + b_heroes)

    # 英雄胜率
    if all_hids:
        hids_str = ','.join(map(str, all_hids))
        hr_rows = query_all(
            f"SELECT hero_id, win_rate FROM hero_stats WHERE hero_id IN ({hids_str}) ORDER BY id DESC")
        hr_map = {}
        for r in hr_rows:
            if r['hero_id'] not in hr_map:
                hr_map[r['hero_id']] = r['win_rate'] or 0.5
        a_hwrs = [hr_map.get(h, 0.5) for h in a_heroes]
        b_hwrs = [hr_map.get(h, 0.5) for h in b_heroes]
    else:
        a_hwrs = b_hwrs = [0.5]

    feats['hero_wr_avg_a'] = np.mean(a_hwrs)
    feats['hero_wr_avg_b'] = np.mean(b_hwrs)
    feats['hero_wr_diff'] = feats['hero_wr_avg_a'] - feats['hero_wr_avg_b']

    # 阵容协同
    syn_pairs = []
    for heroes in [a_heroes, b_heroes]:
        for i in range(5):
            for j in range(i + 1, 5):
                syn_pairs.append((min(heroes[i], heroes[j]), max(heroes[i], heroes[j])))

    syn_map = {}
    if syn_pairs:
        values = ','.join([f"({a},{b})" for a, b in set(syn_pairs)])
        syn_rows = query_all(
            f"SELECT hero_a_id, hero_b_id, win_rate FROM hero_synergy WHERE (hero_a_id, hero_b_id) IN ({values})")
        for r in syn_rows:
            syn_map[(r['hero_a_id'], r['hero_b_id'])] = min(r['win_rate'] or 0.5, 1.0)

    def calc_synergy(heroes):
        syns = []
        for i in range(5):
            for j in range(i + 1, 5):
                key = (min(heroes[i], heroes[j]), max(heroes[i], heroes[j]))
                syns.append(syn_map.get(key, 0.5))
        return syns

    syn_a = calc_synergy(a_heroes)
    syn_b = calc_synergy(b_heroes)
    feats['synergy_avg_a'] = np.mean(syn_a) if syn_a else 0.5
    feats['synergy_avg_b'] = np.mean(syn_b) if syn_b else 0.5
    feats['synergy_diff'] = feats['synergy_avg_a'] - feats['synergy_avg_b']

    # 克制
    ctr_pairs = [(ha, hb) for ha in a_heroes for hb in b_heroes]
    ctr_map = {}
    if ctr_pairs:
        values = ','.join([f"({a},{b})" for a, b in ctr_pairs])
        ctr_rows = query_all(
            f"SELECT hero_a_id, hero_b_id, win_rate_a FROM hero_counter WHERE (hero_a_id, hero_b_id) IN ({values})")
        for r in ctr_rows:
            ctr_map[(r['hero_a_id'], r['hero_b_id'])] = r['win_rate_a'] or 0.5

    ctrs = [ctr_map.get((ha, hb), 0.5) for ha in a_heroes for hb in b_heroes]
    feats['counter_mean'] = np.mean(ctrs) if ctrs else 0.5

    # 队伍实力
    team_names = [n for n in (a_team, b_team) if n]
    tw_map = {}
    if team_names:
        placeholders = ','.join(['%s'] * len(team_names))
        tw_rows = query_all(
            f"SELECT t.team_name, ts.win_rate FROM team_stats ts "
            f"JOIN teams t ON ts.team_id = t.team_id "
            f"WHERE t.team_name IN ({placeholders}) ORDER BY ts.id DESC", team_names)
        seen = set()
        for r in tw_rows:
            if r['team_name'] not in seen:
                tw_map[r['team_name']] = r['win_rate'] or 0.5
                seen.add(r['team_name'])

    feats['team_wr_a'] = tw_map.get(a_team, 0.5) if a_team else 0.5
    feats['team_wr_b'] = tw_map.get(b_team, 0.5) if b_team else 0.5
    feats['team_wr_diff'] = feats['team_wr_a'] - feats['team_wr_b']

    return feats


def get_synergy_details(hero_ids):
    pairs = []
    for i in range(len(hero_ids)):
        for j in range(i + 1, len(hero_ids)):
            a, b = (hero_ids[i], hero_ids[j]) if hero_ids[i] < hero_ids[j] else (hero_ids[j], hero_ids[i])
            r = query_one(
                "SELECT win_rate, games_together FROM hero_synergy WHERE hero_a_id=%s AND hero_b_id=%s", (a, b))
            pairs.append({
                "hero_a": hero_ids[i], "hero_b": hero_ids[j],
                "win_rate": clamp_winrate(r['win_rate']) if r else None,
                "games": r['games_together'] if r else 0,
            })
    return pairs


def get_counter_details(a_heroes, b_heroes):
    details = []
    for ha in a_heroes:
        row_c = []
        for hb in b_heroes:
            r = query_one(
                "SELECT win_rate_a, games_against FROM hero_counter WHERE hero_a_id=%s AND hero_b_id=%s", (ha, hb))
            row_c.append({
                "hero_a": ha, "hero_b": hb,
                "win_rate_a": clamp_winrate(r['win_rate_a']) if r else None,
                "games": r['games_against'] if r else 0,
            })
        details.append(row_c)
    return details


# ============================================================
# API: 选手分析
# ============================================================

@app.route("/api/player/<name>")
def api_player(name):
    pinfo = query_one("SELECT player_name, full_name, player_icon FROM players WHERE player_name=%s", (name,))
    if not pinfo:
        return jsonify({"error": "选手不存在"}), 404

    hero_pool = query_df("""
        SELECT p.hero_id, h.hero_name, p.games_played, p.wins, p.win_rate, p.avg_kda, p.avg_mvp_score
        FROM player_hero_stats p
        JOIN heroes h ON p.hero_id = h.hero_id
        WHERE p.player_name = %s
        ORDER BY p.games_played DESC LIMIT 20
    """, (name,))

    recent = query_df("""
        SELECT gps.battle_id, h.hero_name, gps.position_desc,
               gps.kills, gps.deaths, gps.assists, gps.kda,
               gps.mvp_score, gps.is_mvp,
               gps.gold, gps.hurt_to_hero, gps.be_hurt_by_hero,
               gps.participation_rate,
               m.start_time, gts.is_win
        FROM game_player_stats gps
        JOIN heroes h ON gps.hero_id = h.hero_id
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        JOIN game_team_stats gts ON gps.battle_id = gts.battle_id AND gps.team_id = gts.team_id
        WHERE gps.player_name = %s
        ORDER BY m.start_time DESC LIMIT 20
    """, (name,))

    season_stats = query_df("""
        SELECT l.league_name, COUNT(*) AS games,
               SUM(CASE WHEN gts.is_win THEN 1 ELSE 0 END) AS wins,
               ROUND(AVG(gps.kda), 2) AS avg_kda,
               ROUND(AVG(gps.mvp_score), 1) AS avg_mvp,
               ROUND(AVG(gps.participation_rate), 3) AS avg_part,
               ROUND(AVG(gps.gold), 0) AS avg_gold,
               ROUND(AVG(gps.hurt_to_hero), 0) AS avg_damage
        FROM game_player_stats gps
        JOIN game_team_stats gts ON gps.battle_id = gts.battle_id AND gps.team_id = gts.team_id
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        JOIN leagues l ON m.league_id = l.league_id
        WHERE gps.player_name = %s
        GROUP BY l.league_id, l.league_name
        ORDER BY l.start_time DESC
    """, (name,))

    pos_dist = query_df("""
        SELECT position_desc, COUNT(*) AS cnt
        FROM game_player_stats WHERE player_name = %s
        GROUP BY position_desc ORDER BY cnt DESC
    """, (name,))

    return jsonify({
        "info": pinfo,
        "hero_pool": hero_pool.to_dict(orient="records"),
        "recent_games": recent.to_dict(orient="records"),
        "season_stats": season_stats.to_dict(orient="records"),
        "position_dist": pos_dist.to_dict(orient="records"),
    })


# ============================================================
# API: 选手对比 (新增)
# ============================================================

@app.route("/api/compare")
def api_compare():
    p1 = request.args.get("p1", "")
    p2 = request.args.get("p2", "")
    if not p1 or not p2:
        return jsonify({"error": "请提供两个选手名"}), 400

    result = {"player_a": None, "player_b": None}
    for key, name in [("player_a", p1), ("player_b", p2)]:
        pinfo = query_one("SELECT player_name, full_name, player_icon FROM players WHERE player_name=%s", (name,))
        if not pinfo:
            result[key] = {"error": f"选手 {name} 不存在"}
            continue

        # 总体数据
        agg = query_one("""
            SELECT
                COUNT(*) AS total_games,
                SUM(CASE WHEN gts.is_win THEN 1 ELSE 0 END) AS wins,
                ROUND(AVG(gps.kda), 2) AS avg_kda,
                ROUND(AVG(gps.mvp_score), 1) AS avg_mvp,
                ROUND(AVG(gps.participation_rate), 3) AS avg_part,
                ROUND(AVG(gps.gold), 0) AS avg_gold,
                ROUND(AVG(gps.hurt_to_hero), 0) AS avg_damage,
                ROUND(AVG(gps.be_hurt_by_hero), 0) AS avg_tank
            FROM game_player_stats gps
            JOIN game_team_stats gts ON gps.battle_id = gts.battle_id AND gps.team_id = gts.team_id
            WHERE gps.player_name = %s
        """, (name,))

        result[key] = {
            "info": pinfo,
            "stats": agg,
        }

    return jsonify(result)


# ============================================================
# API: 队伍交手历史 H2H (新增)
# ============================================================

@app.route("/api/h2h")
def api_h2h():
    team_a = request.args.get("team_a", "")
    team_b = request.args.get("team_b", "")
    if not team_a or not team_b:
        return jsonify({"error": "请提供两个队伍ID"}), 400

    try:
        ta_id = int(team_a)
        tb_id = int(team_b)
    except ValueError:
        return jsonify({"error": "队伍ID必须是数字"}), 400

    # 所有交手记录
    records = query_all("""
        SELECT m.match_id, m.start_time, m.league_id, l.league_name,
               t_a.team_name AS team_a_name, t_b.team_name AS team_b_name,
               m.team_a_score, m.team_b_score, m.win_camp,
               CASE WHEN m.team_a_id = %s AND m.win_camp = 1 THEN 'A'
                    WHEN m.team_b_id = %s AND m.win_camp = 2 THEN 'B'
                    ELSE 'draw' END AS winner_side
        FROM matches m
        JOIN teams t_a ON m.team_a_id = t_a.team_id
        JOIN teams t_b ON m.team_b_id = t_b.team_id
        JOIN leagues l ON m.league_id = l.league_id
        WHERE ((m.team_a_id = %s AND m.team_b_id = %s)
            OR (m.team_a_id = %s AND m.team_b_id = %s))
          AND m.status = 2
        ORDER BY m.start_time DESC
    """, (ta_id, tb_id, ta_id, tb_id, tb_id, ta_id))

    a_name = query_one("SELECT team_name FROM teams WHERE team_id=%s", (ta_id,))
    b_name = query_one("SELECT team_name FROM teams WHERE team_id=%s", (tb_id,))
    a_name = a_name['team_name'] if a_name else str(ta_id)
    b_name = b_name['team_name'] if b_name else str(tb_id)

    a_wins = 0
    b_wins = 0
    for r in records:
        winner_name = r['team_a_name'] if r['winner_side'] == 'A' else r['team_b_name']
        if winner_name == a_name:
            a_wins += 1
        elif winner_name == b_name:
            b_wins += 1

    return jsonify({
        "team_a": {"id": ta_id, "name": a_name, "wins": a_wins},
        "team_b": {"id": tb_id, "name": b_name, "wins": b_wins},
        "total": len(records),
        "records": records,
    })


# ============================================================
# API: 战队分析
# ============================================================

@app.route("/api/team/<int:team_id>")
def api_team(team_id):
    tinfo = query_one("SELECT * FROM teams WHERE team_id=%s", (team_id,))
    if not tinfo:
        return jsonify({"error": "战队不存在"}), 404

    league = request.args.get("league_id", "20260003")

    stats = query_one("SELECT * FROM team_stats WHERE team_id=%s AND league_id=%s", (team_id, league))

    recent = query_df("""
        SELECT m.match_id, m.start_time,
               t_a.team_name AS team_a, t_b.team_name AS team_b,
               m.team_a_score, m.team_b_score, m.win_camp,
               CASE WHEN (m.team_a_id=%s AND m.win_camp=1) OR (m.team_b_id=%s AND m.win_camp=2)
                    THEN 1 ELSE 0 END AS our_win
        FROM matches m
        JOIN teams t_a ON m.team_a_id = t_a.team_id
        JOIN teams t_b ON m.team_b_id = t_b.team_id
        WHERE (m.team_a_id=%s OR m.team_b_id=%s) AND m.league_id=%s AND m.status=2
        ORDER BY m.start_time DESC LIMIT 20
    """, (team_id, team_id, team_id, team_id, league))

    players = query_df("""
        SELECT DISTINCT gps.player_name, gps.position_desc, COUNT(*) AS games
        FROM game_player_stats gps
        JOIN game_team_stats gts ON gps.battle_id = gts.battle_id AND gps.team_id = gts.team_id
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE gps.team_id = %s AND m.league_id = %s
        GROUP BY gps.player_name, gps.position_desc
        ORDER BY games DESC
    """, (team_id, league))

    # 常用英雄 (该队Pick最多的英雄)
    top_heroes = query_df("""
        SELECT h.hero_name, COUNT(*) AS pick_count,
               ROUND(AVG(CASE WHEN gts.is_win THEN 1 ELSE 0 END), 3) AS win_rate
        FROM game_player_stats gps
        JOIN game_team_stats gts ON gps.battle_id = gts.battle_id AND gps.team_id = gts.team_id
        JOIN heroes h ON gps.hero_id = h.hero_id
        JOIN games g ON gps.battle_id = g.battle_id
        JOIN matches m ON g.match_id = m.match_id
        WHERE gps.team_id = %s AND m.league_id = %s
        GROUP BY gps.hero_id, h.hero_name
        ORDER BY pick_count DESC LIMIT 10
    """, (team_id, league))

    return jsonify({
        "info": tinfo,
        "stats": stats,
        "recent": recent.to_dict(orient="records") if len(recent) > 0 else [],
        "players": players.to_dict(orient="records") if len(players) > 0 else [],
        "top_heroes": top_heroes.to_dict(orient="records") if len(top_heroes) > 0 else [],
    })


# ============================================================
# API: 英雄分析
# ============================================================

@app.route("/api/hero/<int:hero_id>")
def api_hero(hero_id):
    hinfo = query_one("SELECT * FROM heroes WHERE hero_id=%s", (hero_id,))
    if not hinfo:
        return jsonify({"error": "英雄不存在"}), 404

    synergy = query_df("""
        SELECT CASE WHEN s.hero_a_id=%s THEN s.hero_b_id ELSE s.hero_a_id END AS partner_id,
               h.hero_name, s.games_together, s.win_rate
        FROM hero_synergy s
        JOIN heroes h ON h.hero_id = CASE WHEN s.hero_a_id=%s THEN s.hero_b_id ELSE s.hero_a_id END
        WHERE (s.hero_a_id=%s OR s.hero_b_id=%s) AND s.games_together >= 3
        ORDER BY s.win_rate DESC LIMIT 10
    """, (hero_id, hero_id, hero_id, hero_id))

    counter_win = query_df("""
        SELECT h.hero_name, c.games_against, c.win_rate_a
        FROM hero_counter c
        JOIN heroes h ON c.hero_b_id = h.hero_id
        WHERE c.hero_a_id = %s AND c.games_against >= 3
        ORDER BY c.win_rate_a DESC LIMIT 10
    """, (hero_id,))

    counter_lose = query_df("""
        SELECT h.hero_name, c.games_against, c.win_rate_a
        FROM hero_counter c
        JOIN heroes h ON c.hero_a_id = h.hero_id
        WHERE c.hero_b_id = %s AND c.games_against >= 3
        ORDER BY c.win_rate_a ASC LIMIT 10
    """, (hero_id,))

    hero_stats = query_df("""
        SELECT l.league_name, hs.pick_count, hs.ban_count, hs.win_rate, hs.avg_kda
        FROM hero_stats hs
        JOIN leagues l ON hs.league_id = l.league_id
        WHERE hs.hero_id = %s
        ORDER BY l.start_time DESC LIMIT 6
    """, (hero_id,))

    return jsonify({
        "info": hinfo,
        "synergy": synergy.to_dict(orient="records"),
        "counter_win": counter_win.to_dict(orient="records"),
        "counter_lose": counter_lose.to_dict(orient="records"),
        "season_stats": hero_stats.to_dict(orient="records"),
    })


# ============================================================
# API: 赛季概览
# ============================================================

@app.route("/api/overview")
def api_overview():
    league = request.args.get("league_id", "20260003")

    ranking = query_df("""
        SELECT t.team_name, t.team_abbr,
               ts.win_rate, ts.battle_count, ts.wins, ts.losses,
               ts.avg_kda, ts.avg_gpm, ts.avg_first_blood_cnt,
               ts.avg_tyrant_control_rate, ts.avg_dragon_control_rate
        FROM team_stats ts
        JOIN teams t ON ts.team_id = t.team_id
        WHERE ts.league_id = %s
        ORDER BY ts.win_rate DESC
    """, (league,))

    hot_heroes = query_df("""
        SELECT h.hero_name, hs.pick_count, hs.ban_count, hs.win_rate
        FROM hero_stats hs
        JOIN heroes h ON hs.hero_id = h.hero_id
        WHERE hs.league_id = %s
        ORDER BY (hs.pick_count + hs.ban_count) DESC LIMIT 20
    """, (league,))

    return jsonify({
        "ranking": ranking.to_dict(orient="records"),
        "hot_heroes": hot_heroes.to_dict(orient="records"),
    })


if __name__ == "__main__":
    print("KPL 分析面板启动: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
