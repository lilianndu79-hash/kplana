"""
KPL 数据导入脚本
将 raw_data/ 目录下的JSON文件导入MySQL kpl数据库
"""

import json
import os
import glob
import pymysql
import time

# === 数据库连接 ===
DB = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "auba7956",
    "database": "kpl",
    "charset": "utf8mb4",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw_data")


def get_conn():
    return pymysql.connect(**DB)


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all(pattern):
    """加载所有匹配的JSON文件，合并results/data数组"""
    all_data = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results") or data.get("data") or []
            if isinstance(items, list):
                all_data.extend(items)
            elif isinstance(items, dict):
                all_data.append(items)
    return all_data


def import_leagues(conn):
    """从 matches JSON 中提取赛季信息，同时补全名称"""
    print("导入 leagues...", end=" ")
    cursor = conn.cursor()
    seen = set()
    count = 0

    # 已知赛季的补充信息 (league_id -> name)
    league_names = {
        "20250002": "2025年KPL夏季赛", "20250003": "2025王者荣耀年度总决赛",
        "20250004": "2025年挑战者杯", "20260001": "2026年KPL春季赛",
        "20260002": "2026年挑战者杯", "20260003": "2026年KPL夏季赛",
    }

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "matches_*.json"))):
        fname = os.path.basename(path)
        league_id = fname.replace("matches_", "").replace(".json", "")

        if league_id in seen:
            continue
        seen.add(league_id)

        # matches JSON直接是列表
        matches = load_json(fname)
        if isinstance(matches, dict):
            matches = matches.get("results") or matches.get("data") or []

        start_time = matches[0].get("start_time") if matches else None
        end_time = matches[-1].get("start_time") if matches else None
        name = league_names.get(league_id, "")

        cursor.execute(
            """INSERT IGNORE INTO leagues (league_id, league_name, league_type,
            start_time, end_time, status, cc_league_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (league_id, name, "kpl", start_time, end_time, 2,
             matches[0].get("cc_match_id", "") if matches else "")
        )
        count += 1

    conn.commit()
    print(f"{count} 条")


def import_teams(conn):
    """从 matches JSON 提取所有队伍"""
    print("导入 teams...", end=" ")
    cursor = conn.cursor()
    seen = set()
    count = 0

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "matches_*.json"))):
        matches = load_json(os.path.basename(path))
        if isinstance(matches, dict):
            matches = matches.get("results") or matches.get("data") or []

        for match in matches:
            if not isinstance(match, dict):
                continue
            for camp_key in ("camp1", "camp2"):
                camp = match.get(camp_key) or {}
                tid = camp.get("team_id")
                if tid and tid not in seen:
                    seen.add(tid)
                    cursor.execute(
                        """INSERT IGNORE INTO teams (team_id, team_name, team_abbr, team_icon)
                        VALUES (%s, %s, %s, %s)""",
                        (tid, camp.get("team_name", ""),
                         camp.get("team_abbreviation", ""),
                         camp.get("team_icon", ""))
                    )
                    count += 1

    conn.commit()
    print(f"{count} 条")


def import_heroes(conn):
    """从 battle JSON 提取英雄"""
    print("导入 heroes...", end=" ")
    cursor = conn.cursor()
    seen = set()
    count = 0

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "battle_*.json"))):
        data = load_json(os.path.basename(path))
        detail = data.get("data") or data

        # 从 bp_list 提取
        for bp in detail.get("bp_list", []):
            hid = bp.get("hero_id")
            if hid and hid not in seen:
                seen.add(hid)
                cursor.execute(
                    "INSERT IGNORE INTO heroes (hero_id, hero_name, hero_icon) VALUES (%s, %s, %s)",
                    (hid, bp.get("hero_name", ""), bp.get("hero_icon", ""))
                )
                count += 1

        # 从 battle_player_list 提取
        for pl in detail.get("battle_player_list", []):
            hid = pl.get("hero_id")
            if hid and hid not in seen:
                seen.add(hid)
                cursor.execute(
                    "INSERT IGNORE INTO heroes (hero_id, hero_name, hero_icon) VALUES (%s, %s, %s)",
                    (hid, pl.get("hero_name", ""), pl.get("hero_icon", ""))
                )
                count += 1

    conn.commit()
    print(f"{count} 条")


def import_players(conn):
    """从 battle JSON 提取选手"""
    print("导入 players...", end=" ")
    cursor = conn.cursor()
    seen = set()
    count = 0

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "battle_*.json"))):
        data = load_json(os.path.basename(path))
        detail = data.get("data") or data
        for pl in detail.get("battle_player_list", []):
            pn = pl.get("player_name", "")
            if pn and pn not in seen:
                seen.add(pn)
                cursor.execute(
                    """INSERT IGNORE INTO players (player_name, full_name, player_icon)
                    VALUES (%s, %s, %s)""",
                    (pn, pl.get("actual_player_name", ""), pl.get("player_icon", ""))
                )
                count += 1

    conn.commit()
    print(f"{count} 条")


def import_matches(conn):
    """导入大场"""
    print("导入 matches...", end=" ")
    cursor = conn.cursor()
    count = 0

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "matches_*.json"))):
        matches = load_json(os.path.basename(path))
        # matches JSON直接是列表
        if isinstance(matches, dict):
            matches = matches.get("results") or matches.get("data") or []

        for match in matches:
            if not isinstance(match, dict):
                continue
            cursor.execute(
                """INSERT IGNORE INTO matches
                (match_id, league_id, bo, status, win_camp, start_time, end_time,
                 match_address, stage_name, stage_desc, cc_match_id,
                 team_a_id, team_b_id, team_a_score, team_b_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (match.get("match_id"), match.get("league_id"), match.get("bo"),
                 match.get("status"), match.get("win_camp"),
                 match.get("start_time"), match.get("end_time"),
                 match.get("match_address"), match.get("match_stage_name"),
                 match.get("match_stage_desc"), match.get("cc_match_id"),
                 (match.get("camp1") or {}).get("team_id"),
                 (match.get("camp2") or {}).get("team_id"),
                 (match.get("camp1") or {}).get("score"),
                 (match.get("camp2") or {}).get("score"))
            )
            count += 1

    conn.commit()
    print(f"{count} 条")


def import_games_and_battle_data(conn):
    """导入小局 + 队伍面板 + 选手数据 + BP (从每个 battle JSON)"""
    cursor = conn.cursor()

    battle_files = sorted(glob.glob(os.path.join(DATA_DIR, "battle_*.json")))
    total = len(battle_files)
    print(f"导入 battle 数据 ({total} 局)...")

    game_count = 0
    team_count = 0
    player_count = 0
    bp_count = 0

    # 用内存缓存，每100局批量提交
    game_rows = []
    team_rows = []
    player_rows = []
    bp_rows = []

    def flush():
        nonlocal game_count, team_count, player_count, bp_count
        if game_rows:
            cursor.executemany(
                """INSERT IGNORE INTO games (battle_id, match_id, battle_seq, status,
                win_camp, game_duration, video_id, video_url) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                game_rows)
            game_count += len(game_rows)
            game_rows.clear()
        if team_rows:
            cursor.executemany(
                """INSERT IGNORE INTO game_team_stats
                (battle_id, camp, team_id, is_win, kills, deaths, assists, kda, gold,
                 tower_kills, tyrant_kills, dark_tyrant_kills, prophet_dragon_kills,
                 shadow_dragon_kills, storm_dragon_king_kills, big_dragon_kills)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                team_rows)
            team_count += len(team_rows)
            team_rows.clear()
        if player_rows:
            cursor.executemany(
                """INSERT IGNORE INTO game_player_stats
                (battle_id, team_id, camp, player_name, actual_player_name, hero_id,
                 position, position_desc, is_mvp, is_lose_mvp, mvp_score, kills, deaths,
                 assists, kda, participation_rate, gold, hurt_total, hurt_to_hero,
                 hurt_total_rate, hurt_to_hero_rate, be_hurt_total, be_hurt_by_hero,
                 be_hurt_total_rate, be_hurt_by_hero_rate, summoner_ability_id,
                 symbol_ids, equip_ids)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                player_rows)
            player_count += len(player_rows)
            player_rows.clear()
        if bp_rows:
            cursor.executemany(
                """INSERT IGNORE INTO draft_picks (battle_id, camp, is_pick, position, hero_id)
                VALUES (%s,%s,%s,%s,%s)""",
                bp_rows)
            bp_count += len(bp_rows)
            bp_rows.clear()
        conn.commit()

    for i, path in enumerate(battle_files):
        data = load_json(os.path.basename(path))
        detail = data.get("data") or data
        bid = detail.get("battle_id", "")
        if not bid:
            continue

        # --- game ---
        # 从文件名反推 match_id (battle_id格式: xxx_N_xxx, match在matches表里)
        game_rows.append((
            bid,
            "",  # match_id 稍后update关联
            detail.get("battle_seq"), detail.get("status"),
            detail.get("win_camp"), detail.get("game_duration"),
            (detail.get("video_list") or [{}])[0].get("video_id", ""),
            (detail.get("video_list") or [{}])[0].get("video_url", "")
        ))

        # --- game_team_stats ---
        for camp_key, camp_num in [("camp1", 1), ("camp2", 2)]:
            camp = detail.get(camp_key, {})
            if not camp:
                continue
            team_rows.append((
                bid, camp_num, camp.get("team_id"), camp.get("is_win"),
                camp.get("kill_num"), camp.get("death_num"), camp.get("assist_num"),
                camp.get("kda"), camp.get("gold"),
                camp.get("push_tower_num"), camp.get("kill_tyrant_num"),
                camp.get("kill_dark_tyrant_num"), camp.get("kill_prophet_dragon_num"),
                camp.get("kill_shadow_dragon_num"), camp.get("kill_storm_dragon_king_num"),
                camp.get("kill_big_dragon_num")
            ))

        # --- game_player_stats ---
        for pl in detail.get("battle_player_list", []):
            equip_ids = "+".join(str(e.get("equip_id", "")) for e in pl.get("BriefEquipList", []))
            player_rows.append((
                bid, pl.get("team_id"), pl.get("camp"),
                pl.get("player_name"), pl.get("actual_player_name"),
                pl.get("hero_id"), pl.get("position"), pl.get("position_desc"),
                bool(pl.get("is_mvp")), bool(pl.get("is_lose_mvp")),
                pl.get("mvp_score"), pl.get("kill_num"), pl.get("death_num"),
                pl.get("assist_num"), pl.get("kda"), pl.get("participation_rate"),
                pl.get("gold"), pl.get("hurt_total"), pl.get("hurt_to_hero_total"),
                pl.get("hurt_total_rate"), pl.get("hurt_to_hero_total_rate"),
                pl.get("be_hurt_total"), pl.get("be_hurt_by_hero_total"),
                pl.get("be_hurt_total_rate"), pl.get("be_hurt_by_hero_total_rate"),
                pl.get("SummonerAbilityInfo", {}).get("summoner_ability_id"),
                pl.get("symbol_ids", ""), equip_ids
            ))

        # --- draft_picks ---
        for bp in detail.get("bp_list", []):
            bp_rows.append((
                bid, bp.get("camp"), bool(bp.get("is_ban_or_pick")),
                bp.get("position"), bp.get("hero_id")
            ))

        if (i + 1) % 100 == 0:
            flush()
            print(f"  {i+1}/{total} 局...")

    flush()

    # 关联 match_id: 从 matches JSON 构建 battle_id → match_id 映射
    print("  关联 match_id...")
    battle_to_match = {}
    for mpath in sorted(glob.glob(os.path.join(DATA_DIR, "matches_*.json"))):
        matches = load_json(os.path.basename(mpath))
        if isinstance(matches, dict):
            matches = matches.get("results") or matches.get("data") or []
        for match in matches:
            if not isinstance(match, dict):
                continue
            mid = match.get("match_id")
            for bv in match.get("match_battle_video_list", []):
                bid = bv.get("battle_id")
                if bid and mid:
                    battle_to_match[bid] = mid

    # 逐条更新 (用内存映射比SQL JOIN更可靠)
    cursor.execute("SELECT battle_id FROM games WHERE match_id = ''")
    blank_games = [r[0] for r in cursor.fetchall()]
    updated = 0
    for bid in blank_games:
        mid = battle_to_match.get(bid, "")
        if mid:
            cursor.execute("UPDATE games SET match_id = %s WHERE battle_id = %s", (mid, bid))
            updated += 1
    conn.commit()
    print(f"  关联成功: {updated}/{len(blank_games)}")

    print(f"  games: {game_count}, team_stats: {team_count}, "
          f"players: {player_count}, bp: {bp_count}")


def import_team_stats(conn):
    """导入战队赛季统计"""
    print("导入 team_stats...", end=" ")
    cursor = conn.cursor()
    count = 0

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "team_stats_*.json"))):
        fname = os.path.basename(path)
        league_id = fname.replace("team_stats_", "").replace(".json", "")
        data = load_json(fname)
        items = data.get("data", [])

        for item in items:
            ti = item.get("team_info", {})
            si = item.get("statistics_info", {})
            if not ti.get("team_id"):
                continue

            cursor.execute(
                """INSERT IGNORE INTO team_stats
                (team_id, league_id, battle_count, wins, losses, win_rate,
                 avg_kills, avg_deaths, avg_assists, avg_kda, avg_gold,
                 avg_game_duration, avg_gpm, avg_first_blood_cnt,
                 avg_push_tower_num, avg_by_others_push_tower_num,
                 avg_tyrant_cnt, avg_dark_tyrant_cnt, avg_tyrant_control_rate,
                 avg_prophet_dragon_cnt, avg_shadow_dragon_cnt,
                 avg_dragon_control_rate, avg_storm_dragon_king_cnt,
                 avg_big_dragon_cnt, avg_hurt_total, avg_hurt_to_hero,
                 avg_be_hurt_total, avg_be_hurt_by_hero,
                 avg_per_min_hurt_total, avg_per_min_hurt_to_hero)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (ti.get("team_id"), league_id,
                 si.get("battle_count"), si.get("victory_battle_count"),
                 si.get("defeated_battle_count"), si.get("win_rate"),
                 si.get("avg_kill_num"), si.get("avg_death_num"),
                 si.get("avg_assist_num"), si.get("avg_kda"),
                 si.get("avg_gold"), si.get("avg_game_duration"),
                 si.get("avg_gpm"), si.get("avg_first_blood_cnt"),
                 si.get("avg_push_tower_num"),
                 si.get("avg_other_camp_push_tower_num"),
                 si.get("avg_kill_tyrant_num"),
                 si.get("avg_kill_dark_tyrant_num"),
                 si.get("avg_kill_all_tyrant_control_rate"),
                 si.get("avg_kill_prophet_dragon_num"),
                 si.get("avg_kill_shadow_dragon_num"),
                 si.get("avg_kill_all_dragon_control_rate"),
                 si.get("avg_kill_storm_dragon_king_num"),
                 si.get("avg_kill_big_dragon_num"),
                 si.get("avg_hurt_total"), si.get("avg_hurt_to_hero_total"),
                 si.get("avg_be_hurt_total"), si.get("avg_be_hurt_by_hero_total"),
                 si.get("avg_per_min_hurt_total"),
                 si.get("avg_per_min_hurt_to_hero_total"))
            )
            count += 1

    conn.commit()
    print(f"{count} 条")


def derive_hero_synergy(conn):
    """从比赛数据计算英雄协同"""
    print("计算 hero_synergy...", end=" ")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM hero_synergy")

    cursor.execute("""
        INSERT INTO hero_synergy (hero_a_id, hero_b_id, games_together, wins)
        SELECT
            LEAST(a.hero_id, b.hero_id),
            GREATEST(a.hero_id, b.hero_id),
            COUNT(DISTINCT a.battle_id),
            SUM(CASE WHEN gts.is_win THEN 1 ELSE 0 END)
        FROM game_player_stats a
        JOIN game_player_stats b ON a.battle_id = b.battle_id
            AND a.team_id = b.team_id AND a.hero_id != b.hero_id
        JOIN game_team_stats gts ON a.battle_id = gts.battle_id
            AND a.team_id = gts.team_id
        GROUP BY LEAST(a.hero_id, b.hero_id),
                 GREATEST(a.hero_id, b.hero_id)
        HAVING COUNT(DISTINCT a.battle_id) >= 3
    """)
    cursor.execute("""
        UPDATE hero_synergy SET win_rate = LEAST(ROUND(wins / 2.0 / games_together, 4), 1.0)
        WHERE games_together > 0
    """)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM hero_synergy")
    cnt = cursor.fetchone()[0]
    print(f"{cnt} 条")


def derive_hero_counter(conn):
    """从比赛数据计算英雄克制关系"""
    print("计算 hero_counter...", end=" ")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM hero_counter")

    cursor.execute("""
        INSERT INTO hero_counter (hero_a_id, hero_b_id, games_against, wins_a)
        SELECT
            a.hero_id,
            b.hero_id,
            COUNT(DISTINCT a.battle_id),
            SUM(CASE WHEN gts_a.is_win THEN 1 ELSE 0 END)
        FROM game_player_stats a
        JOIN game_player_stats b ON a.battle_id = b.battle_id
            AND a.team_id != b.team_id AND a.hero_id != b.hero_id
        JOIN game_team_stats gts_a ON a.battle_id = gts_a.battle_id
            AND a.team_id = gts_a.team_id
        GROUP BY a.hero_id, b.hero_id
        HAVING COUNT(DISTINCT a.battle_id) >= 3
    """)
    cursor.execute("""
        UPDATE hero_counter SET win_rate_a = ROUND(wins_a / games_against, 4)
        WHERE games_against > 0
    """)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM hero_counter")
    cnt = cursor.fetchone()[0]
    print(f"{cnt} 条")


def derive_player_hero_stats(conn):
    """从比赛数据计算选手英雄熟练度"""
    print("计算 player_hero_stats...", end=" ")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM player_hero_stats")

    cursor.execute("""
        INSERT INTO player_hero_stats (player_name, hero_id, games_played, wins,
                                        avg_kda, avg_mvp_score)
        SELECT
            gps.player_name,
            gps.hero_id,
            COUNT(*) AS games,
            SUM(CASE WHEN gts.is_win THEN 1 ELSE 0 END),
            ROUND(AVG(gps.kda), 2),
            ROUND(AVG(gps.mvp_score), 1)
        FROM game_player_stats gps
        JOIN game_team_stats gts ON gps.battle_id = gts.battle_id
            AND gps.team_id = gts.team_id
        GROUP BY gps.player_name, gps.hero_id
        HAVING games >= 2
    """)
    cursor.execute("""
        UPDATE player_hero_stats
        SET win_rate = ROUND(wins / games_played, 4)
        WHERE games_played > 0
    """)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM player_hero_stats")
    cnt = cursor.fetchone()[0]
    print(f"{cnt} 条")


def show_summary(conn):
    """显示导入结果"""
    cursor = conn.cursor()
    tables = ["leagues", "teams", "players", "heroes", "matches", "games",
              "game_team_stats", "game_player_stats", "draft_picks",
              "team_stats", "hero_synergy", "hero_counter", "player_hero_stats"]
    print("\n=== 数据导入摘要 ===")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cursor.fetchone()[0]
        print(f"  {t}: {cnt}")


def main():
    start = time.time()
    conn = get_conn()
    try:
        import_leagues(conn)
        import_teams(conn)
        import_heroes(conn)
        import_players(conn)
        import_matches(conn)
        import_games_and_battle_data(conn)
        import_team_stats(conn)
        import_heroes(conn)
        derive_hero_synergy(conn)
        derive_hero_counter(conn)
        derive_player_hero_stats(conn)
        show_summary(conn)
    finally:
        conn.close()
    print(f"\n总耗时: {(time.time()-start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
