"""
KPL 数据采集脚本
从 prod.comp.smoba.qq.com 拉取一年内的比赛数据
"""

import requests
import json
import os
import time
from datetime import datetime

# 强制实时输出，防止Windows缓冲导致看不到进度
import builtins as _builtins
_builtins._orig_print = _builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _builtins._orig_print(*args, **kwargs)
_builtins.print = _flush_print

BASE = "https://prod.comp.smoba.qq.com"

# 一年内的6个赛季 (2025.06 ~ 2026.06)
TARGET_LEAGUES = [
    "20250002",  # 2025 KPL夏季赛
    "20250003",  # 2025王者荣耀年度总决赛
    "20250004",  # 2025年挑战者杯
    "20260001",  # 2026 KPL春季赛
    "20260002",  # 2026年挑战者杯
    "20260003",  # 2026 KPL夏季赛(进行中)
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw_data")
os.makedirs(DATA_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://pvp.qq.com/",
})


def api_get(path, params=None):
    """统一请求，带重试和限速"""
    url = f"{BASE}{path}"
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 200:
                print(f"  API返回非200: {data.get('message')}")
                return None
            time.sleep(0.3)  # 限速
            return data
        except Exception as e:
            print(f"  请求失败(attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def save_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def collect():
    print(f"开始采集 {len(TARGET_LEAGUES)} 个赛季的数据\n")

    total_matches = 0
    total_battles = 0

    for league_id in TARGET_LEAGUES:
        print(f"--- 赛季 {league_id} ---")

        # 1. 获取大场列表
        resp = api_get("/leaguesite/matches/open", {"league_id": league_id})
        if not resp:
            print(f"  跳过赛季 {league_id}")
            continue
        matches = resp.get("results", [])
        save_json(f"matches_{league_id}.json", matches)
        print(f"  大场数: {len(matches)}")

        # 2. 获取战队统计
        team_stats = api_get("/leaguesite/league/team/settle_list/open",
                             {"league_id": league_id})
        if team_stats:
            save_json(f"team_stats_{league_id}.json", team_stats)

        # 3. 获取英雄统计
        hero_stats = api_get("/leaguesite/league/hero/settle_list/open",
                             {"league_id": league_id})
        if hero_stats:
            save_json(f"hero_stats_{league_id}.json", hero_stats)

        # 4. 逐个大场 → 小局列表 → 小局详情
        finished_count = 0
        for match in matches:
            mid = match["match_id"]
            if match["status"] == 0:
                continue

            resp = api_get("/leaguesite/match/battles/open",
                           {"league_id": league_id, "match_id": mid})
            if not resp:
                continue
            battles = resp.get("results", [])
            total_battles += len(battles)

            for battle in battles:
                bid = battle["battle_id"]
                if battle["status"] not in (1, 2):
                    continue

                detail = api_get("/leaguesite/battle/open", {"battle_id": bid})
                if detail:
                    save_json(f"battle_{bid}.json", detail)

            finished_count += 1
            total_matches += 1
            if finished_count % 5 == 0:
                print(f"  进度: {finished_count} 场大场已完成...")

            time.sleep(0.3)

        league_matches = sum(1 for m in matches if m["status"] != 0)
        print(f"  已完成 {league_matches} 场大场\n")

    print(f"\n采集完成: {total_matches} 大场, {total_battles} 小局")
    print(f"原始数据保存在: {DATA_DIR}")


if __name__ == "__main__":
    start = time.time()
    collect()
    elapsed = time.time() - start
    print(f"耗时: {elapsed/60:.1f} 分钟")
