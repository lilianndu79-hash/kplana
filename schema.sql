-- KPL 电竞数据分析系统 - 完整建表SQL
-- 基于 pvp.qq.com / prod.comp.smoba.qq.com 真实API

-- ============================================================
-- 模块1: 基础字典
-- ============================================================

CREATE TABLE IF NOT EXISTS leagues (
    league_id       VARCHAR(20) PRIMARY KEY,
    league_name     VARCHAR(100) NOT NULL,
    league_type     VARCHAR(50),
    season          TINYINT,
    year            SMALLINT,
    start_time      DATETIME,
    end_time        DATETIME,
    status          TINYINT DEFAULT 2,
    league_icon     VARCHAR(255),
    cc_league_id    VARCHAR(50)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS teams (
    team_id         INT PRIMARY KEY,
    team_name       VARCHAR(100) NOT NULL,
    team_abbr       VARCHAR(20),
    team_icon       VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS players (
    player_id       INT AUTO_INCREMENT PRIMARY KEY,
    player_name     VARCHAR(50) NOT NULL,
    full_name       VARCHAR(100),
    player_icon     VARCHAR(255),
    UNIQUE KEY uk_name_team (player_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS heroes (
    hero_id         INT PRIMARY KEY,
    hero_name       VARCHAR(50) NOT NULL,
    hero_icon       VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 模块2: 比赛核心
-- ============================================================

CREATE TABLE IF NOT EXISTS matches (
    match_id        VARCHAR(20) PRIMARY KEY,
    league_id       VARCHAR(20) NOT NULL,
    bo              TINYINT NOT NULL,
    status          TINYINT,
    win_camp        TINYINT,
    start_time      DATETIME,
    end_time        DATETIME,
    match_address   VARCHAR(50),
    stage_name      VARCHAR(50),
    stage_desc      VARCHAR(100),
    cc_match_id     VARCHAR(50),
    team_a_id       INT,
    team_b_id       INT,
    team_a_score    TINYINT,
    team_b_score    TINYINT,
    INDEX idx_league (league_id),
    INDEX idx_team_a (team_a_id),
    INDEX idx_team_b (team_b_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS games (
    battle_id       VARCHAR(50) PRIMARY KEY,
    match_id        VARCHAR(20) NOT NULL,
    battle_seq      TINYINT NOT NULL,
    status          TINYINT,
    win_camp        TINYINT,
    game_duration   INT,
    video_id        VARCHAR(50),
    video_url       VARCHAR(255),
    INDEX idx_match (match_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS game_team_stats (
    id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
    battle_id               VARCHAR(50) NOT NULL,
    camp                    TINYINT NOT NULL,
    team_id                 INT NOT NULL,
    is_win                  BOOLEAN,
    kills                   SMALLINT,
    deaths                  SMALLINT,
    assists                 SMALLINT,
    kda                     DECIMAL(8,4),
    gold                    INT,
    tower_kills             TINYINT,
    tyrant_kills            TINYINT,
    dark_tyrant_kills       TINYINT,
    prophet_dragon_kills    TINYINT,
    shadow_dragon_kills     TINYINT,
    storm_dragon_king_kills TINYINT,
    big_dragon_kills        TINYINT,
    UNIQUE KEY uk_battle_camp (battle_id, camp),
    INDEX idx_team (team_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS game_player_stats (
    id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
    battle_id               VARCHAR(50) NOT NULL,
    team_id                 INT NOT NULL,
    camp                    TINYINT,
    player_name             VARCHAR(50),
    actual_player_name      VARCHAR(100),
    hero_id                 INT NOT NULL,
    position                TINYINT,
    position_desc           VARCHAR(10),
    is_mvp                  BOOLEAN,
    is_lose_mvp             BOOLEAN,
    mvp_score               DECIMAL(4,1),
    kills                   SMALLINT,
    deaths                  SMALLINT,
    assists                 SMALLINT,
    kda                     DECIMAL(8,4),
    participation_rate      DECIMAL(6,4),
    gold                    INT,
    hurt_total              INT,
    hurt_to_hero            INT,
    hurt_total_rate         DECIMAL(6,4),
    hurt_to_hero_rate       DECIMAL(6,4),
    be_hurt_total           INT,
    be_hurt_by_hero         INT,
    be_hurt_total_rate      DECIMAL(6,4),
    be_hurt_by_hero_rate    DECIMAL(6,4),
    summoner_ability_id     INT,
    symbol_ids              VARCHAR(500),
    equip_ids               VARCHAR(500),
    UNIQUE KEY uk_battle_player_hero (battle_id, team_id, hero_id),
    INDEX idx_player (player_name),
    INDEX idx_hero (hero_id),
    INDEX idx_battle (battle_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 模块3: BP记录
-- ============================================================

CREATE TABLE IF NOT EXISTS draft_picks (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    battle_id       VARCHAR(50) NOT NULL,
    camp            TINYINT NOT NULL,
    is_pick         BOOLEAN NOT NULL,
    position        TINYINT,
    hero_id         INT NOT NULL,
    INDEX idx_battle (battle_id),
    INDEX idx_hero (hero_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 模块4: 战队赛季统计
-- ============================================================

CREATE TABLE IF NOT EXISTS team_stats (
    id                              BIGINT AUTO_INCREMENT PRIMARY KEY,
    team_id                         INT NOT NULL,
    league_id                       VARCHAR(20) NOT NULL,
    battle_count                    INT,
    wins                            INT,
    losses                          INT,
    win_rate                        DECIMAL(5,4),
    avg_kills                       DECIMAL(6,2),
    avg_deaths                      DECIMAL(6,2),
    avg_assists                     DECIMAL(6,2),
    avg_kda                         DECIMAL(8,4),
    avg_gold                        DECIMAL(12,2),
    avg_game_duration               DECIMAL(12,2),
    avg_gpm                         DECIMAL(8,2),
    avg_first_blood_cnt             DECIMAL(5,4),
    avg_push_tower_num              DECIMAL(5,2),
    avg_by_others_push_tower_num    DECIMAL(5,2),
    avg_tyrant_cnt                  DECIMAL(5,2),
    avg_dark_tyrant_cnt             DECIMAL(5,2),
    avg_tyrant_control_rate         DECIMAL(5,4),
    avg_prophet_dragon_cnt          DECIMAL(5,2),
    avg_shadow_dragon_cnt           DECIMAL(5,2),
    avg_dragon_control_rate         DECIMAL(5,4),
    avg_storm_dragon_king_cnt       DECIMAL(5,2),
    avg_big_dragon_cnt              DECIMAL(5,2),
    avg_hurt_total                  DECIMAL(14,2),
    avg_hurt_to_hero                DECIMAL(14,2),
    avg_be_hurt_total               DECIMAL(14,2),
    avg_be_hurt_by_hero             DECIMAL(14,2),
    avg_per_min_hurt_total          DECIMAL(12,2),
    avg_per_min_hurt_to_hero        DECIMAL(12,2),
    UNIQUE KEY uk_team_league (team_id, league_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 模块5: V2英雄图谱 (衍生计算)
-- ============================================================

CREATE TABLE IF NOT EXISTS hero_synergy (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    league_id       VARCHAR(20),
    hero_a_id       INT NOT NULL,
    hero_b_id       INT NOT NULL,
    games_together  INT DEFAULT 0,
    wins            INT DEFAULT 0,
    win_rate        DECIMAL(5,4),
    UNIQUE KEY uk_pair (hero_a_id, hero_b_id, league_id),
    CHECK (hero_a_id < hero_b_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS hero_counter (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    league_id       VARCHAR(20),
    hero_a_id       INT NOT NULL,
    hero_b_id       INT NOT NULL,
    games_against   INT DEFAULT 0,
    wins_a          INT DEFAULT 0,
    win_rate_a      DECIMAL(5,4),
    UNIQUE KEY uk_matchup (hero_a_id, hero_b_id, league_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS player_hero_stats (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    player_name     VARCHAR(50) NOT NULL,
    hero_id         INT NOT NULL,
    games_played    INT DEFAULT 0,
    wins            INT DEFAULT 0,
    win_rate        DECIMAL(5,4),
    avg_kda         DECIMAL(6,2),
    avg_mvp_score   DECIMAL(4,1),
    UNIQUE KEY uk_player_hero (player_name, hero_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
