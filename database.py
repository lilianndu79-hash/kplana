"""
KPL 数据库连接模块
所有数据库操作统一入口
"""
import os
import sys
import pymysql
import pandas as pd
from decimal import Decimal
from dbutils.pooled_db import PooledDB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 优先本地配置，否则使用默认配置
try:
    from config import DB_CONFIG
except ImportError:
    DB_CONFIG = {}

try:
    from config_local import DB_CONFIG as LOCAL_CONFIG
    DB_CONFIG.update(LOCAL_CONFIG)
except ImportError:
    pass

# 环境变量覆盖
for key in ('host', 'port', 'user', 'password', 'database'):
    env_key = f'KPL_DB_{key.upper()}'
    if os.getenv(env_key):
        DB_CONFIG[key] = os.getenv(env_key) if key != 'port' else int(os.getenv(env_key))

DB_CONFIG.setdefault('charset', 'utf8mb4')

_pool = None


def get_pool():
    global _pool
    if _pool is None:
        _pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            mincached=3,
            maxcached=5,
            blocking=True,
            **DB_CONFIG
        )
    return _pool


def query_df(sql, params=None):
    """返回 DataFrame"""
    conn = get_pool().connection()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def query_one(sql, params=None):
    """返回单行 dict"""
    conn = get_pool().connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(sql, params)
            row = c.fetchone()
        if row:
            for k, v in row.items():
                if isinstance(v, Decimal):
                    row[k] = float(v)
        return row
    finally:
        conn.close()


def query_all(sql, params=None):
    """返回多行 dict 列表"""
    conn = get_pool().connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(sql, params)
            rows = c.fetchall()
        for r in rows:
            for k, v in r.items():
                if isinstance(v, Decimal):
                    r[k] = float(v)
        return rows
    finally:
        conn.close()
