"""评测环境自检 —— 每次回归前先跑一遍，避免"跑完 79 条才发现环境不对"

    python eval/tools/check_env.py

检查五项：
  1. 模型配置与 API key 可用性（base/tuned 切换靠环境变量，这里确认当前生效的是哪个）
  2. 数据库连通性 + 实际用的是只读账号还是主账号
  3. **只读账号是不是真的只读**（实测写操作是否被拒）
  4. 房源库 schema 与行数（黄金结果集依赖它）
  5. sql_guard 安全层自检（放行/拒绝各一条）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "[OK]  ", "[FAIL]", "[WARN]"


def check_model() -> bool:
    from src.agent.common.llm import get_model_config, model

    cfg = get_model_config()
    print(f"  配置：profile={cfg['profile'] or '(未设)'} model={cfg['model']} "
          f"base_url={cfg['base_url'] or '(DeepSeek 官方)'} temperature={cfg['temperature']}")
    try:
        resp = model.invoke("回复两个字：可用")
        print(f"{OK} 模型调用成功：{str(resp.content)[:20]}")
        return True
    except Exception as exc:
        print(f"{BAD} 模型调用失败：{type(exc).__name__}: {str(exc)[:200]}")
        print("       → 检查 .env 里的 DEEPSEEK_API_KEY；"
              "注意机器级环境变量会盖住 .env（本项目已改用 override=True）")
        return False


def check_db_account() -> bool:
    """确认连的是哪个账号——评测链路应当走只读账号"""
    import os

    from eval.runner.db import connect

    if os.getenv("DB_RO_USER"):
        print(f"{OK} 已配置 DB_RO_USER={os.getenv('DB_RO_USER')}，评测链路走只读账号")
        ro = True
    else:
        print(f"{WARN} 未配置 DB_RO_USER，评测链路会用主账号（可写）——建议补上")
        ro = False
    try:
        conn = connect()
        conn.close()
        print(f"{OK} 数据库可连接：{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
        return ro
    except Exception as exc:
        print(f"{BAD} 数据库连接失败：{type(exc).__name__}: {str(exc)[:200]}")
        return False


def check_readonly_enforced() -> bool:
    """只读账号必须真的只读——写操作要被 DB 权限拦住（M0 的纵深防御）"""
    import os

    import pymysql

    user = os.getenv("DB_RO_USER")
    if not user:
        print(f"{WARN} 无只读账号，跳过「只读是否真的生效」检查")
        return False

    conf = dict(
        host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", "3306")),
        user=user, password=os.getenv("DB_RO_PASSWORD") or os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"), charset="utf8mb4", connect_timeout=10,
    )
    # 用 WHERE 匹配不到任何行的写语句探测：权限检查在写入前发生，
    # 即便权限意外放行也不会改动任何数据
    probes = [
        ("UPDATE", "UPDATE house SET price=price WHERE id=-1"),
        ("CREATE TEMPORARY TABLE", "CREATE TEMPORARY TABLE _probe (id int)"),
    ]
    all_denied = True
    for label, sql in probes:
        try:
            conn = pymysql.connect(**conf)
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.close()
            print(f"{BAD} {label} 竟然成功了 —— 只读账号权限配错，写操作没被拦住！")
            all_denied = False
        except pymysql.MySQLError as exc:
            msg = exc.args[1] if len(exc.args) > 1 else str(exc)
            print(f"{OK} {label} 被拒（符合预期）：{str(msg)[:60]}")
    return all_denied


def check_schema() -> bool:
    from eval.runner.db import run_sql

    try:
        cols, rows = run_sql("SELECT COUNT(*) FROM house")
        print(f"{OK} 房源库可查：house 表 {rows[0][0]} 行")
        cols, rows = run_sql("SELECT id, name, price FROM house LIMIT 1")
        print(f"      列名抽样：{cols}")
        return True
    except Exception as exc:
        print(f"{BAD} 房源库查询失败：{type(exc).__name__}: {str(exc)[:200]}")
        return False


def check_guard() -> bool:
    from src.agent.common.sql_guard import REJECT_MARKER, validate_sql
    from src.agent.node.recommend import _get_query

    allowed = validate_sql("SELECT id FROM house ORDER BY price LIMIT 3")
    rejected = validate_sql("DELETE FROM house")
    ok = allowed.allowed and not rejected.allowed
    print(f"{OK if ok else BAD} sql_guard 自检：放行 SELECT={allowed.allowed}，拒绝 DELETE={not rejected.allowed}")

    if _get_query is None:
        print(f"{BAD} 安全层没有挂到 sql_db_query 上（_get_query is None）——DB 不可用？")
        return False
    print(f"{OK} 安全层已挂载：工具名={_get_query.name}，拒绝标记={REJECT_MARKER}")
    return ok


def main() -> int:
    print("=" * 66)
    print("Roomie 评测环境自检")
    print("=" * 66)

    results = []
    for title, fn in [
        ("1. 模型接入", check_model),
        ("2. 数据库账号", check_db_account),
        ("3. 只读权限是否真的生效", check_readonly_enforced),
        ("4. 房源库 schema", check_schema),
        ("5. 安全层挂载", check_guard),
    ]:
        print(f"\n--- {title} ---")
        try:
            results.append(bool(fn()))
        except Exception as exc:
            print(f"{BAD} 检查本身抛异常：{type(exc).__name__}: {exc}")
            results.append(False)

    print("\n" + "=" * 66)
    print(f"通过 {sum(results)}/{len(results)} 项")
    print("=" * 66)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
