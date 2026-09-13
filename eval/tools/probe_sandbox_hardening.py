"""代码沙箱的取证探针：把同一段探测代码同时喂给**改之前**和**改之后**两版实现。

用途：改 `src/agent/node/finance.py` 的沙箱之后，用可复现的数字说明"到底紧了什么"。
探测内容就是"模型生成的代码真这么写就能做到"的四件事：

    E 读环境变量（.env 里的 key 就在子进程环境里）
    F 起子进程 + 往仓库写文件（升级前的 cwd 就是仓库根）
    G 死循环（超时）
    H print 洪水（输出内存）

    .\\venv\\Scripts\\python.exe eval\\tools\\probe_sandbox_hardening.py
    .\\venv\\Scripts\\python.exe eval\\tools\\probe_sandbox_hardening.py --old HEAD~1

**注意**：它对比的是"改前 vs 改后"，所以要在**提交之前**跑；提交之后再跑，
`--old` 取到的就是同一版，E/F/G/H 会打印成"旧 = 新"（那时它退化成一份
"当前实现会拒绝哪四类行为"的自检）。

只打印布尔、字符数与规则名，**不打印任何密钥值**。
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agent.node import finance as new_sandbox  # noqa: E402

OLD_MODULE_PATH = REPO / ".tmp_finance_old.py"

# (标签, 代码, 超时)
PROBES = (
    ("E 读环境变量",
     "import os\n"
     "k = 'DEEP' + 'SEEK_API_KEY'\n"
     "print('env_visible=', bool(os.environ.get(k)))\n"
     "print('cwd=', os.getcwd())\n",
     None),
    ("F 起子进程 + 往仓库写文件",
     "import subprocess, os, sys\n"
     "p = subprocess.run([sys.executable, '-c', 'print(1)'], capture_output=True, text=True)\n"
     "print('subprocess_ok=', p.returncode == 0)\n"
     "open('.tmp_pwned.txt', 'w').write('x')\n"
     "print('write_ok=', os.path.exists('.tmp_pwned.txt'))\n",
     None),
    ("G 死循环超时", "while True:\n    pass\n", 3),
    ("H print 洪水", "while True:\n    print('x' * 100)\n", 5),
)


def load_old(ref: str):
    """把 ref 里的旧版节点作为独立模块加载（不污染 sys.modules 里的当前版）。"""
    src = subprocess.run(
        ["git", "show", f"{ref}:src/agent/node/finance.py"],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO,
    )
    if src.returncode != 0:
        print(f"取不到 {ref} 版本的 finance.py：{src.stderr.strip()}")
        raise SystemExit(2)
    OLD_MODULE_PATH.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_finance_old", OLD_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="代码沙箱改动前后的行为取证")
    ap.add_argument("--old", default="HEAD", help="旧版所在的 git ref（默认 HEAD）")
    args = ap.parse_args()

    old_sandbox = load_old(args.old)
    print(f"对比 {args.old} 与工作区\n")

    for label, code, timeout in PROBES:
        print("=" * 72)
        print(label)
        for name, mod in ((f"旧({args.old})", old_sandbox), ("新(工作区)", new_sandbox)):
            fn = mod.execute_code_sandbox
            r = fn(code, timeout=timeout) if timeout else fn(code)
            stdout = r.get("stdout") or ""
            print(f"  {name}: exit={r.get('exit_code')} rejected={r.get('rejected')} "
                  f"timed_out={r.get('timed_out')} stdout_len={len(stdout)}")
            print(f"      stdout={stdout.strip().replace(chr(10), ' | ')[:120]!r}")
            if r.get("stderr"):
                print(f"      stderr={r['stderr'].strip()[:110]!r}")

    try:
        OLD_MODULE_PATH.unlink()
    except OSError:
        pass
    print("\n（若 F 那行旧版报了 write_ok=True，仓库根会多出 .tmp_pwned.txt，删掉即可）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
