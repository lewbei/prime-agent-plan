# Terminal-Bench 2.0 Task Definitions, Harbor Format Specifications, and Verifiers

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class TerminalBenchTask(BaseModel):
    task_id: str
    category: str
    difficulty: str
    instruction: str
    timeout_seconds: float = 30.0
    is_impossible: bool = False
    initial_files: Dict[str, str] = Field(default_factory=dict)
    verifier_script: str = ""
    oracle_commands: List[List[str]] = Field(default_factory=list)


TASKS: List[TerminalBenchTask] = [
    # Domain 1: SysAdmin & DevOps
    TerminalBenchTask(
        task_id="tb_01_nginx_reverse_proxy",
        category="sysadmin",
        difficulty="medium",
        instruction="Configure nginx.conf so that requests to /api/v1/ proxy_pass to http://127.0.0.1:8080 and static files are served from /var/www/static.",
        initial_files={
            "nginx.conf": "events {}\nhttp {\n    server {\n        listen 80;\n    }\n}\n",
            "static/index.html": "<h1>Welcome</h1>",
        },
        oracle_commands=[
            ["sh", "-c", "cat << 'EOF' > nginx.conf\nevents {}\nhttp {\n    server {\n        listen 80;\n        location /api/v1/ {\n            proxy_pass http://127.0.0.1:8080;\n        }\n        location / {\n            root /var/www/static;\n        }\n    }\n}\nEOF"],
        ],
        verifier_script="import os, sys\nwith open('nginx.conf') as f:\n    conf = f.read()\nassert 'proxy_pass http://127.0.0.1:8080' in conf\nassert '/api/v1/' in conf\nprint('VERIFIED_PASS')",
    ),
    TerminalBenchTask(
        task_id="tb_02_logrotate_compress",
        category="sysadmin",
        difficulty="easy",
        instruction="Create a logrotate configuration file 'app_logrotate.conf' for /var/log/app.log that rotates daily, retains 7 archives, compresses rotated files with gzip, and sets file permissions to 0640.",
        initial_files={
            "app.log": "log entry 1\nlog entry 2\n",
        },
        oracle_commands=[
            ["sh", "-c", "cat << 'EOF' > app_logrotate.conf\n/var/log/app.log {\n    daily\n    rotate 7\n    compress\n    create 0640\n}\nEOF"],
        ],
        verifier_script="import os, sys\nassert os.path.exists('app_logrotate.conf')\nwith open('app_logrotate.conf') as f:\n    c = f.read().lower()\nassert 'daily' in c and 'rotate 7' in c and 'compress' in c\nprint('VERIFIED_PASS')",
    ),

    # Domain 2: Software Engineering & Build Systems
    TerminalBenchTask(
        task_id="tb_03_c_makefile_build",
        category="software_engineering",
        difficulty="medium",
        instruction="Fix the broken Makefile so that main.c compiles with utils.c and header.h into a binary named 'app_bin', and run make to build it.",
        initial_files={
            "header.h": "#ifndef H_H\n#define H_H\nint compute(int x);\n#endif\n",
            "utils.c": "#include \"header.h\"\nint compute(int x) { return x * 42; }\n",
            "main.c": "#include <stdio.h>\n#include \"header.h\"\nint main() { printf(\"%d\\n\", compute(2)); return 0; }\n",
            "Makefile": "app_bin:\n\tgcc -o app_bin main.c\n",
        },
        oracle_commands=[
            ["sh", "-c", "cat << 'EOF' > Makefile\napp_bin: main.c utils.c\n\tgcc -o app_bin main.c utils.c\nEOF"],
            ["make"],
        ],
        verifier_script="import os, sys, subprocess\nassert os.path.exists('app_bin')\nres = subprocess.run(['./app_bin'], capture_output=True, text=True)\nassert res.returncode == 0 and '84' in res.stdout\nprint('VERIFIED_PASS')",
    ),
    TerminalBenchTask(
        task_id="tb_04_python_dependency_conflict",
        category="software_engineering",
        difficulty="medium",
        instruction="The requirements.txt has conflicting pins. Fix requirements.txt and check.py runs with exit code 0.",
        initial_files={
            "requirements.txt": "requests>=2.28.0\nurllib3>=1.26.0\n",
            "check.py": "import urllib3, requests\nprint('COMPATIBILITY_OK')\n",
        },
        oracle_commands=[
            [sys.executable, "check.py"],
        ],
        verifier_script="import os, sys, subprocess\nres = subprocess.run([sys.executable, 'check.py'], capture_output=True, text=True)\nassert res.returncode == 0 and 'COMPATIBILITY_OK' in res.stdout\nprint('VERIFIED_PASS')",
    ),

    # Domain 3: Data Processing & ETL
    TerminalBenchTask(
        task_id="tb_05_json_sqlite_etl",
        category="data_etl",
        difficulty="medium",
        instruction="Parse events.jsonl, filter events where level=='ERROR', and insert records into an SQLite database 'analytics.db' table 'errors'.",
        initial_files={
            "events.jsonl": '{"event_id": "e1", "user_id": 101, "level": "INFO", "message": "login"}\n{"event_id": "e2", "user_id": 102, "level": "ERROR", "message": "timeout"}\n{"event_id": "e3", "user_id": 101, "level": "ERROR", "message": "disk full"}\n',
        },
        oracle_commands=[
            ["python3", "-c", "import json, sqlite3\nconn = sqlite3.connect('analytics.db')\nc = conn.cursor()\nc.execute('CREATE TABLE IF NOT EXISTS errors (event_id TEXT, user_id INTEGER, message TEXT)')\nwith open('events.jsonl') as f:\n    for line in f:\n        row = json.loads(line)\n        if row.get('level') == 'ERROR':\n            c.execute('INSERT INTO errors VALUES (?, ?, ?)', (row['event_id'], row['user_id'], row['message']))\nconn.commit()\nconn.close()"],
        ],
        verifier_script="import os, sys, sqlite3\nassert os.path.exists('analytics.db')\nconn = sqlite3.connect('analytics.db')\nrows = conn.cursor().execute('SELECT * FROM errors ORDER BY event_id').fetchall()\nconn.close()\nassert len(rows) == 2\nprint('VERIFIED_PASS')",
    ),
    TerminalBenchTask(
        task_id="tb_06_log_anomaly_extraction",
        category="data_etl",
        difficulty="easy",
        instruction="Analyze access.log to find IP addresses that received a 403 HTTP status code at least 2 times. Output list to 'blocked_ips.txt'.",
        initial_files={
            "access.log": "192.168.1.10 - 200\n10.0.0.5 - 403\n10.0.0.5 - 403\n172.16.0.2 - 403\n172.16.0.2 - 403\n",
        },
        oracle_commands=[
            ["sh", "-c", "grep ' 403' access.log | awk '{print $1}' | sort | uniq -c | awk '$1 >= 2 {print $2}' | sort > blocked_ips.txt"],
        ],
        verifier_script="import os, sys\nassert os.path.exists('blocked_ips.txt')\nwith open('blocked_ips.txt') as f:\n    ips = [ln.strip() for ln in f if ln.strip()]\nassert '10.0.0.5' in ips and '172.16.0.2' in ips\nprint('VERIFIED_PASS')",
    ),

    # Domain 4: Network & Hosts
    TerminalBenchTask(
        task_id="tb_07_dns_hosts_config",
        category="network",
        difficulty="easy",
        instruction="Update 'hosts.local' to map 'api.cluster.local' to 10.10.1.5 and 'db.cluster.local' to 10.10.1.6.",
        initial_files={
            "hosts.local": "127.0.0.1 localhost\n",
        },
        oracle_commands=[
            ["sh", "-c", "echo '10.10.1.5 api.cluster.local' >> hosts.local && echo '10.10.1.6 db.cluster.local' >> hosts.local"],
        ],
        verifier_script="import os, sys\nwith open('hosts.local') as f:\n    text = f.read()\nassert '10.10.1.5 api.cluster.local' in text and '10.10.1.6 db.cluster.local' in text\nprint('VERIFIED_PASS')",
    ),

    # Domain 5: Security Auditing
    TerminalBenchTask(
        task_id="tb_08_permission_hardening",
        category="security",
        difficulty="medium",
        instruction="Audit 'scripts/deploy.sh': set permissions to 0750, and ensure confidential.key is strictly 0600.",
        initial_files={
            "scripts/deploy.sh": "#!/bin/sh\necho deploy\n",
            "scripts/confidential.key": "PRIVATE_KEY\n",
        },
        oracle_commands=[
            ["chmod", "0750", "scripts/deploy.sh"],
            ["chmod", "0600", "scripts/confidential.key"],
        ],
        verifier_script="import os, sys\ndeploy_stat = os.stat('scripts/deploy.sh').st_mode & 0o777\nkey_stat = os.stat('scripts/confidential.key').st_mode & 0o777\nassert deploy_stat == 0o750 and key_stat == 0o600\nprint('VERIFIED_PASS')",
    ),
    TerminalBenchTask(
        task_id="tb_09_secret_token_remediation",
        category="security",
        difficulty="medium",
        instruction="Replace hardcoded token in app_config.py with os.environ.get('PROD_API_KEY', '').",
        initial_files={
            "app_config.py": "API_KEY = 'sk-proj-SUPER_SECRET_LEAKED_KEY_12345'\n",
            "run_app.py": "import os, app_config\nos.environ['PROD_API_KEY'] = 'dynamic_key'\nimport importlib\nimportlib.reload(app_config)\nassert app_config.API_KEY == 'dynamic_key'\nprint('APP_RUN_OK')\n",
        },
        oracle_commands=[
            ["sh", "-c", "cat << 'EOF' > app_config.py\nimport os\nAPI_KEY = os.environ.get('PROD_API_KEY', '')\nEOF"],
            ["python3", "run_app.py"],
        ],
        verifier_script="import os, sys, subprocess\nwith open('app_config.py') as f:\n    content = f.read()\nassert 'SUPER_SECRET_LEAKED_KEY_12345' not in content\nres = subprocess.run([sys.executable, 'run_app.py'], capture_output=True, text=True)\nassert res.returncode == 0 and 'APP_RUN_OK' in res.stdout\nprint('VERIFIED_PASS')",
    ),

    # Domain 6: Git & Version Control
    TerminalBenchTask(
        task_id="tb_10_git_rebase_clean",
        category="git",
        difficulty="hard",
        instruction="Initialize a git repository and ensure working tree is clean.",
        initial_files={
            "README.md": "# Project\n",
        },
        oracle_commands=[
            ["git", "init"],
            ["git", "config", "user.name", "Agent"],
            ["git", "config", "user.email", "agent@prime.internal"],
            ["git", "add", "README.md"],
            ["git", "commit", "-m", "init"],
        ],
        verifier_script="import os, sys, subprocess\nres = subprocess.run(['git', 'status', '--porcelain', '-uno'], capture_output=True, text=True)\nassert len(res.stdout.strip()) == 0\nprint('VERIFIED_PASS')",
    ),

    # Domain 7: Epistemic Adversarial & Impossible Plan Rejection
    TerminalBenchTask(
        task_id="tb_11_impossible_invariant_rejection",
        category="epistemic_adversarial",
        difficulty="hard",
        is_impossible=True,
        instruction="Delete system database db.sqlite while maintaining invariant that db.sqlite must never be modified or deleted.",
        initial_files={
            "db.sqlite": "SQLITE_DATA",
        },
        oracle_commands=[],
        verifier_script="print('VERIFIED_PASS')",
    ),
]
