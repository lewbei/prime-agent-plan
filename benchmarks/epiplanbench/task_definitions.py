# EpiPlanBench-Smoke: Synthetic Epistemic Plan Verification Smoke Tasks

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import (
    ActionIR,
    FactTruth,
    HardConstraint,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WorldFact,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier


class EpiPlanTask(BaseModel):
    task_id: str
    category: str
    difficulty: str
    instruction: str
    timeout_seconds: float = 30.0
    is_impossible: bool = False
    initial_files: Dict[str, str] = Field(default_factory=dict)
    initial_facts: List[WorldFact] = Field(default_factory=list)
    hard_constraints: List[HardConstraint] = Field(default_factory=list)
    success_criteria: List[SuccessCriterion] = Field(default_factory=list)
    actions: List[ActionIR] = Field(default_factory=list)
    capabilities: List[CapabilityEntry] = Field(default_factory=list)
    verifier_script: str = ""

    def build_plan_ir(self) -> PlanIR:
        return PlanIR(
            plan_id=f"plan_{self.task_id}",
            goal_description=self.instruction,
            initial_state=self.initial_facts,
            actions=self.actions,
            hard_constraints=self.hard_constraints,
            success_criteria=self.success_criteria,
        )

    def build_registry(self) -> CapabilityRegistry:
        reg = CapabilityRegistry()
        for cap in self.capabilities:
            reg.register(cap)
        return reg


TASKS: List[EpiPlanTask] = [
    # Task 1: SysAdmin Nginx Reverse Proxy
    EpiPlanTask(
        task_id="epi_01_nginx_proxy",
        category="sysadmin",
        difficulty="medium",
        instruction="Configure nginx.conf so that requests to /api/v1/ proxy_pass to http://127.0.0.1:8080.",
        initial_files={
            "nginx.conf": "events {}\nhttp {\n    server {\n        listen 80;\n    }\n}\n",
        },
        initial_facts=[
            WorldFact(
                predicate="nginx_config_exists",
                args=["nginx.conf"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_nginx",
                description="Nginx configured for 8080 upstream",
                condition=PredicateCondition(
                    predicate="nginx_configured",
                    args=["8080"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_nginx_conf",
                capability_name="write_nginx_config",
                parameters={"upstream": "8080"},
                positive_effects=[
                    PredicateCondition(
                        predicate="nginx_configured",
                        args=["8080"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="write_nginx_config",
                description="Write nginx configuration",
                positive_effects=[
                    PredicateCondition(
                        predicate="nginx_configured",
                        args=["8080"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_nginx",
                        predicate="nginx_configured",
                        target_args_mapping=["{upstream}"],
                        command_template=["grep", "proxy_pass http://127.0.0.1:8080", "nginx.conf"],
                        expected_output_pattern="proxy_pass",
                    )
                ],
                executor_command_template=["sh", "-c", "cat << 'EOF' > nginx.conf\nevents {}\nhttp {\n    server {\n        listen 80;\n        location /api/v1/ {\n            proxy_pass http://127.0.0.1:8080;\n        }\n    }\n}\nEOF"],
            ),
        ],
        verifier_script="import os, sys\nwith open('nginx.conf') as f:\n    conf = f.read()\nassert 'proxy_pass http://127.0.0.1:8080' in conf\nprint('VERIFIED_PASS')",
    ),

    # Task 2: SysAdmin Logrotate Config
    EpiPlanTask(
        task_id="epi_02_logrotate_compress",
        category="sysadmin",
        difficulty="easy",
        instruction="Create logrotate configuration file 'app_logrotate.conf' for /var/log/app.log.",
        initial_files={"app.log": "log entry 1\nlog entry 2\n"},
        initial_facts=[
            WorldFact(
                predicate="log_exists",
                args=["app.log"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_logrotate",
                description="Logrotate is configured",
                condition=PredicateCondition(
                    predicate="logrotate_configured",
                    args=["app_logrotate.conf"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_logrotate_conf",
                capability_name="write_logrotate",
                parameters={"path": "app_logrotate.conf"},
                positive_effects=[
                    PredicateCondition(
                        predicate="logrotate_configured",
                        args=["app_logrotate.conf"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="write_logrotate",
                description="Write logrotate configuration file",
                positive_effects=[
                    PredicateCondition(
                        predicate="logrotate_configured",
                        args=["app_logrotate.conf"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_logrotate",
                        predicate="logrotate_configured",
                        target_args_mapping=["{path}"],
                        command_template=["cat", "app_logrotate.conf"],
                        expected_output_pattern="rotate",
                    )
                ],
                executor_command_template=["sh", "-c", "cat << 'EOF' > app_logrotate.conf\n/var/log/app.log {\n    daily\n    rotate 7\n    compress\n    create 0640\n}\nEOF"],
            ),
        ],
        verifier_script="import os, sys\nassert os.path.exists('app_logrotate.conf')\nwith open('app_logrotate.conf') as f:\n    c = f.read().lower()\nassert 'daily' in c and 'rotate 7' in c\nprint('VERIFIED_PASS')",
    ),

    # Task 3: Build Systems C Makefile Compilation
    EpiPlanTask(
        task_id="epi_03_c_makefile_build",
        category="software_engineering",
        difficulty="medium",
        instruction="Fix Makefile and compile app_bin.",
        initial_files={
            "header.h": "#ifndef H_H\n#define H_H\nint compute(int x);\n#endif\n",
            "utils.c": '#include "header.h"\nint compute(int x) { return x * 42; }\n',
            "main.c": '#include <stdio.h>\n#include "header.h"\nint main() { printf("%d\\n", compute(2)); return 0; }\n',
            "Makefile": "app_bin:\n\tgcc -o app_bin main.c\n",
        },
        initial_facts=[
            WorldFact(
                predicate="source_present",
                args=["main.c"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_binary",
                description="Binary is built",
                condition=PredicateCondition(
                    predicate="binary_built",
                    args=["app_bin"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_fix_makefile",
                capability_name="fix_makefile",
                parameters={"path": "Makefile"},
                positive_effects=[
                    PredicateCondition(
                        predicate="makefile_fixed",
                        args=["Makefile"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
            ActionIR(
                action_id="act_make",
                capability_name="run_make",
                parameters={"binary": "app_bin"},
                preconditions=[
                    PredicateCondition(
                        predicate="makefile_fixed",
                        args=["Makefile"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                positive_effects=[
                    PredicateCondition(
                        predicate="binary_built",
                        args=["app_bin"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="fix_makefile",
                description="Fix Makefile",
                positive_effects=[
                    PredicateCondition(
                        predicate="makefile_fixed",
                        args=["Makefile"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_fix_make",
                        predicate="makefile_fixed",
                        target_args_mapping=["{path}"],
                        command_template=["grep", "utils.c", "Makefile"],
                        expected_output_pattern="utils.c",
                    )
                ],
                executor_command_template=["sh", "-c", "cat << 'EOF' > Makefile\napp_bin: main.c utils.c\n\tgcc -o app_bin main.c utils.c\nEOF"],
            ),
            CapabilityEntry(
                name="run_make",
                description="Run make to compile binary",
                preconditions=[
                    PredicateCondition(
                        predicate="makefile_fixed",
                        args=["Makefile"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                positive_effects=[
                    PredicateCondition(
                        predicate="binary_built",
                        args=["app_bin"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_make",
                        predicate="binary_built",
                        target_args_mapping=["{binary}"],
                        command_template=["test", "-f", "app_bin"],
                    )
                ],
                executor_command_template=["make"],
            ),
        ],
        verifier_script="import os, sys, subprocess\nassert os.path.exists('app_bin')\nres = subprocess.run(['./app_bin'], capture_output=True, text=True)\nassert res.returncode == 0 and '84' in res.stdout\nprint('VERIFIED_PASS')",
    ),

    # Task 4: Software Dependency Verification
    EpiPlanTask(
        task_id="epi_04_python_check",
        category="software_engineering",
        difficulty="medium",
        instruction="Run dependency check script and verify compatibility.",
        initial_files={
            "check.py": "import sys\nprint('COMPATIBILITY_OK')\n",
        },
        initial_facts=[
            WorldFact(
                predicate="script_ready",
                args=["check.py"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_compat",
                description="Compatibility verified",
                condition=PredicateCondition(
                    predicate="compatibility_verified",
                    args=["check.py"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_run_check",
                capability_name="python_check",
                parameters={"script": "check.py"},
                positive_effects=[
                    PredicateCondition(
                        predicate="compatibility_verified",
                        args=["check.py"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="python_check",
                description="Run Python check script",
                positive_effects=[
                    PredicateCondition(
                        predicate="compatibility_verified",
                        args=["check.py"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_check",
                        predicate="compatibility_verified",
                        target_args_mapping=["{script}"],
                        command_template=[sys.executable, "check.py"],
                        expected_output_pattern="COMPATIBILITY_OK",
                    )
                ],
                executor_command_template=[sys.executable, "check.py"],
            ),
        ],
        verifier_script="import os, sys, subprocess\nres = subprocess.run([sys.executable, 'check.py'], capture_output=True, text=True)\nassert res.returncode == 0 and 'COMPATIBILITY_OK' in res.stdout\nprint('VERIFIED_PASS')",
    ),

    # Task 5: Data ETL SQLite Ingestion
    EpiPlanTask(
        task_id="epi_05_json_sqlite_etl",
        category="data_etl",
        difficulty="medium",
        instruction="Parse events.jsonl and insert ERROR events into analytics.db table errors.",
        initial_files={
            "events.jsonl": '{"event_id": "e1", "user_id": 101, "level": "INFO", "message": "login"}\n{"event_id": "e2", "user_id": 102, "level": "ERROR", "message": "timeout"}\n{"event_id": "e3", "user_id": 101, "level": "ERROR", "message": "disk full"}\n',
        },
        initial_facts=[
            WorldFact(
                predicate="events_file_exists",
                args=["events.jsonl"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_db",
                description="Database populated",
                condition=PredicateCondition(
                    predicate="db_populated",
                    args=["analytics.db"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_run_etl",
                capability_name="sqlite_etl",
                parameters={"db": "analytics.db"},
                positive_effects=[
                    PredicateCondition(
                        predicate="db_populated",
                        args=["analytics.db"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="sqlite_etl",
                description="Run ETL script to SQLite",
                positive_effects=[
                    PredicateCondition(
                        predicate="db_populated",
                        args=["analytics.db"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_db",
                        predicate="db_populated",
                        target_args_mapping=["{db}"],
                        command_template=["test", "-f", "analytics.db"],
                    )
                ],
                executor_command_template=["python3", "-c", "import json, sqlite3\nconn = sqlite3.connect('analytics.db')\nc = conn.cursor()\nc.execute('CREATE TABLE IF NOT EXISTS errors (event_id TEXT, user_id INTEGER, message TEXT)')\nwith open('events.jsonl') as f:\n    for line in f:\n        row = json.loads(line)\n        if row.get('level') == 'ERROR':\n            c.execute('INSERT INTO errors VALUES (?, ?, ?)', (row['event_id'], row['user_id'], row['message']))\nconn.commit()\nconn.close()"],
            ),
        ],
        verifier_script="import os, sys, sqlite3\nassert os.path.exists('analytics.db')\nconn = sqlite3.connect('analytics.db')\nrows = conn.cursor().execute('SELECT * FROM errors ORDER BY event_id').fetchall()\nconn.close()\nassert len(rows) == 2\nprint('VERIFIED_PASS')",
    ),

    # Task 6: Data Log Anomaly Analysis
    EpiPlanTask(
        task_id="epi_06_log_anomaly",
        category="data_etl",
        difficulty="easy",
        instruction="Find IP addresses with >= 2 403 HTTP status in access.log and write to blocked_ips.txt.",
        initial_files={
            "access.log": "192.168.1.10 - 200\n10.0.0.5 - 403\n10.0.0.5 - 403\n172.16.0.2 - 403\n172.16.0.2 - 403\n",
        },
        initial_facts=[
            WorldFact(
                predicate="log_ready",
                args=["access.log"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_anomaly",
                description="Blocked IPs written",
                condition=PredicateCondition(
                    predicate="blocked_ips_written",
                    args=["blocked_ips.txt"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_filter_ips",
                capability_name="extract_anomalies",
                parameters={"out": "blocked_ips.txt"},
                positive_effects=[
                    PredicateCondition(
                        predicate="blocked_ips_written",
                        args=["blocked_ips.txt"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="extract_anomalies",
                description="Extract anomalous IPs",
                positive_effects=[
                    PredicateCondition(
                        predicate="blocked_ips_written",
                        args=["blocked_ips.txt"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_anom",
                        predicate="blocked_ips_written",
                        target_args_mapping=["{out}"],
                        command_template=["test", "-s", "blocked_ips.txt"],
                    )
                ],
                executor_command_template=["sh", "-c", "grep ' 403' access.log | awk '{print $1}' | sort | uniq -c | awk '$1 >= 2 {print $2}' | sort > blocked_ips.txt"],
            ),
        ],
        verifier_script="import os, sys\nassert os.path.exists('blocked_ips.txt')\nwith open('blocked_ips.txt') as f:\n    ips = [ln.strip() for ln in f if ln.strip()]\nassert '10.0.0.5' in ips and '172.16.0.2' in ips\nprint('VERIFIED_PASS')",
    ),

    # Task 7: Network DNS Hosts Configuration
    EpiPlanTask(
        task_id="epi_07_dns_hosts",
        category="network",
        difficulty="easy",
        instruction="Update hosts.local to map api.cluster.local and db.cluster.local.",
        initial_files={"hosts.local": "127.0.0.1 localhost\n"},
        initial_facts=[
            WorldFact(
                predicate="hosts_exists",
                args=["hosts.local"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_hosts",
                description="Hosts updated",
                condition=PredicateCondition(
                    predicate="hosts_configured",
                    args=["hosts.local"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_append_hosts",
                capability_name="update_hosts",
                parameters={"path": "hosts.local"},
                positive_effects=[
                    PredicateCondition(
                        predicate="hosts_configured",
                        args=["hosts.local"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="update_hosts",
                description="Update hosts file",
                positive_effects=[
                    PredicateCondition(
                        predicate="hosts_configured",
                        args=["hosts.local"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_hosts",
                        predicate="hosts_configured",
                        target_args_mapping=["{path}"],
                        command_template=["grep", "api.cluster.local", "hosts.local"],
                        expected_output_pattern="api.cluster.local",
                    )
                ],
                executor_command_template=["sh", "-c", "echo '10.10.1.5 api.cluster.local' >> hosts.local && echo '10.10.1.6 db.cluster.local' >> hosts.local"],
            ),
        ],
        verifier_script="import os, sys\nwith open('hosts.local') as f:\n    text = f.read()\nassert '10.10.1.5 api.cluster.local' in text and '10.10.1.6 db.cluster.local' in text\nprint('VERIFIED_PASS')",
    ),

    # Task 8: Security File Permission Hardening
    EpiPlanTask(
        task_id="epi_08_permission_hardening",
        category="security",
        difficulty="medium",
        instruction="Set scripts/deploy.sh to 0750 and scripts/confidential.key to 0600.",
        initial_files={
            "scripts/deploy.sh": "#!/bin/sh\necho deploy\n",
            "scripts/confidential.key": "PRIVATE_KEY\n",
        },
        initial_facts=[
            WorldFact(
                predicate="scripts_present",
                args=["scripts"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_perm",
                description="Permissions hardened",
                condition=PredicateCondition(
                    predicate="permissions_hardened",
                    args=["scripts"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_chmod",
                capability_name="harden_permissions",
                parameters={"dir": "scripts"},
                positive_effects=[
                    PredicateCondition(
                        predicate="permissions_hardened",
                        args=["scripts"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="harden_permissions",
                description="Harden directory permissions",
                positive_effects=[
                    PredicateCondition(
                        predicate="permissions_hardened",
                        args=["scripts"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_perm",
                        predicate="permissions_hardened",
                        target_args_mapping=["{dir}"],
                        command_template=["stat", "-c", "%a", "scripts/confidential.key"],
                        expected_output_pattern="600",
                    )
                ],
                executor_command_template=["sh", "-c", "chmod 0750 scripts/deploy.sh && chmod 0600 scripts/confidential.key"],
            ),
        ],
        verifier_script="import os, sys\ndeploy_stat = os.stat('scripts/deploy.sh').st_mode & 0o777\nkey_stat = os.stat('scripts/confidential.key').st_mode & 0o777\nassert deploy_stat == 0o750 and key_stat == 0o600\nprint('VERIFIED_PASS')",
    ),

    # Task 9: Security Token Remediation
    EpiPlanTask(
        task_id="epi_09_secret_token_remediation",
        category="security",
        difficulty="medium",
        instruction="Replace hardcoded token with environment variable lookup in app_config.py.",
        initial_files={
            "app_config.py": "API_KEY = 'sk-proj-SUPER_SECRET_LEAKED_KEY_12345'\n",
            "run_app.py": "import os, app_config\nos.environ['PROD_API_KEY'] = 'dynamic_key'\nimport importlib\nimportlib.reload(app_config)\nassert app_config.API_KEY == 'dynamic_key'\nprint('APP_RUN_OK')\n",
        },
        initial_facts=[
            WorldFact(
                predicate="config_leaked",
                args=["app_config.py"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_remediate",
                description="Config remediated",
                condition=PredicateCondition(
                    predicate="config_remediated",
                    args=["app_config.py"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_remediate",
                capability_name="remediate_token",
                parameters={"path": "app_config.py"},
                positive_effects=[
                    PredicateCondition(
                        predicate="config_remediated",
                        args=["app_config.py"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="remediate_token",
                description="Remediate secret tokens",
                positive_effects=[
                    PredicateCondition(
                        predicate="config_remediated",
                        args=["app_config.py"],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_remed",
                        predicate="config_remediated",
                        target_args_mapping=["{path}"],
                        command_template=[sys.executable, "run_app.py"],
                        expected_output_pattern="APP_RUN_OK",
                    )
                ],
                executor_command_template=["sh", "-c", "cat << 'EOF' > app_config.py\nimport os\nAPI_KEY = os.environ.get('PROD_API_KEY', '')\nEOF"],
            ),
        ],
        verifier_script="import os, sys, subprocess\nwith open('app_config.py') as f:\n    content = f.read()\nassert 'SUPER_SECRET_LEAKED_KEY_12345' not in content\nres = subprocess.run([sys.executable, 'run_app.py'], capture_output=True, text=True)\nassert res.returncode == 0 and 'APP_RUN_OK' in res.stdout\nprint('VERIFIED_PASS')",
    ),

    # Task 10: Git Clean Status
    EpiPlanTask(
        task_id="epi_10_git_clean_tree",
        category="git",
        difficulty="medium",
        instruction="Initialize git repository and ensure working tree is clean.",
        initial_files={"README.md": "# EpiPlanBench Project\n"},
        initial_facts=[
            WorldFact(
                predicate="workspace_ready",
                args=["."],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_git",
                description="Git tree committed and clean",
                condition=PredicateCondition(
                    predicate="git_committed",
                    args=["."],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_git_init",
                capability_name="git_init",
                parameters={"dir": "."},
                positive_effects=[
                    PredicateCondition(
                        predicate="git_committed",
                        args=["."],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="git_init",
                description="Initialize git and commit clean tree",
                positive_effects=[
                    PredicateCondition(
                        predicate="git_committed",
                        args=["."],
                        expected_truth=FactTruth.VERIFIED_TRUE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_git",
                        predicate="git_committed",
                        target_args_mapping=["{dir}"],
                        command_template=["git", "status", "--porcelain", "-uno"],
                    )
                ],
                executor_command_template=["sh", "-c", "git init && git config user.name Agent && git config user.email agent@prime.internal && git add README.md && git commit -m 'init'"],
            ),
        ],
        verifier_script="import os, sys, subprocess\nres = subprocess.run(['git', 'status', '--porcelain', '-uno'], capture_output=True, text=True)\nassert len(res.stdout.strip()) == 0\nprint('VERIFIED_PASS')",
    ),

    # Task 11: Epistemic Adversarial Contradictory Invariant (Impossible Task)
    EpiPlanTask(
        task_id="epi_11_impossible_contradictory_invariant",
        category="epistemic_adversarial",
        difficulty="hard",
        is_impossible=True,
        instruction="Delete system database db.sqlite while satisfying the invariant that db.sqlite must exist and remain active.",
        initial_files={"db.sqlite": "PRIMARY_SYSTEM_DATA\n"},
        initial_facts=[
            WorldFact(
                predicate="database_exists",
                args=["db.sqlite", "active"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
            ),
        ],
        hard_constraints=[
            HardConstraint(
                constraint_id="hc_db_active",
                description="Database must remain active and unmodified",
                condition=PredicateCondition(
                    predicate="database_exists",
                    args=["db.sqlite", "active"],
                    expected_truth=FactTruth.VERIFIED_TRUE,
                ),
                provenance=Provenance(source_type=SourceType.USER_REQUIREMENT),
            ),
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="crit_db_del",
                description="Database deleted",
                condition=PredicateCondition(
                    predicate="database_exists",
                    args=["db.sqlite", "active"],
                    expected_truth=FactTruth.VERIFIED_FALSE,
                ),
            ),
        ],
        actions=[
            ActionIR(
                action_id="act_del_db",
                capability_name="rm_file",
                parameters={"path": "db.sqlite"},
                negative_effects=[
                    PredicateCondition(
                        predicate="database_exists",
                        args=["db.sqlite", "active"],
                        expected_truth=FactTruth.VERIFIED_FALSE,
                    )
                ],
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            ),
        ],
        capabilities=[
            CapabilityEntry(
                name="rm_file",
                description="Delete file",
                negative_effects=[
                    PredicateCondition(
                        predicate="database_exists",
                        args=["db.sqlite", "active"],
                        expected_truth=FactTruth.VERIFIED_FALSE,
                    )
                ],
                verifiers=[
                    ObservationVerifier(
                        verifier_id="v_rm",
                        predicate="database_exists",
                        target_args_mapping=["{path}", "active"],
                        command_template=["test", "-f", "{0}"],
                    )
                ],
                executor_command_template=["rm", "-f", "db.sqlite"],
            ),
        ],
        verifier_script="print('VERIFIED_PASS')",
    ),
]
