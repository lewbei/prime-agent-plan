"""Authoritative post-audit correctness hardening for the legacy plan_mode engine.

The legacy implementation is executed directly in the ``plan_mode`` module
namespace.  ``install`` replaces only behavior whose invariants were proven
wrong by adversarial tests, keeping all existing state/data formats compatible.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import hmac
import inspect
import json
import os
import random
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .fact_identity import canonical_fact_identity

_JUDGE_ATTESTATION_KEY = secrets.token_bytes(32)


def _stable_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _sign_judge_attestation(plan_hash: str, provider: str, model: str, response_digest: str, issued_at: float) -> dict[str, Any]:
    body = {
        "plan_hash": plan_hash,
        "provider": provider,
        "model": model,
        "response_digest": response_digest,
        "issued_at": round(float(issued_at), 6),
    }
    message = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["signature"] = hmac.new(_JUDGE_ATTESTATION_KEY, message, hashlib.sha256).hexdigest()
    return body


def _verify_judge_attestation(attestation: Any, expected_plan_hash: str) -> bool:
    if not isinstance(attestation, dict):
        return False
    required = {"plan_hash", "provider", "model", "response_digest", "issued_at", "signature"}
    if not required.issubset(attestation):
        return False
    if str(attestation.get("plan_hash")) != expected_plan_hash:
        return False
    body = {k: attestation[k] for k in ("plan_hash", "provider", "model", "response_digest", "issued_at")}
    message = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hmac.new(_JUDGE_ATTESTATION_KEY, message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, str(attestation.get("signature") or ""))


def _round_plan_text(state: Mapping[str, Any], version: int | None = None) -> str:
    rounds = list(state.get("rounds") or [])
    ver = version if version is not None else state.get("best_version")
    if isinstance(ver, int) and 1 <= ver <= len(rounds):
        return str(rounds[ver - 1].get("plan_text") or "")
    return ""


def install(ns: dict[str, Any]) -> None:
    """Install all audited invariants into the live ``plan_mode`` namespace."""
    from . import ast_search as ast_search_mod
    from . import execution_contract as execution_contract_mod
    from . import execution_trace as execution_trace_mod
    from . import ir_search as ir_search_mod
    from . import isolation as isolation_mod
    from . import judges as judges_mod
    from . import probing as probing_mod
    from . import recovery as recovery_mod
    from . import recovery_graph as recovery_graph_mod
    from . import registry as registry_mod
    from . import search_engine as search_engine_mod
    from .ir import PlanIR
    from .runtime import sandbox as sandbox_mod
    from .session import PlanningSession, SessionState

    # ------------------------------------------------------------------
    # Stable local search: never depend on CPython's process-randomized hash().
    # ------------------------------------------------------------------
    def stable_mutations(plan_text: str, width: int, critiques=None):
        digest = hashlib.sha256(search_engine_mod._norm(plan_text).encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        targeted = []
        if critiques:
            seen = set()
            for critique in critiques:
                section = (
                    critique.get("id", "").split(":", 1)[0]
                    if isinstance(critique, dict)
                    else str(critique).split(":", 1)[0]
                )
                for key, template in search_engine_mod._SECTION_TEMPLATES.items():
                    if key in section and key not in seen:
                        seen.add(key)
                        targeted.append((
                            f"target-{key}",
                            lambda text, template=template: text + template,
                        ))
        choices = targeted if targeted else list(search_engine_mod._MUTATIONS)
        selected = rng.sample(choices, min(width, len(choices)))
        return [{"text": fn(plan_text), "note": name} for name, fn in selected]

    search_engine_mod._mutations = stable_mutations

    # ---------------------------------------------------------
    # Plan identity: HMAC authorization must bind execution semantics.
    # ---------------------------------------------------------
    def hardened_plan_hash(self: PlanIR) -> str:
        def condition(cond):
            return {
                "key": cond.fact_key,
                "truth": cond.expected_truth.value,
                "active_until_action_id": cond.active_until_action_id,
            }

        dump_data = {
            "plan_id": self.plan_id,
            "goal_description": self.goal_description,
            "version": self.version,
            "initial_state": [
                {
                    "key": fact.fact_key,
                    "truth": fact.truth.value,
                    "witnessability": fact.witnessability.value,
                    "source": fact.provenance.source_type.value,
                }
                for fact in self.initial_state
            ],
            "actions": [
                {
                    "action_id": action.action_id,
                    "capability_name": action.capability_name,
                    "parameters": action.parameters,
                    "preconditions": [condition(c) for c in action.preconditions],
                    "positive_effects": [condition(c) for c in action.positive_effects],
                    "negative_effects": [condition(c) for c in action.negative_effects],
                    "compensation_action_id": action.compensation_action_id,
                    "is_idempotent": action.is_idempotent,
                    "timeout_seconds": action.timeout_seconds,
                }
                for action in self.actions
            ],
            "hard_constraints": [
                {
                    "constraint_id": constraint.constraint_id,
                    "condition": condition(constraint.condition),
                    "enforcement_level": constraint.enforcement_level,
                    "active_until_action_id": constraint.active_until_action_id,
                }
                for constraint in self.hard_constraints
            ],
            "success_criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "condition": condition(criterion.condition),
                    "is_mandatory": criterion.is_mandatory,
                }
                for criterion in self.success_criteria
            ],
        }
        return _stable_digest(dump_data)

    PlanIR.compute_hash = hardened_plan_hash

    # ---------------------------------------------------------
    # Session state machine: terminal states restart through DRAFT legally.
    # ---------------------------------------------------------
    raw_submit_draft = PlanningSession.submit_draft

    def hardened_submit_draft(self, plan_ir):
        if self.current_state in {SessionState.COMMITTED, SessionState.ROLLED_BACK, SessionState.FAILED}:
            self.transition_to(SessionState.DRAFT)
        return raw_submit_draft(self, plan_ir)

    PlanningSession.submit_draft = hardened_submit_draft

    # ---------------------------------------------------------
    # Diagnostic identity and parser failure semantics.
    # ---------------------------------------------------------
    probing_mod.DiagnosticProbe.fact_key = property(
        lambda self: canonical_fact_identity(self.target_predicate, self.target_args)
    )
    raw_parse_probe = probing_mod.VOIProbingEngine.parse_probe_output

    def hardened_parse_probe_output(self, probe, stdout: str, returncode: int):
        parser = probe.expected_output_parser
        if parser == "integer" and returncode == 0:
            cleaned = stdout.strip()
            try:
                value = int(cleaned)
            except (TypeError, ValueError):
                return ns["FactTruth"].UNKNOWN
            return ns["FactTruth"].VERIFIED_TRUE if value > 0 else ns["FactTruth"].VERIFIED_FALSE
        if parser == "json" and returncode == 0:
            try:
                data = json.loads(stdout)
            except (TypeError, ValueError, json.JSONDecodeError):
                return ns["FactTruth"].UNKNOWN
            return ns["FactTruth"].VERIFIED_TRUE if data else ns["FactTruth"].VERIFIED_FALSE
        return raw_parse_probe(self, probe, stdout, returncode)

    probing_mod.VOIProbingEngine.parse_probe_output = hardened_parse_probe_output

    # ---------------------------------------------------------
    # Registry schema is closed when a schema is declared.
    # ---------------------------------------------------------
    raw_validate_action = registry_mod.CapabilityRegistry.validate_action

    def hardened_validate_action(self, action):
        capability = self.get(action.capability_name)
        if capability.input_schema:
            extras = sorted(set(action.parameters) - set(capability.input_schema))
            if extras:
                raise registry_mod.SchemaMismatchError(
                    f"Action '{action.action_id}' supplied undeclared parameters for capability "
                    f"'{capability.name}': {extras}"
                )
        return raw_validate_action(self, action)

    registry_mod.CapabilityRegistry.validate_action = hardened_validate_action

    # ---------------------------------------------------------
    # AST transposition identity includes semantic action state.
    # ---------------------------------------------------------
    def hardened_ast_state_hash(self, ast):
        def atom(value):
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if hasattr(value, "__dict__"):
                return {k: atom(v) for k, v in sorted(value.__dict__.items())}
            if isinstance(value, dict):
                return {str(k): atom(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
            if isinstance(value, (list, tuple)):
                return [atom(v) for v in value]
            if isinstance(value, set):
                return sorted(atom(v) for v in value)
            return value

        payload = {
            "goal": ast.goal,
            "initial_state": atom(ast.initial_state),
            "target_propositions": atom(ast.target_propositions),
            "constraints": atom(ast.constraints),
            "actions": [
                {
                    "id": action.id,
                    "name": action.name,
                    "preconditions": atom(action.preconditions),
                    "add_effects": atom(action.add_effects),
                    "del_effects": atom(action.del_effects),
                    "inputs": atom(action.inputs),
                    "outputs": atom(action.outputs),
                    "dependencies": atom(action.dependencies),
                    "duration": action.duration,
                    "resources": atom(action.resources),
                }
                for action in ast.actions
            ],
        }
        return _stable_digest(payload)

    ast_search_mod.ASTSearchEngine._state_hash = hardened_ast_state_hash

    # ---------------------------------------------------------
    # Recovery graph must not mutate caller evidence.
    # ---------------------------------------------------------
    raw_recovery_decision = recovery_graph_mod.recovery_decision

    def hardened_recovery_decision(evidence, *, K: int = 2, policy=None):
        if K < 1:
            raise ValueError("K must be >= 1")
        return raw_recovery_decision(copy.deepcopy(evidence), K=K, policy=policy)

    recovery_graph_mod.recovery_decision = hardened_recovery_decision
    ns["recovery_decision"] = hardened_recovery_decision

    # ---------------------------------------------------------
    # Isolation conflict detection operates on currently-held ownership only.
    # ---------------------------------------------------------
    def hardened_detect_conflicts(self, *, isolation=isolation_mod.OperationIsolation.SERIALIZABLE):
        active = []
        for record in self.records:
            if record.operation == "release":
                active = [
                    prior for prior in active
                    if not (
                        prior.path == record.path
                        and prior.workspace == record.workspace
                        and prior.agent_id == record.agent_id
                    )
                ]
            else:
                active.append(record)

        conflicts = []
        seen = set()
        writes = [r for r in active if r.operation in ("write", "delete")]
        for index, left in enumerate(writes):
            for right in writes[index + 1:]:
                if left.path != right.path or left.workspace != right.workspace or left.agent_id == right.agent_id:
                    continue
                if isolation == isolation_mod.OperationIsolation.OPTIMISTIC and left.version != right.version:
                    continue
                key = (left.path, left.workspace, *sorted((left.agent_id, right.agent_id)))
                if key in seen:
                    continue
                seen.add(key)
                conflicts.append({
                    "path": left.path,
                    "workspace": left.workspace,
                    "agents": sorted({left.agent_id, right.agent_id}),
                    "operations": [left.operation, right.operation],
                    "versions": [left.version, right.version],
                    "isolation": isolation.value,
                })
        if isolation == isolation_mod.OperationIsolation.SERIALIZABLE:
            reads = [r for r in active if r.operation == "read"]
            for read in reads:
                for write in writes:
                    if (
                        read.path == write.path
                        and read.workspace == write.workspace
                        and read.agent_id != write.agent_id
                        and read.created_at <= write.created_at
                    ):
                        conflicts.append({
                            "path": read.path,
                            "workspace": read.workspace,
                            "agents": sorted({read.agent_id, write.agent_id}),
                            "operations": ["read", write.operation],
                            "versions": [read.version, write.version],
                            "isolation": isolation.value,
                            "reason": "read-write ordering conflict",
                        })
        return isolation_mod.ConflictReport(
            ok=not conflicts,
            conflicts=conflicts,
            artifacts=list(self.records),
        )

    isolation_mod.IsolationManager.detect_conflicts = hardened_detect_conflicts

    # ---------------------------------------------------------
    # Plan-declared filesystem observations stay inside the requested workspace.
    # ---------------------------------------------------------
    raw_artifact_audit = execution_contract_mod.artifact_audit

    def _safe_workspace_path(raw: str, cwd: str | Path) -> Path:
        root = Path(cwd).resolve()
        candidate = Path(raw)
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path '{raw}' resolves outside workspace '{root}'") from exc
        return resolved

    def hardened_artifact_audit(contract, *, cwd=None):
        root = Path(cwd or Path.cwd()).resolve()
        errors = []
        for raw in contract.expected_artifacts:
            try:
                _safe_workspace_path(str(raw), root)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            return {"ok": False, "errors": errors, "artifacts": {}}
        return raw_artifact_audit(contract, cwd=root)

    execution_contract_mod.artifact_audit = hardened_artifact_audit
    execution_trace_mod.artifact_audit = hardened_artifact_audit
    ns["artifact_audit"] = hardened_artifact_audit

    # ---------------------------------------------------------
    # Sandbox: mask Prime credentials, enforce env whitelist, total timeout,
    # drain every stage, and use pipefail semantics.
    # ---------------------------------------------------------
    raw_get_profile = sandbox_mod.SecurityProfile.get_profile

    def hardened_get_profile(cls, profile_type):
        policy = raw_get_profile(profile_type)
        prime_dir = os.path.expanduser("~/.prime")
        if prime_dir not in policy.blocked_paths:
            policy.blocked_paths.append(prime_dir)
        return policy

    sandbox_mod.SecurityProfile.get_profile = classmethod(hardened_get_profile)

    def hardened_execute_argv_pipeline(
        self,
        pipeline,
        cwd=None,
        env=None,
        timeout_seconds=10.0,
        input_data=None,
    ):
        if not pipeline:
            return sandbox_mod.SandboxExecutionResult(returncode=0)
        if timeout_seconds <= 0:
            return sandbox_mod.SandboxExecutionResult(
                stderr="Execution timeout must be > 0.", returncode=124, timeout_exceeded=True
            )
        isolation_required = self.policy.require_bwrap or not self.policy.allow_unisolated_fallback
        if isolation_required and not self.kernel_isolation_ready:
            return sandbox_mod.SandboxExecutionResult(
                stderr=(
                    "Security violation: kernel container isolation backend (bwrap) "
                    "is unavailable; execution refused by fail-closed policy."
                ),
                returncode=126,
            )
        effective_cwd = cwd or self.policy.workspace_dir or os.getcwd()
        exec_env = dict(self.DEFAULT_ENV_WHITELIST)
        for key in self.policy.env_whitelist:
            if key in os.environ and key not in exec_env:
                exec_env[key] = os.environ[key]
        if env:
            for key, value in env.items():
                if key in self.policy.env_whitelist:
                    exec_env[key] = value
        sec_err = self._check_command_security(pipeline, effective_cwd)
        if sec_err:
            return sandbox_mod.SandboxExecutionResult(stderr=sec_err, returncode=126)
        if self.policy.workspace_dir:
            try:
                sandbox_mod.validate_path_within_workspace(effective_cwd, self.policy.workspace_dir)
            except (sandbox_mod.PathTraversalEscapeError, sandbox_mod.SymlinkEscapeError) as exc:
                return sandbox_mod.SandboxExecutionResult(stderr=f"Security violation: {exc}", returncode=126)
        if not self.policy.allow_network:
            exec_env["PRIME_NETWORK_DENY"] = "1"
        if self.policy.read_only_root:
            exec_env["PRIME_READ_ONLY_ROOT"] = "1"
        if self.policy.workspace_dir:
            exec_env["PRIME_WORKSPACE_DIR"] = self.policy.workspace_dir

        hook_dir = tempfile.mkdtemp(prefix="prime_sec_hook_")
        hook_file = os.path.join(hook_dir, "sitecustomize.py")
        with open(hook_file, "w", encoding="utf-8") as handle:
            handle.write(sandbox_mod._NET_BLOCKER_SCRIPT)
        orig_pypath = exec_env.get("PYTHONPATH", "")
        exec_env["PYTHONPATH"] = f"{hook_dir}:{orig_pypath}" if orig_pypath else hook_dir

        start = time.monotonic()
        payload = input_data
        stderr_parts = []
        last_stdout = ""
        preexec = self._build_preexec_fn()
        try:
            for raw_cmd in pipeline:
                remaining = timeout_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(raw_cmd, timeout_seconds)
                cmd = self._wrap_command_with_bwrap(raw_cmd, effective_cwd)
                completed = subprocess.run(
                    cmd,
                    input=payload,
                    capture_output=True,
                    text=True,
                    cwd=effective_cwd,
                    env=exec_env,
                    timeout=remaining,
                    preexec_fn=preexec,
                )
                last_stdout = completed.stdout or ""
                if completed.stderr:
                    stderr_parts.append(completed.stderr)
                if completed.returncode != 0:
                    return sandbox_mod.SandboxExecutionResult(
                        stdout=self._truncate_and_scrub(last_stdout),
                        stderr=self._truncate_and_scrub("".join(stderr_parts)),
                        returncode=completed.returncode,
                        duration_ms=round((time.monotonic() - start) * 1000.0, 2),
                    )
                payload = last_stdout
            return sandbox_mod.SandboxExecutionResult(
                stdout=self._truncate_and_scrub(last_stdout),
                stderr=self._truncate_and_scrub("".join(stderr_parts)),
                returncode=0,
                duration_ms=round((time.monotonic() - start) * 1000.0, 2),
            )
        except subprocess.TimeoutExpired:
            return sandbox_mod.SandboxExecutionResult(
                stderr="Execution timed out and was forcefully terminated.",
                returncode=124,
                duration_ms=round((time.monotonic() - start) * 1000.0, 2),
                timeout_exceeded=True,
            )
        except Exception as exc:
            return sandbox_mod.SandboxExecutionResult(
                stderr=f"Execution error: {exc}",
                returncode=1,
                duration_ms=round((time.monotonic() - start) * 1000.0, 2),
            )
        finally:
            shutil.rmtree(hook_dir, ignore_errors=True)

    sandbox_mod.ExecutionSandbox.execute_argv_pipeline = hardened_execute_argv_pipeline

    # ---------------------------------------------------------
    # Legacy saga must execute compensation; never synthesize exit_code=0.
    # ---------------------------------------------------------
    def hardened_saga_rollback(self, executed_steps, plan_ir, registry, ledger, session):
        steps_to_undo = [
            step for step in executed_steps
            if step.exit_code == 0 and step.witness_status == recovery_mod.WitnessStatus.WITNESSED_TRUE
        ]
        if not steps_to_undo:
            if session.current_state != SessionState.ROLLED_BACK:
                session.transition_to(SessionState.ROLLED_BACK)
            return recovery_mod.SagaRecoveryReport(status=recovery_mod.RecoveryStatus.NO_RECOVERY_NEEDED)
        if session.current_state in (SessionState.EXECUTING, SessionState.DIAGNOSING):
            session.transition_to(SessionState.COMPENSATING)

        action_map = {action.action_id: action for action in plan_ir.actions}
        compensated = 0
        uncompensated = []
        notes = []
        for step in reversed(steps_to_undo):
            action = action_map.get(step.step_id)
            if action is None:
                continue
            cap = registry.get(action.capability_name)
            spec = cap.default_compensation
            if spec is None:
                uncompensated.append(action.capability_name)
                notes.append(f"Step '{action.action_id}' ({action.capability_name}) has no declared compensation action.")
                continue
            try:
                comp_cap = registry.get(spec.capability_name)
            except Exception as exc:
                uncompensated.append(action.capability_name)
                notes.append(f"Compensation capability unavailable for '{action.action_id}': {exc}")
                continue
            if not comp_cap.executor_command_template:
                uncompensated.append(action.capability_name)
                notes.append(f"Compensation '{spec.capability_name}' has no executor command template.")
                continue
            params = {}
            for target, source in spec.parameter_mapping.items():
                source_name = source[1:-1] if source.startswith("{") and source.endswith("}") else (
                    source[1:] if source.startswith("$") else source
                )
                if source_name not in action.parameters:
                    uncompensated.append(action.capability_name)
                    notes.append(f"Compensation parameter '{target}' references missing '{source_name}'.")
                    params = None
                    break
                params[target] = action.parameters[source_name]
            if params is None:
                continue
            command = []
            for token in comp_cap.executor_command_template:
                rendered = token
                for key, value in params.items():
                    rendered = rendered.replace(f"{{{key}}}", str(value)).replace(f"${key}", str(value))
                command.append(rendered)
            ledger.append_record(
                recovery_mod.LedgerEventType.COMPENSATION_TRIGGERED,
                {"step_id": action.action_id, "compensation_id": spec.compensation_id},
            )
            result = self.sandbox.execute_argv_pipeline([command], timeout_seconds=action.timeout_seconds)
            ledger.append_record(
                recovery_mod.LedgerEventType.COMPENSATION_EXECUTED,
                {"step_id": action.action_id, "exit_code": result.returncode, "parameters": params},
            )
            if result.returncode != 0:
                uncompensated.append(action.capability_name)
                notes.append(result.stderr or f"Compensation for '{action.action_id}' exited {result.returncode}.")
                break
            compensated += 1

        if uncompensated:
            if session.current_state != SessionState.FAILED:
                session.transition_to(SessionState.FAILED)
            return recovery_mod.SagaRecoveryReport(
                status=recovery_mod.RecoveryStatus.CONTAINMENT_FAILED,
                compensated_steps_count=compensated,
                failed_compensation_step_id=(steps_to_undo[-1].step_id if steps_to_undo else None),
                uncompensated_capabilities=uncompensated,
                damage_mitigation_notes=notes,
            )
        session.transition_to(SessionState.ROLLED_BACK)
        return recovery_mod.SagaRecoveryReport(
            status=recovery_mod.RecoveryStatus.ROLLED_BACK,
            compensated_steps_count=compensated,
            damage_mitigation_notes=notes,
        )

    recovery_mod.SagaRecoveryManager.execute_saga_rollback = hardened_saga_rollback

    # ---------------------------------------------------------
    # IR-search judge cache binds plan + world + registry + judge identity.
    # ---------------------------------------------------------
    raw_run_judge = ir_search_mod.EpistemicPlanSearch._run_judge

    def hardened_run_judge(self, plan, observed_world_state):
        if self.judge is None:
            return None
        if isinstance(observed_world_state, dict):
            facts = list(observed_world_state.values())
        else:
            facts = list(observed_world_state or [])
        world_payload = sorted(
            (fact.fact_key, fact.truth.value, getattr(fact, "updated_at", None))
            for fact in facts
        )
        registry_hash = self.registry.compute_registry_hash()
        judge_identity = f"{type(self.judge).__module__}.{type(self.judge).__qualname__}:{getattr(self.judge, 'model', '')}"
        cache_key = _stable_digest({
            "plan": plan.compute_hash(),
            "world": world_payload,
            "registry": registry_hash,
            "judge": judge_identity,
        })
        existing = self._judge_cache.get(cache_key)
        if existing is not None:
            return existing

        # Temporarily isolate the legacy plan-only cache so it cannot return a
        # verdict produced for another world state.
        legacy_cache = self._judge_cache
        self._judge_cache = {}
        try:
            verdict = raw_run_judge(self, plan, observed_world_state)
        finally:
            self._judge_cache = legacy_cache
        if verdict is not None:
            self._judge_cache[cache_key] = verdict
        return verdict

    ir_search_mod.EpistemicPlanSearch._run_judge = hardened_run_judge

    # ---------------------------------------------------------
    # LLM judge truth defaults and outer timeout boundary.
    # ---------------------------------------------------------
    judges_mod.JudgeVerdict.model_fields["falsifiable_criteria"].default = False
    raw_ensemble_evaluate = judges_mod.EnsembleJudge.evaluate

    async def hardened_ensemble_evaluate(
        self,
        plan_ir,
        goal_description="",
        registry=None,
        observed_world_state=None,
        timeout=30.0,
    ):
        if timeout <= 0:
            return judges_mod.JudgeVerdict(
                verdict="UNKNOWN", blockers=["judge timeout must be > 0"],
                falsifiable_criteria=False,
            )
        async def run_one(judge):
            try:
                return await asyncio.wait_for(
                    judge.evaluate(
                        plan_ir,
                        goal_description=goal_description,
                        registry=registry,
                        observed_world_state=observed_world_state,
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return judges_mod.JudgeVerdict(
                    verdict="UNKNOWN", feasibility_0_100=0.0, confidence=0.0,
                    blockers=[f"{type(judge).__name__} exceeded {timeout:.3f}s timeout"],
                    provider="timeout", model=type(judge).__name__,
                    falsifiable_criteria=False,
                )
            except Exception as exc:
                return judges_mod.JudgeVerdict(
                    verdict="UNKNOWN", feasibility_0_100=0.0, confidence=0.0,
                    blockers=[f"{type(judge).__name__} failed: {exc}"],
                    provider="error", model=type(judge).__name__,
                    falsifiable_criteria=False,
                )
        verdicts = await asyncio.gather(*(run_one(judge) for judge in self.judges))
        if not verdicts:
            return judges_mod.JudgeVerdict(falsifiable_criteria=False)
        feasibilities = sorted(v.feasibility_0_100 for v in verdicts)
        median = feasibilities[(len(feasibilities) - 1) // 2]
        representative = min(verdicts, key=lambda v: abs(v.feasibility_0_100 - median))
        return judges_mod.JudgeVerdict(
            verdict=representative.verdict,
            feasibility_0_100=median,
            confidence=round(sum(v.confidence for v in verdicts) / len(verdicts), 2),
            blockers=sorted({b for v in verdicts for b in v.blockers}),
            falsifiable_criteria=all(v.falsifiable_criteria for v in verdicts),
            summary=f"Ensemble median feasibility: {median:.1f} across {len(verdicts)} judges.",
            token_usage={
                "prompt_tokens": sum(v.token_usage.get("prompt_tokens", 0) for v in verdicts),
                "completion_tokens": sum(v.token_usage.get("completion_tokens", 0) for v in verdicts),
                "cost_usd": round(sum(v.token_usage.get("cost_usd", 0.0) for v in verdicts), 6),
            },
            provider="ensemble",
            model=f"{len(verdicts)}_judges",
            individual_verdicts=verdicts,
        )

    judges_mod.EnsembleJudge.evaluate = hardened_ensemble_evaluate

    # ---------------------------------------------------------
    # Legacy public engine hardening and external-judge provenance.
    # ---------------------------------------------------------
    raw_assess = ns["assess"]
    raw_run = ns["run"]
    raw_release = ns["release"]
    raw_finish = ns["finish"]
    raw_execute_plan = ns["execute_plan"]
    raw_execute_plan_sync = ns["execute_plan_sync"]
    raw_record_judge = ns["record_judge"]
    raw_judge = ns["judge"]
    raw_judge_ensemble = ns["judge_ensemble"]
    raw_assess_candidates = ns["assess_candidates"]
    ns["_raw_assess"] = raw_assess
    ns["_raw_release"] = raw_release
    ns["_raw_finish"] = raw_finish
    ns["_raw_execute_plan"] = raw_execute_plan

    def plans_dir_for(session, plans_dir=None):
        if plans_dir is not None:
            return Path(plans_dir)
        if isinstance(session, dict) and session.get("plans_dir"):
            return Path(session["plans_dir"])
        return Path(ns["DEFAULT_PLANS_DIR"])

    def load_state(session, plans_dir=None):
        if isinstance(session, dict):
            return session
        try:
            return ns["_load_session"](plans_dir_for(session, plans_dir), session)
        except Exception:
            return None

    def best_plan_text(session, plans_dir=None):
        state = load_state(session, plans_dir)
        return _round_plan_text(state or {})

    ns["_best_plan_text"] = best_plan_text

    def persist_status(session, status_value, plans_dir=None, **metadata):
        state = load_state(session, plans_dir)
        if not isinstance(state, dict):
            return
        state["status"] = status_value
        state.update(metadata)
        ns["_save_session"](plans_dir_for(state, plans_dir), state)

    def hardened_assess(
        session, plan_text, *, note=None, addressed=None, plans_dir=None,
        require_execution_contract=False, run_probe=False, probe_cwd=None,
        execution_evidence=None, require_execution_evidence=False,
        conflicts=None, require_conflict_free=False,
    ):
        result = raw_assess(
            session, plan_text, note=note, addressed=addressed, plans_dir=plans_dir,
            require_execution_contract=require_execution_contract, run_probe=run_probe,
            probe_cwd=probe_cwd, execution_evidence=execution_evidence,
            require_execution_evidence=require_execution_evidence,
            conflicts=conflicts, require_conflict_free=require_conflict_free,
        )
        if result.get("status") != "converged":
            return result
        cwd = Path(probe_cwd or Path.cwd())
        verified = ns["verify"](plan_text)
        grounded = ns["ground_check"](plan_text, cwd=cwd)
        simulated = ns["simulate"](plan_text, initial_state=set(grounded.get("verified", [])))
        clean = bool(verified.get("ok") and grounded.get("ok") and simulated.get("executable_plan"))
        if require_execution_contract:
            clean = clean and bool((result.get("execution_contract") or {}).get("ok"))
            if run_probe and (result.get("probe") or {}).get("configured"):
                clean = clean and bool((result.get("probe") or {}).get("ok"))
        if require_execution_evidence:
            clean = clean and bool((result.get("execution_trace") or {}).get("ok"))
        result["clean_convergence"] = clean
        result["convergence_checks"] = {
            "zero_critiques": not bool(result.get("critiques")),
            "verify_ok": bool(verified.get("ok")),
            "ground_ok": bool(grounded.get("ok")),
            "sim_ok": bool(simulated.get("executable_plan")),
        }
        result["open_critiques"] = len(result.get("critiques") or [])
        if clean:
            result["convergence_quality"] = "hard-gates-clean"
            persist_status(session, "converged", plans_dir,
                           convergence_quality="hard-gates-clean",
                           open_critiques=result["open_critiques"])
            return result
        result.update({
            "status": "plateaued", "continue": False, "requires_revision": True,
            "convergence_quality": "stopped-with-hard-gate-failure",
        })
        persist_status(session, "plateaued", plans_dir,
                       convergence_quality="stopped-with-hard-gate-failure",
                       open_critiques=result["open_critiques"])
        return result

    def hardened_run(objective, draft_plan, *, plans_dir=None, max_rounds=None, note=None):
        if max_rounds is None:
            max_rounds = ns["DEFAULT_MAX_ROUNDS"]
        session = ns["start"](objective, plans_dir=plans_dir, max_rounds=max_rounds)
        hardened_assess(session, draft_plan, note=note, plans_dir=plans_dir)
        return session

    async def hardened_judge(plan_text: str, objective: str = "", **kwargs):
        result = await raw_judge(plan_text, objective, **kwargs)
        if isinstance(result, dict) and result.get("ok") and result.get("source") == "external_llm":
            public_result = {k: v for k, v in result.items() if k != "_judge_attestation"}
            result = dict(result)
            result["_judge_attestation"] = _sign_judge_attestation(
                _text_hash(plan_text),
                str(result.get("provider") or "unknown"),
                str(result.get("model") or "unknown"),
                _stable_digest(public_result),
                time.time(),
            )
        return result

    def hardened_record_judge(session, verdict, *, round_version=None, plans_dir=None):
        pdir = plans_dir_for(session, plans_dir)
        sid = session if isinstance(session, str) else session.get("session_id", "default")
        with ns["session_lock"](pdir, sid):
            state = ns["_load_session"](pdir, session) if isinstance(session, str) else session
            version = round_version if round_version is not None else state.get("best_version") or len(state.get("rounds", []))
            plan_text = _round_plan_text(state, version)
            expected_hash = _text_hash(plan_text)
            entry = dict(verdict)
            attested = _verify_judge_attestation(entry.get("_judge_attestation"), expected_hash)
            entry["plan_hash"] = expected_hash
            entry["external_attested"] = attested
            if attested:
                entry["external"] = True
                entry["source"] = "external_llm"
            else:
                entry["external"] = False
                if entry.get("source") == "external_llm":
                    entry["source"] = "unattested"
                entry.pop("_judge_attestation", None)
            return raw_record_judge(state, entry, round_version=version, plans_dir=pdir)

    async def hardened_judge_ensemble(session, plan_text, objective, *, n=3, plans_dir=None, **kwargs):
        pdir = plans_dir_for(session, plans_dir)
        state = ns["_load_session"](pdir, session) if isinstance(session, str) else session
        current_version = state.get("best_version") or len(state.get("rounds", []))
        current_text = _round_plan_text(state, current_version) or plan_text
        current_hash = _text_hash(current_text)
        original_log = list(state.get("judge_log", []))
        state["judge_log"] = [
            vote for vote in original_log
            if vote.get("round_version") == current_version and vote.get("plan_hash") == current_hash
        ]
        try:
            entry = await raw_judge_ensemble(state, plan_text, objective, n=n, plans_dir=pdir, **kwargs)
        finally:
            generated = list(state.get("judge_log", []))
            new_entries = [item for item in generated if item not in state.get("judge_log", [])[:0]]
            state["judge_log"] = original_log
        # raw_judge_ensemble records through the hardened record_judge global;
        # reload the persisted current-version entry and preserve historical log.
        persisted = ns["_load_session"](pdir, state["session_id"])
        persisted_new = [
            vote for vote in persisted.get("judge_log", [])
            if vote.get("round_version") == current_version and vote.get("plan_hash") == current_hash
        ]
        state["judge_log"] = original_log + [vote for vote in persisted_new if vote not in original_log]
        ns["_save_session"](pdir, state)
        if isinstance(entry, dict):
            entry["votes"] = [
                vote for vote in entry.get("votes", [])
                if vote.get("round_version") in (None, current_version)
                and (vote.get("plan_hash") in (None, current_hash))
            ]
        return entry

    def hardened_release(
        session, *, min_score=90.0, require_judge=True,
        require_external_judge=False, require_execution_contract=False,
        execution_cwd=None, execution_evidence=None,
        require_execution_evidence=False, conflicts=None,
        require_conflict_free=False, plans_dir=None,
    ):
        pdir = plans_dir_for(session, plans_dir)
        sid = session if isinstance(session, str) else session.get("session_id", "default")
        with ns["session_lock"](pdir, sid):
            state = ns["_load_session"](pdir, session) if isinstance(session, str) else session
            snapshot = {
                key: copy.deepcopy(state.get(key))
                for key in (
                    "committed_version", "committed_score", "committed_at",
                    "committed_plan_hash", "release_gate",
                )
            }
            gate = raw_release(
                state, min_score=min_score, require_judge=require_judge,
                require_external_judge=require_external_judge,
                require_execution_contract=require_execution_contract,
                execution_cwd=execution_cwd, execution_evidence=execution_evidence,
                require_execution_evidence=require_execution_evidence,
                conflicts=conflicts, require_conflict_free=require_conflict_free,
                plans_dir=pdir,
            )
            if execution_cwd is not None:
                text = _round_plan_text(state)
                cwd = Path(execution_cwd).resolve()
                grounded = ns["ground_check"](text, cwd=cwd) if text else {
                    "ok": False, "missing": ["no best plan"], "verified": []
                }
                simulated = ns["simulate"](
                    text, initial_state=set(grounded.get("verified", []))
                ) if text else {"executable_plan": False, "problems": ["no best plan"]}
                gate["execution_cwd_checks"] = {
                    "cwd": str(cwd),
                    "ground_ok": bool(grounded.get("ok")),
                    "sim_ok": bool(simulated.get("executable_plan")),
                    "missing": grounded.get("missing", []),
                    "simulation_problems": simulated.get("problems", []),
                }
                if not grounded.get("ok"):
                    gate["ok"] = False
                    gate.setdefault("problems", []).append(
                        f"execution_cwd grounding failed: {grounded.get('missing', [])[:5]}"
                    )
                if not simulated.get("executable_plan"):
                    gate["ok"] = False
                    gate.setdefault("problems", []).append(
                        f"execution_cwd simulation failed: {simulated.get('problems', [])[:5]}"
                    )
            if not gate.get("ok"):
                for key, value in snapshot.items():
                    state[key] = value
                state["release_gate"] = gate
                ns["_save_session"](pdir, state)
            return gate

    def hardened_finish(
        session, *, verdict="converged", plans_dir=None, require_release=True,
        min_score=90.0, require_judge=True, require_external_judge=False,
        require_execution_contract=False, execution_cwd=None,
        execution_evidence=None, require_execution_evidence=False,
        conflicts=None, require_conflict_free=False,
    ):
        if require_release:
            gate = hardened_release(
                session, min_score=min_score, require_judge=require_judge,
                require_external_judge=require_external_judge,
                require_execution_contract=require_execution_contract,
                execution_cwd=execution_cwd, execution_evidence=execution_evidence,
                require_execution_evidence=require_execution_evidence,
                conflicts=conflicts, require_conflict_free=require_conflict_free,
                plans_dir=plans_dir,
            )
            if not gate.get("ok"):
                return {
                    "ok": False,
                    "status": (load_state(session, plans_dir) or {}).get("status"),
                    "error": "release gate failed",
                    "release_gate": gate,
                }
            result = raw_finish(
                session, verdict=verdict, plans_dir=plans_dir, require_release=False,
            )
            if isinstance(result, dict):
                result["release_gate"] = gate
            return result
        return raw_finish(session, verdict=verdict, plans_dir=plans_dir, require_release=False)

    def is_async_handler(handler):
        return bool(
            inspect.iscoroutinefunction(handler)
            or inspect.iscoroutinefunction(getattr(handler, "__call__", None))
        )

    async def hardened_execute_plan(
        plan_text, task_handlers=None, *, dry_run=False,
        continue_on_error=False, timeout_per_task=None, context=None,
    ):
        handlers = dict(task_handlers or {})
        nodes = list((ns["plan_dag"](plan_text) or {}).get("nodes", []))
        if not dry_run:
            missing = [task_id for task_id in nodes if task_id not in handlers]
            if missing:
                return {
                    "ok": False,
                    "error": f"missing task handlers for tasks {missing}; refusing synthetic success",
                    "failed_task": missing[0], "executed_tasks": [], "recovered": False,
                }
            sync_handlers = [task_id for task_id, handler in handlers.items() if not is_async_handler(handler)]
            if sync_handlers:
                return {
                    "ok": False,
                    "error": f"synchronous task handlers are not allowed: {sync_handlers}",
                    "failed_task": sync_handlers[0], "executed_tasks": [], "recovered": False,
                }
        effective_timeout = 60.0 if timeout_per_task is None else float(timeout_per_task)
        if effective_timeout <= 0:
            raise ValueError("timeout_per_task must be > 0")
        return await raw_execute_plan(
            plan_text, task_handlers=handlers, dry_run=dry_run,
            continue_on_error=continue_on_error, timeout_per_task=effective_timeout,
            context=context,
        )

    def hardened_execute_plan_sync(
        plan_text, task_handlers=None, *, dry_run=False,
        continue_on_error=False, timeout_per_task=None, context=None,
    ):
        def run_sync():
            return asyncio.run(hardened_execute_plan(
                plan_text, task_handlers=task_handlers, dry_run=dry_run,
                continue_on_error=continue_on_error, timeout_per_task=timeout_per_task,
                context=context,
            ))
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run_sync()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(run_sync).result()

    # Same-model/same-thinking candidate ranking is installed in plan_mode too.
    from .self_verification import (
        DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        DEFAULT_SELF_VERIFICATION_PIVOTS,
        DEFAULT_VERIFIER_MAX_CALLS,
        DEFAULT_VERIFIER_MAX_WORKERS,
        DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
        ProbabilisticSelfVerifier,
        resolve_implementation_model,
        resolve_implementation_thinking,
    )

    def hardened_assess_candidates(
        session, drafts, *, notes=None, plans_dir=None, verifier=None,
        implementation_model=None, implementation_thinking=None,
        n_evaluations=DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        pivots=DEFAULT_SELF_VERIFICATION_PIVOTS,
        request_timeout_seconds=DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
        max_workers=DEFAULT_VERIFIER_MAX_WORKERS,
        max_verifier_calls=DEFAULT_VERIFIER_MAX_CALLS,
    ):
        if not drafts:
            raise ValueError("drafts must be non-empty")
        if len(drafts) == 1:
            result = raw_assess_candidates(session, drafts, notes=notes, plans_dir=plans_dir)
            result["selection_method"] = "deterministic-single-candidate"
            return result
        checked = []
        eligible = []
        for index, draft in enumerate(drafts):
            verified = ns["verify"](draft)
            grounded = ns["ground_check"](draft)
            simulated = ns["simulate"](draft, initial_state=set(grounded.get("verified", [])))
            hard_pass = bool(verified.get("ok") and grounded.get("ok") and simulated.get("executable_plan"))
            checked.append({
                "candidate": index,
                "verify_ok": bool(verified.get("ok")),
                "feasibility_ok": bool(grounded.get("ok")),
                "sim_ok": bool(simulated.get("executable_plan")),
                "hard_pass": hard_pass,
            })
            if hard_pass:
                eligible.append(index)
        if not eligible:
            eligible = list(range(len(drafts)))
        if len(eligible) == 1:
            chosen = eligible[0]
            result = hardened_assess(
                session, drafts[chosen],
                note=(notes or [None] * len(drafts))[chosen], plans_dir=plans_dir,
            )
            result.update({
                "selection_method": "deterministic-prefilter-single",
                "selected_candidate": chosen,
                "candidate_checks": checked,
                "candidates_scored": len(drafts),
            })
            return result
        state = load_state(session, plans_dir)
        active_model = resolve_implementation_model(implementation_model, session=state)
        active_thinking = resolve_implementation_thinking(implementation_thinking, session=state)
        if not active_model:
            result = raw_assess_candidates(session, drafts, notes=notes, plans_dir=plans_dir)
            result.update({
                "selection_method": "deterministic-fallback-no-model",
                "self_verification_available": False,
                "candidate_checks": checked,
            })
            return result
        selector = verifier or ProbabilisticSelfVerifier()
        try:
            soft = selector.select(
                problem=str((state or {}).get("objective") or "Select the best candidate plan"),
                candidates=[drafts[i] for i in eligible],
                model=active_model,
                thinking_profile=active_thinking,
                n_evaluations=n_evaluations,
                pivots=pivots,
                request_timeout_seconds=request_timeout_seconds,
                max_workers=max_workers,
                max_verifier_calls=max_verifier_calls,
            )
            chosen = eligible[soft.selected_index]
            result = hardened_assess(
                session, drafts[chosen],
                note=(notes or [None] * len(drafts))[chosen], plans_dir=plans_dir,
            )
            result.update({
                "selection_method": "inherited-same-model-same-thinking-self-verification",
                "selected_candidate": chosen,
                "implementation_model": active_model,
                "generator_model": active_model,
                "verifier_model": active_model,
                "implementation_thinking": dict(active_thinking),
                "generator_thinking": dict(active_thinking),
                "verifier_thinking": dict(active_thinking),
                "is_self_verification": True,
                "is_same_thinking": True,
                "candidate_checks": checked,
                "candidates_scored": len(drafts),
                "estimated_verifier_calls": getattr(soft, "estimated_calls", None),
            })
            return result
        except Exception as exc:
            result = raw_assess_candidates(session, drafts, notes=notes, plans_dir=plans_dir)
            result.update({
                "selection_method": "deterministic-fallback",
                "implementation_model": active_model,
                "implementation_thinking": dict(active_thinking),
                "self_verification_available": False,
                "self_verification_error": f"{type(exc).__name__}: {exc}",
                "candidate_checks": checked,
            })
            return result

    ns.update({
        "assess": hardened_assess,
        "run": hardened_run,
        "judge": hardened_judge,
        "record_judge": hardened_record_judge,
        "judge_ensemble": hardened_judge_ensemble,
        "release": hardened_release,
        "finish": hardened_finish,
        "execute_plan": hardened_execute_plan,
        "execute_plan_sync": hardened_execute_plan_sync,
        "assess_candidates": hardened_assess_candidates,
    })
