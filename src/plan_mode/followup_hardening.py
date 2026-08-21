"""Second-pass hardening discovered after the initial PR5 GREEN matrix.

The fixes here patch already-imported public classes/modules so direct imports
and the top-level API share the same fail-closed behavior without duplicating
the large legacy implementation files.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
from pathlib import Path
from typing import Any, MutableMapping


def install_followup_hardening(ns: MutableMapping[str, Any]) -> None:
    _harden_execution_trace()
    _harden_provider_judges()
    _harden_cordis(ns)
    _harden_release_judge_binding(ns)


def _harden_execution_trace() -> None:
    from . import execution_trace as trace

    def safe_trace_exit_errors(contract: Any, parsed: Any) -> list[str]:
        errors: list[str] = []
        for criterion in contract.exit_criteria:
            raw_task = criterion.get("task") or criterion.get("task_id") or 0
            try:
                task_id = int(raw_task)
            except (TypeError, ValueError):
                errors.append(f"exit criterion task id is invalid: {raw_task!r}")
                continue
            entry = next((task for task in parsed.tasks if task.task_id == task_id), None)
            if entry is None:
                errors.append(f"exit criterion for task {task_id}: no task evidence")
                continue
            expected_command = trace._as_str_list(criterion.get("command"))
            result = next((cmd for cmd in entry.commands if cmd.command == expected_command), None)
            if result is None:
                errors.append(
                    f"exit criterion for task {task_id}: command {criterion.get('command')!r} was not recorded"
                )
            else:
                errors.extend(trace._command_matches(result, criterion))
        return errors

    trace._trace_exit_errors = safe_trace_exit_errors


def _harden_provider_judges() -> None:
    from .judges import BaseLLMJudge

    original = BaseLLMJudge.evaluate
    if getattr(original, "_pr5_hardened", False):
        return

    async def evaluate(self: BaseLLMJudge, plan_ir: Any, goal_description: str = "",
                       registry: Any = None, observed_world_state: Any = None,
                       timeout: float = 30.0):
        import time
        t0 = time.time()
        if timeout <= 0:
            return self._unknown(
                f"{self.PROVIDER_NAME} total timeout must be > 0",
                (time.time() - t0) * 1000.0,
            )
        if self.mock_error is not None:
            return self._unknown(
                f"{self.PROVIDER_NAME} API error: {self.mock_error}",
                (time.time() - t0) * 1000.0,
            )
        if self.mock_response is not None:
            response = dict(self.mock_response)
            prompt_tokens = int(response.pop("prompt_tokens", 800))
            completion_tokens = int(response.pop("completion_tokens", 200))
            return self._verdict_from_payload(
                response,
                (time.time() - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        if not self.api_key and self.http_client is None:
            return self._unknown(
                f"Provider credential/API key not configured for {self.PROVIDER_NAME}",
                (time.time() - t0) * 1000.0,
            )
        try:
            return await asyncio.wait_for(
                self._dispatch_api_request(plan_ir, goal_description, float(timeout), t0),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            return self._unknown(
                f"{self.PROVIDER_NAME} total timeout exceeded after {float(timeout):.3f}s",
                (time.time() - t0) * 1000.0,
            )
        except Exception as exc:
            return self._unknown(
                f"{self.PROVIDER_NAME} HTTP dispatch failed: {exc}",
                (time.time() - t0) * 1000.0,
            )

    evaluate._pr5_hardened = True  # type: ignore[attr-defined]
    BaseLLMJudge.evaluate = evaluate  # type: ignore[assignment]


def _harden_cordis(ns: MutableMapping[str, Any]) -> None:
    from . import cordis

    Context = cordis.Context
    if getattr(Context.effect, "_pr5_hardened", False):
        return

    def _sync_inverse(inv: Any) -> None:
        if inspect.iscoroutinefunction(inv):
            raise RuntimeError("async inverse requires async_dispose(); sync rollback cannot attest completion")
        result = inv()
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise RuntimeError("inverse returned an awaitable during sync rollback")

    def _raise_cleanup(errors: list[BaseException]) -> None:
        if errors:
            detail = "; ".join(f"{type(err).__name__}: {err}" for err in errors[:4])
            raise RuntimeError(f"Rollback failed: {detail}")

    def effect(self: Any, callback: Any):
        with self._lock:
            if self._disposed:
                raise RuntimeError("Cannot execute effect on a disposed context")
            if inspect.iscoroutinefunction(callback) or inspect.iscoroutine(callback) or inspect.isawaitable(callback):
                raise TypeError(
                    "Cannot pass an async function, coroutine, or awaitable to sync ctx.effect(). "
                    "Use 'await ctx.async_effect(...)' instead."
                )

            armed = [True]
            inverses: list[Any] = []
            result = callback()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError(
                    "ctx.effect() callback returned an awaitable/coroutine. "
                    "Use 'await ctx.async_effect(...)' instead."
                )
            if inspect.isgenerator(result) or hasattr(result, "__next__"):
                try:
                    for inv in result:
                        if not armed[0]:
                            break
                        if callable(inv):
                            inverses.append(inv)
                except Exception as exc:
                    cleanup_errors: list[BaseException] = []
                    for inv in reversed(inverses):
                        try:
                            _sync_inverse(inv)
                        except BaseException as cleanup:
                            cleanup_errors.append(cleanup)
                    if cleanup_errors:
                        raise RuntimeError(
                            f"effect iterator failed ({exc}); rollback also failed: {cleanup_errors[0]}"
                        ) from exc
                    raise
            elif callable(result):
                inverses.append(result)

            def dispose_one() -> None:
                with self._lock:
                    if not armed[0]:
                        return
                    armed[0] = False
                    errors: list[BaseException] = []
                    for inv in reversed(inverses):
                        try:
                            _sync_inverse(inv)
                        except BaseException as exc:
                            errors.append(exc)
                    _raise_cleanup(errors)

            async def dispose_one_async() -> None:
                with self._lock:
                    if not armed[0]:
                        return
                    armed[0] = False
                    errors: list[BaseException] = []
                    for inv in reversed(inverses):
                        try:
                            await cordis._run_inv_async(inv)
                        except BaseException as exc:
                            errors.append(exc)
                    _raise_cleanup(errors)

            pair = (dispose_one, dispose_one_async)
            self._inverses.append(pair)
            if self.parent is not None:
                self.parent._inverses.append(pair)
            return dispose_one

    async def async_effect(self: Any, callback: Any, *, token: Any = None):
        if self._disposed:
            raise RuntimeError("Cannot execute effect on a disposed context")
        armed = [True]
        inverses: list[Any] = []
        result = callback() if callable(callback) else callback
        if inspect.isawaitable(result):
            result = await result

        async def rollback_collected() -> None:
            errors: list[BaseException] = []
            for inv in reversed(inverses):
                try:
                    await cordis._run_inv_async(inv)
                except BaseException as exc:
                    errors.append(exc)
            _raise_cleanup(errors)

        if hasattr(result, "__anext__") or inspect.isasyncgen(result):
            try:
                async for inv in result:
                    if not armed[0] or (token and token.is_cancelled):
                        break
                    if callable(inv):
                        inverses.append(inv)
            except Exception:
                await rollback_collected()
                raise
        elif callable(result):
            inverses.append(result)
        elif inspect.isgenerator(result) or hasattr(result, "__next__"):
            try:
                for inv in result:
                    if not armed[0] or (token and token.is_cancelled):
                        break
                    if callable(inv):
                        inverses.append(inv)
            except Exception:
                await rollback_collected()
                raise

        def dispose_one() -> None:
            with self._lock:
                if not armed[0]:
                    return
                armed[0] = False
                errors: list[BaseException] = []
                for inv in reversed(inverses):
                    try:
                        _sync_inverse(inv)
                    except BaseException as exc:
                        errors.append(exc)
                _raise_cleanup(errors)

        async def dispose_one_async() -> None:
            with self._lock:
                if not armed[0]:
                    return
                armed[0] = False
                errors: list[BaseException] = []
                for inv in reversed(inverses):
                    try:
                        await cordis._run_inv_async(inv)
                    except BaseException as exc:
                        errors.append(exc)
                _raise_cleanup(errors)

        pair = (dispose_one, dispose_one_async)
        self._inverses.append(pair)
        if self.parent is not None:
            self.parent._inverses.append(pair)
        return dispose_one

    def dispose(self: Any) -> None:
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            pending = list(reversed(self._inverses))
            self._inverses.clear()
            errors: list[BaseException] = []
            for dispose_sync, _ in pending:
                try:
                    dispose_sync()
                except BaseException as exc:
                    errors.append(exc)
            _raise_cleanup(errors)

    async def async_dispose(self: Any) -> None:
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            pending = list(reversed(self._inverses))
            self._inverses.clear()
            errors: list[BaseException] = []
            for dispose_sync, dispose_async in pending:
                try:
                    if dispose_async is not None:
                        await dispose_async()
                    elif dispose_sync is not None:
                        dispose_sync()
                except BaseException as exc:
                    errors.append(exc)
            _raise_cleanup(errors)

    effect._pr5_hardened = True  # type: ignore[attr-defined]
    Context.effect = effect
    Context.async_effect = async_effect
    Context.dispose = dispose
    Context.async_dispose = async_dispose

    async def execute_plan(plan_text: str, task_handlers: dict[int, Any] | None = None, *,
                           dry_run: bool = False, continue_on_error: bool = False,
                           timeout_per_task: float | None = None, context: Any = None):
        ctx = context or ns["get_root_context"]().derive(name="plan_execution")
        dag = ns["plan_dag"](plan_text)
        nodes = dag["nodes"]
        task_handlers = task_handlers or {}
        executed_tasks: list[int] = []
        task_results: dict[int, Any] = {}

        try:
            for task_id in nodes:
                unsatisfied = [dep for dep in dag["edges"].get(task_id, []) if dep not in executed_tasks]
                if unsatisfied:
                    raise RuntimeError(
                        f"Task {task_id} blocked: dependencies not completed: {unsatisfied}"
                    )
                task_ctx = ctx.derive(name=f"task_{task_id}")
                handler = task_handlers.get(task_id)
                if dry_run:
                    task_results[task_id] = {"status": "dry_run", "task": task_id}
                elif handler:
                    result = handler(task_ctx)
                    if inspect.isawaitable(result):
                        result = (
                            await asyncio.wait_for(result, timeout=timeout_per_task)
                            if timeout_per_task is not None
                            else await result
                        )
                    task_results[task_id] = result
                else:
                    task_results[task_id] = {"status": "success", "task": task_id}
                executed_tasks.append(task_id)
            return {
                "ok": True,
                "executed_tasks": executed_tasks,
                "results": task_results,
                "recovered": False,
            }
        except Exception as exc:
            failed_task = task_id if "task_id" in locals() else (nodes[0] if nodes else None)
            if continue_on_error:
                return {
                    "ok": False,
                    "error": str(exc),
                    "failed_task": failed_task,
                    "executed_tasks": executed_tasks,
                    "recovered": False,
                }
            try:
                await ctx.async_dispose()
            except Exception as cleanup:
                return {
                    "ok": False,
                    "error": str(exc),
                    "failed_task": failed_task,
                    "executed_tasks": executed_tasks,
                    "recovered": False,
                    "recovery_error": str(cleanup),
                    "recovery_message": "Rollback was attempted but did not complete successfully.",
                }
            return {
                "ok": False,
                "error": str(exc),
                "failed_task": failed_task,
                "executed_tasks": executed_tasks,
                "recovered": True,
                "recovery_message": "All intermediate mutations rolled back in LIFO order via Cordis accumulator",
            }

    ns["execute_plan"] = execute_plan


def _harden_release_judge_binding(ns: MutableMapping[str, Any]) -> None:
    base_release = ns["release"]
    if getattr(base_release, "_pr5_hash_hardened", False):
        return

    def release(session: dict[str, Any] | str, **kwargs: Any):
        plans_dir = Path(kwargs.get("plans_dir")) if kwargs.get("plans_dir") else (
            Path(session.get("plans_dir"))
            if isinstance(session, dict) and session.get("plans_dir")
            else ns["DEFAULT_PLANS_DIR"]
        )
        sid = session if isinstance(session, str) else session.get("session_id", "default")
        with ns["session_lock"](plans_dir, sid):
            state = ns["_load_session"](plans_dir, session) if isinstance(session, str) else session
            original_log = copy.deepcopy(state.get("judge_log", []))
            rounds = state.get("rounds") or []
            best_ver = state.get("best_version")
            current_hash = None
            if isinstance(best_ver, int) and 1 <= best_ver <= len(rounds):
                current_hash = __import__("hashlib").sha256(
                    str(rounds[best_ver - 1].get("plan_text", "")).encode("utf-8")
                ).hexdigest()

            if kwargs.get("require_judge", True) and current_hash is not None:
                filtered: list[dict[str, Any]] = []
                for entry in original_log:
                    if entry.get("round_version") == best_ver:
                        recorded_hash = entry.get("plan_hash")
                        if recorded_hash is None and len(rounds) > 1:
                            continue
                        if recorded_hash is not None and recorded_hash != current_hash:
                            continue
                    filtered.append(entry)
                state["judge_log"] = filtered

            try:
                result = base_release(state, **kwargs)
            finally:
                state["judge_log"] = original_log
                ns["_save_session"](plans_dir, state)
            return result

    release._pr5_hash_hardened = True  # type: ignore[attr-defined]
    ns["release"] = release
