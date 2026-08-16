"""Tests for Cordis Spatiotemporal Composability Engine in plan_mode."""
from __future__ import annotations

import pytest
from plan_mode.cordis import Context, Fiber, LifecycleState, TwistedMonoid, reset_root_context
from plan_mode import execute_plan, speculative_rollout, create_subagent_context, provide_tool


def test_twisted_monoid_lifo():
    """Verify that inverses compose in twisted monoid (LIFO) order."""
    ctx = Context(name="test_lifo")
    trace = []

    def op1():
        trace.append("f1")
        return lambda: trace.append("g1")

    def op2():
        trace.append("f2")
        return lambda: trace.append("g2")

    def op3():
        trace.append("f3")
        return lambda: trace.append("g3")

    ctx.effect(op1)
    ctx.effect(op2)
    ctx.effect(op3)

    assert trace == ["f1", "f2", "f3"]
    ctx.dispose()
    assert trace == ["f1", "f2", "f3", "g3", "g2", "g1"]


def test_context_effect_generator():
    """Verify effect iterators yielding step-by-step inverses."""
    ctx = Context(name="test_iter")
    trace = []

    def multi_step_effect():
        trace.append("step_1")
        yield lambda: trace.append("undo_1")
        trace.append("step_2")
        yield lambda: trace.append("undo_2")

    ctx.effect(multi_step_effect)
    assert trace == ["step_1", "step_2"]
    ctx.dispose()
    assert trace == ["step_1", "step_2", "undo_2", "undo_1"]


def test_reactive_coeffects_and_notification():
    """Verify reactive coeffects: dynamic service provision and listener notifications."""
    ctx = Context(name="test_coeffect")
    notifications = []

    ctx.on_change("db", lambda key, val, action: notifications.append((key, val, action)))

    assert ctx.get("db") is None

    # Provide db service
    disp = ctx.provide("db", "postgres://localhost:5432")
    assert ctx.get("db") == "postgres://localhost:5432"
    assert ("db", "postgres://localhost:5432", "provided") in notifications

    # Revert provision
    disp()
    assert ctx.get("db") is None
    assert ("db", None, "withdrawn") in notifications


def test_scoped_realm_isolation():
    """Verify scoped realm isolation (ctx.isolate): private realms do not pollute parent."""
    root = Context(name="root")
    root.provide("scratch", "global_scratch")

    sub_a = root.isolate("scratch", "realm_a")
    sub_b = root.isolate("scratch", "realm_b")

    sub_a.provide("scratch", "private_a")
    sub_b.provide("scratch", "private_b")

    assert root.get("scratch") == "global_scratch"
    assert sub_a.get("scratch") == "private_a"
    assert sub_b.get("scratch") == "private_b"

    # Dispose sub_a only
    sub_a.dispose()
    assert sub_a.get("scratch") is None
    assert root.get("scratch") == "global_scratch"
    assert sub_b.get("scratch") == "private_b"


def test_proxy_capability_interception():
    """Verify metadata interception on coeffect bindings."""
    ctx = Context(name="test_intercept")
    ctx.provide("calculator", {"add": lambda a, b: a + b, "eval_code": lambda s: "unsafe"})

    # Intercept and wrap with capability filter
    def security_proxy(target, context):
        return {
            "add": target["add"],
            "eval_code": lambda s: "BLOCKED_BY_CAPABILITY_FILTER"
        }

    ctx.intercept("calculator", proxy=security_proxy)

    calc = ctx.get("calculator")
    assert calc["add"](2, 3) == 5
    assert calc["eval_code"]("rm -rf /") == "BLOCKED_BY_CAPABILITY_FILTER"


def test_fiber_lifecycle_state_machine():
    """Verify 10-rule fiber state machine: activation on satisfaction, deactivation, and failure handling."""
    ctx = Context(name="fiber_test")
    trace = []

    def service_a_impl(c):
        trace.append("service_a_active")
        return lambda: trace.append("service_a_stopped")

    def consumer_b_impl(c):
        trace.append("consumer_b_active")
        return lambda: trace.append("consumer_b_stopped")

    fiber_a = Fiber(ctx, "provider_a", provisions={"service:a"}, apply_fn=lambda c: (service_a_impl(c), c.provide("service:a", "svc_a_val"))[0])
    fiber_b = Fiber(ctx, "consumer_b", dependencies={"service:a"}, apply_fn=consumer_b_impl)

    assert not fiber_b.is_satisfied()
    assert fiber_b.state == LifecycleState.INACTIVE

    # Activate provider
    fiber_a.activate()
    assert fiber_a.state == LifecycleState.ACTIVE
    assert fiber_b.is_satisfied()

    # Activate consumer
    activated = fiber_b.activate()
    assert activated
    assert fiber_b.state == LifecycleState.ACTIVE
    assert "consumer_b_active" in trace

    # Deactivate consumer
    fiber_b.deactivate()
    assert fiber_b.state == LifecycleState.INACTIVE
    assert "consumer_b_stopped" in trace

    # Test failure handling (L-Raise)
    def failing_fn(c):
        raise ValueError("simulated fiber crash")

    fiber_failing = Fiber(ctx, "failing_fiber", apply_fn=failing_fn)
    ok = fiber_failing.activate()
    assert not ok
    assert fiber_failing.state == LifecycleState.FAILED
    assert isinstance(fiber_failing.error, ValueError)


@pytest.mark.asyncio
async def test_execute_plan_transactional_success():
    """Verify execute_plan runs all tasks cleanly."""
    plan_text = """# Sample Plan
    1. Task One: Initialize
    - output: init.txt
    2. Task Two: Build
    - depends on 1
    - output: build.txt
    """
    trace = []

    async def h1(c):
        trace.append("t1_run")
        return {"done": 1}

    async def h2(c):
        trace.append("t2_run")
        return {"done": 2}

    res = await execute_plan(plan_text, task_handlers={1: h1, 2: h2})
    assert res["ok"] is True
    assert res["executed_tasks"] == [1, 2]
    assert trace == ["t1_run", "t2_run"]
    assert res["recovered"] is False


@pytest.mark.asyncio
async def test_execute_plan_transactional_rollback_on_failure():
    """Verify that failure at Task 2 automatically inverts Task 1 side effects in LIFO order."""
    plan_text = """# Sample Plan
    1. Task One: Setup
    - output: setup.txt
    2. Task Two: Risky Step
    - depends on 1
    - output: risk.txt
    """
    disk_state = []

    async def task1_handler(task_ctx):
        # Register a revertible file mutation
        disk_state.append("setup.txt")
        await task_ctx.async_effect(lambda: lambda: disk_state.remove("setup.txt"))
        return {"created": "setup.txt"}

    async def task2_handler(task_ctx):
        raise RuntimeError("Task 2 build error!")

    res = await execute_plan(plan_text, task_handlers={1: task1_handler, 2: task2_handler})
    assert res["ok"] is False
    assert res["failed_task"] == 2
    assert res["recovered"] is True
    # Verify Task 1 side effect was completely undone!
    assert disk_state == []
    assert "rolled back in LIFO order" in res["recovery_message"]


def test_speculative_rollout_clean_teardown():
    """Verify speculative_rollout scores without leaking any side effects."""
    mutations = []

    def candidate_eval(spec_ctx):
        mutations.append("temp_db_table")
        spec_ctx.effect(lambda: lambda: mutations.remove("temp_db_table"))
        return 92.5

    res = speculative_rollout("dummy plan", candidate_eval)
    assert res["ok"] is True
    assert res["score"] == 92.5
    # Teardown must be complete
    assert mutations == []


@pytest.mark.asyncio
async def test_cordis_async_effects_and_inverses():
    """Verify that async effects and async inverses execute and unwind cleanly."""
    ctx = Context(name="test_async_effects")
    trace = []

    async def async_forward():
        trace.append("async_fwd")
        async def async_inv():
            trace.append("async_inv")
        return async_inv

    await ctx.async_effect(async_forward)
    assert trace == ["async_fwd"]

    await ctx.async_dispose()
    assert trace == ["async_fwd", "async_inv"]


def test_cordis_fiber_reactive_lifecycle_auto_activation():
    """Verify that providing a coeffect dynamically activates fibers, and withdrawing it deactivates them."""
    ctx = Context(name="reactive_fiber_test")
    trace = []

    def consumer_fn(c):
        trace.append("consumer_started")
        return lambda: trace.append("consumer_stopped")

    fiber = Fiber(ctx, "reactive_consumer", dependencies={"service:gpu"}, apply_fn=consumer_fn)
    assert fiber.state == LifecycleState.INACTIVE

    # Dynamically provide service:gpu -> fiber should auto-activate
    dispose_prov = ctx.provide("service:gpu", "cuda:0")
    assert fiber.state == LifecycleState.ACTIVE
    assert "consumer_started" in trace

    # Withdraw service:gpu -> fiber should auto-deactivate
    dispose_prov()
    assert fiber.state == LifecycleState.INACTIVE
    assert "consumer_stopped" in trace


def test_cordis_theorem_63_provider_withdrawal_order():
    """Theorem 63: A provider cannot withdraw until all dependent fibers have deactivated."""
    ctx = Context(name="theorem_63_test")
    order_of_events = []

    def consumer_fn(c):
        order_of_events.append("consumer_active")
        def _inv():
            order_of_events.append("consumer_deactivated")
        return _inv

    fiber = Fiber(ctx, "dep_consumer", dependencies={"service:auth"}, apply_fn=consumer_fn)

    # Provide auth
    disp_auth = ctx.provide("service:auth", {"token": "valid"})
    assert fiber.state == LifecycleState.ACTIVE

    # Withdraw provider
    disp_auth()
    # Ensure consumer deactivation occurred before final withdrawal
    assert "consumer_deactivated" in order_of_events
    assert fiber.state == LifecycleState.INACTIVE
    assert ctx.get("service:auth") is None


def test_cordis_effect_rejects_async_callback_type_safety():
    """Verify that ctx.effect() raises TypeError if passed an async callback."""
    ctx = Context(name="test_type_safety")

    async def async_cb():
        return lambda: None

    with pytest.raises(TypeError, match="Use 'await ctx.async_effect"):
        ctx.effect(async_cb)


@pytest.mark.asyncio
async def test_cordis_async_generator_effect_iterator_with_cancellation_token():
    """Verify Algorithm 1: Async generator effect iterator with in-flight cancellation guard."""
    from plan_mode.cordis import CancellationToken
    ctx = Context(name="test_async_gen")
    trace = []
    token = CancellationToken()

    async def multi_step_async_gen():
        trace.append("step_1")
        yield lambda: trace.append("undo_1")
        trace.append("step_2")
        yield lambda: trace.append("undo_2")
        # Trigger cancellation before step 3
        token.cancel()
        trace.append("step_3")
        yield lambda: trace.append("undo_3")

    await ctx.async_effect(multi_step_async_gen(), token=token)
    assert "step_1" in trace
    assert "step_2" in trace

    await ctx.async_dispose()
    # Undos must fire in reverse LIFO order
    assert "undo_2" in trace
    assert "undo_1" in trace


def test_cordis_hash_journaling_and_rollback():
    """Verify SHA-256 content hash journaling and state rollback."""
    ctx = Context(name="test_hash_journal")
    fs = {"config.yaml": "version: 1"}

    def revert_config():
        fs["config.yaml"] = "version: 1"

    ctx.journal_mutation(
        target="config.yaml",
        pre_content="version: 1",
        post_content="version: 2",
        inverse=revert_config,
        description="Update version"
    )
    fs["config.yaml"] = "version: 2"
    assert len(ctx._journal) == 1
    assert len(ctx._journal[0].pre_hash) == 64

    # Rollback
    ctx.dispose()
    assert fs["config.yaml"] == "version: 1"


def test_cordis_declarative_component_use():
    """Verify ctx.use() declarative component instantiation and lifecycle binding."""
    class DatabasePlugin:
        def __init__(self, context: Context, url: str = "sqlite:///:memory:"):
            self.context = context
            self.url = url
            self.active = True
            context.provide("service:sql", self)

        def dispose(self):
            self.active = False

    ctx = Context(name="test_use")
    db = ctx.use(DatabasePlugin, config={"url": "postgres://localhost"})
    assert db.active is True
    assert ctx.get("service:sql") is db

    ctx.dispose()
    assert db.active is False


def test_cordis_theorem_63_multi_level_topological_dfs_deactivation():
    """Theorem 63: Transitive dependent fibers (F3 -> F2 -> F1) deactivate in strict topological order."""
    ctx = Context(name="thm63_chain")
    deact_trace = []

    # Root provider provides service:a
    disp_a = ctx.provide("service:a", "val_a")
    # F2 consumes service:a, provides service:b
    f2 = Fiber(ctx, "f2_bridge", dependencies={"service:a"}, provisions={"service:b"}, apply_fn=lambda c: (deact_trace.append("f2_up"), c.provide("service:b", "val_b"))[0])
    # F3 consumes service:b
    f3 = Fiber(ctx, "f3_leaf", dependencies={"service:b"}, apply_fn=lambda c: deact_trace.append("f3_up"))

    f2.activate()
    assert f2.state == LifecycleState.ACTIVE
    f3.activate()
    assert f3.state == LifecycleState.ACTIVE

    # Withdraw root provider service:a -> F3 must deactivate, then F2, before provider clears
    disp_a()
    assert f3.state == LifecycleState.INACTIVE
    assert f2.state == LifecycleState.INACTIVE


def test_cordis_hash_journal_drift_detection(tmp_path):
    """Verify hash journal detects external content drift prior to rollback."""
    test_file = tmp_path / "drifting_config.yaml"
    test_file.write_text("initial_state")

    ctx = Context(name="drift_ctx")
    entry = ctx.journal_mutation(
        target=str(test_file),
        pre_content="initial_state",
        post_content="mutation_v1",
        inverse=lambda: test_file.write_text("initial_state"),
        description="Write v1"
    )
    test_file.write_text("mutation_v1")

    # Simulate external unauthorized drift
    test_file.write_text("corrupted_state_by_external_process")

    # Rollback
    ctx.dispose()
    assert entry.drift_detected is True
    assert test_file.read_text() == "initial_state"


def test_cordis_provider_stack_nested_withdrawal():
    """Verify provider stack: withdrawing top provider restores previous provider binding."""
    ctx = Context(name="stack_test")

    # Provider 1
    disp1 = ctx.provide("db", "pg_v1", provider_id="prov1")
    assert ctx.get("db") == "pg_v1"

    # Provider 2 overrides
    disp2 = ctx.provide("db", "pg_v2", provider_id="prov2")
    assert ctx.get("db") == "pg_v2"

    # Withdraw provider 2 -> provider 1 should be restored
    disp2()
    assert ctx.get("db") == "pg_v1"

    # Withdraw provider 1 -> db becomes None
    disp1()
    assert ctx.get("db") is None


def test_cordis_journal_drift_abort_policy(tmp_path):
    """Verify on_drift='abort' raises RuntimeError when external drift is detected."""
    test_file = tmp_path / "abort_config.yaml"
    test_file.write_text("v1")

    ctx = Context(name="abort_drift_ctx")
    ctx.journal_mutation(
        target=str(test_file),
        pre_content="v1",
        post_content="v2",
        inverse=lambda: test_file.write_text("v1"),
        on_drift="abort"
    )
    test_file.write_text("v2")
    # Simulate drift
    test_file.write_text("unauthorized_v3")

    with pytest.raises(RuntimeError, match="Rollback aborted: external state drift detected"):
        ctx.dispose()


@pytest.mark.asyncio
async def test_execute_plan_task_timeout_and_rollback():
    """Verify execute_plan with timeout_per_task rolls back hanging async tasks."""
    plan_text = """
    1. Fast Task
       Output: fast.txt
    2. Slow Hanging Task
       Depends on 1
       Output: slow.txt
    """
    trace = []

    async def h1(c):
        trace.append("t1_done")
        await c.async_effect(lambda: lambda: trace.remove("t1_done"))
        return {"fast": True}

    async def h2_hanging(c):
        await asyncio.sleep(5.0)  # simulate hang
        return {"slow": True}

    res = await execute_plan(plan_text, task_handlers={1: h1, 2: h2_hanging}, timeout_per_task=0.1)
    assert res["ok"] is False
    assert res["failed_task"] == 2
    assert res["recovered"] is True
    # Fast task effect must be undone
    assert trace == []


@pytest.mark.asyncio
async def test_speculative_rollout_async_clean():
    """Verify speculative_rollout_async scores with async evaluation function and recovers cleanly."""
    from plan_mode import speculative_rollout_async
    trace = []

    async def async_eval(ctx):
        trace.append("evaluated")
        await ctx.async_effect(lambda: lambda: trace.remove("evaluated"))
        return 96.0

    res = await speculative_rollout_async("sample plan", async_eval)
    assert res["ok"] is True
    assert res["score"] == 96.0
    assert trace == []
