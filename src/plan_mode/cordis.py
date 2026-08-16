"""Spatiotemporal Composability Engine (Cordis Paradigm).

Formal runtime realization of 'A Programming Paradigm for Spatiotemporal Composability'
(Shi, Zhang, Cui 2026):

1. Revertible Effects (Section 3.1):
   - Every context mutation carries an explicit left inverse g: Gamma -> Gamma.
   - Twisted composition monoid T_Gamma: (f1, g1) o (f2, g2) = (f1 o f2, g2 o g1).
   - Accumulator phi rolls up inverses in LIFO order for complete context recovery.
   - Step-boundary effect iterators with cancellation guards (Algorithm 1).
   - Explicit sync effect (ctx.effect) and async effect (ctx.async_effect) pipelines.

2. Reactive Coeffects (Section 3.2):
   - Partial dependent coeffect context Sigma: (k: K) -> V_k.
   - Reactive notification: changes classified as activating / deactivating / neutral.
   - Dynamic fiber lifecycle reactions: fibers activate when dependencies are satisfied
     and deactivate when coeffects are withdrawn.
   - Theorem 63 Provider Withdrawal Guard: a provider's withdrawal automatically
     deactivates dependent fibers before the binding is unmounted.
   - Scoped Realm Isolation (isolate): 2-layer resolution key -> rho(k) -> sigma(rho(k)).
   - Metadata Interception (intercept): capability mediation without triggering reloads.

3. Fiber Lifecycle Calculus (Section 4):
   - Fibers <d, p, e, pi, sigma, tau, theta> spanning 10 formal operational rules:
     Orchestration: O-Insert, O-Retire, O-Remove
     Lifecycle: L-Begin, L-Iter, L-Finish, L-Divert, L-Raise, L-Leave, L-Unload
   - Dynamic state transitions (INACTIVE, RELOADING, ACTIVE, UNLOADING, FAILED).

4. Dynamic Execution & Rollback for Agent Harness & Planning:
   - Speculative execution rollouts with instant state recovery.
   - Isolated subagent fiber realms.
   - Ephemeral tool/verifier synthesis as revertible fibers.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Generator, Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


def _run_inv_sync(inv: Callable[[], Any]) -> None:
    """Run an inverse callable synchronously."""
    if inspect.iscoroutinefunction(inv):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(inv())
        except RuntimeError:
            asyncio.run(inv())
        return
    res = inv()
    if inspect.isawaitable(res):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(res)
        except RuntimeError:
            asyncio.run(res)


async def _run_inv_async(inv: Callable[[], Any]) -> None:
    """Run an inverse callable asynchronously, awaiting any coroutine."""
    if inspect.iscoroutinefunction(inv):
        await inv()
        return
    res = inv()
    if inspect.isawaitable(res):
        await res


# ---------------------------------------------------------------------------
# 1. Revertible Effects & Twisted Monoid Accumulator
# ---------------------------------------------------------------------------

class TwistedMonoid:
    """Twisted composition monoid over context transformations.

    Given (f1, g1) and (f2, g2), the twisted composition is:
        (f1, g1) o (f2, g2) = (f1 o f2, g2 o g1)
    Inverses accumulate in reverse (LIFO) order.
    """

    @staticmethod
    def compose_inverses(g1: Callable[[], Any], g2: Callable[[], Any]) -> Callable[[], Any]:
        """Compose two cleanup/inverse functions: g1 is executed after g2 (LIFO)."""
        def _composite():
            res2 = None
            res1 = None
            try:
                _run_inv_sync(g2)
            except Exception:
                pass  # preserve secondary errors while continuing recovery
            try:
                _run_inv_sync(g1)
            except Exception:
                pass
            return (res2, res1)
        return _composite


class LifecycleState(Enum):
    INACTIVE = "INACTIVE"
    RELOADING = "RELOADING"
    ACTIVE = "ACTIVE"
    UNLOADING = "UNLOADING"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# 2. Unified Context Type (Gamma_infinity)
# ---------------------------------------------------------------------------

class Context:
    """Recursive unified context Gamma_infinity = mu Gamma. Gamma x (Gamma -> Gamma) x Sigma.

    Carries:
    - Current context state (recursive)
    - Effect accumulator (phi)
    - Coeffect store (Sigma)
    - Realm table (rho) and Interception metadata (iota)
    - Registry of active component fibers (F_gamma)
    """

    def __init__(self, parent: Context | None = None, name: str = "root"):
        self.parent = parent
        self.name = name
        self.uid = f"{name}_{id(self)}"

        # @@store: Value store sigma: (r: Realm) -> Typed Value
        self._store: dict[str, Any] = {} if parent is None else parent._store

        # @@isolate: Realm table rho: Map(Key, Realm)
        self._isolate: dict[str, str] = {} if parent is None else dict(parent._isolate)

        # @@intercept: Metadata table iota: (Key -> Metadata)
        self._intercept: dict[str, dict[str, Any]] = {} if parent is None else copy.deepcopy(parent._intercept)

        # Effect accumulator phi: LIFO list of (sync_dispose, async_dispose) pairs
        self._inverses: list[tuple[Callable[[], Any], Callable[[], Coroutine[Any, Any, None]]]] = []
        self._disposed = False

        # Fiber registry: F_gamma (shared across hierarchy)
        self._registry: dict[str, Fiber] = {} if parent is None else parent._registry

        # Reactive listeners on coeffect keys (shared across hierarchy)
        self._listeners: dict[str, list[Callable[[str, Any, str], None]]] = {} if parent is None else parent._listeners

        # Lock for thread safety during concurrent transitions
        self._lock = threading.RLock()

    # --- Revertible Effect Tracking (ctx.effect / ctx.async_effect) ---

    def effect(self, callback: Callable[[], Any | Iterator[Any] | Callable[[], Any]]) -> Callable[[], Any]:
        """Execute a synchronous revertible effect and register its left inverse into the accumulator.

        Accepts:
        - A plain function returning an inverse callable: `def fn(): ... return inverse_fn`
        - A generator/iterator yielding inverses at each step (effect iterator, Definition 51)

        For asynchronous callbacks or coroutines, use `await ctx.async_effect(...)`.
        Returns a dispose closure that recovers the effect immediately.
        """
        with self._lock:
            if self._disposed:
                raise RuntimeError("Cannot execute effect on a disposed context")

            if inspect.iscoroutinefunction(callback):
                raise TypeError("Cannot pass an async function/coroutine to sync ctx.effect(). Use 'await ctx.async_effect(...)' instead.")

            armed = [True]
            inverses_collected: list[Callable[[], Any]] = []

            # 1. Execute forward effect
            res = None
            try:
                res = callback()
            except Exception as err:
                self._disposed = False
                raise err

            if inspect.isawaitable(res):
                raise TypeError("ctx.effect() callback returned an awaitable/coroutine. Use 'await ctx.async_effect(...)' instead.")

            # Handle generator / iterator (Effect Iterator)
            if inspect.isgenerator(res) or hasattr(res, "__next__"):
                try:
                    for inv in res:
                        if not armed[0]:
                            break
                        if callable(inv):
                            inverses_collected.append(inv)
                except Exception as iter_err:
                    # rollback accumulated so far on error
                    for inv in reversed(inverses_collected):
                        try:
                            _run_inv_sync(inv)
                        except Exception:
                            pass
                    raise iter_err
            elif callable(res):
                inverses_collected.append(res)

            # Define self-disposal closures (LIFO)
            def _dispose():
                with self._lock:
                    if not armed[0]:
                        return
                    armed[0] = False
                    for inv in reversed(inverses_collected):
                        try:
                            _run_inv_sync(inv)
                        except Exception:
                            pass

            async def _dispose_async():
                with self._lock:
                    if not armed[0]:
                        return
                    armed[0] = False
                    for inv in reversed(inverses_collected):
                        try:
                            await _run_inv_async(inv)
                        except Exception:
                            pass

            pair = (_dispose, _dispose_async)
            self._inverses.append(pair)
            if self.parent is not None:
                self.parent._inverses.append(pair)

            return _dispose

    async def async_effect(self, callback: Callable[[], Any | Coroutine[Any, Any, Any]]) -> Callable[[], Any]:
        """Execute an async revertible effect and register its left inverse into the accumulator.

        Supports async functions, coroutines, and generators.
        """
        if self._disposed:
            raise RuntimeError("Cannot execute effect on a disposed context")

        armed = [True]
        inverses_collected: list[Callable[[], Any]] = []

        res = callback()
        if inspect.isawaitable(res):
            res = await res

        if callable(res):
            inverses_collected.append(res)
        elif inspect.isgenerator(res) or hasattr(res, "__next__"):
            for inv in res:
                if not armed[0]:
                    break
                if callable(inv):
                    inverses_collected.append(inv)

        def _dispose():
            with self._lock:
                if not armed[0]:
                    return
                armed[0] = False
                for inv in reversed(inverses_collected):
                    try:
                        _run_inv_sync(inv)
                    except Exception:
                        pass

        async def _dispose_async():
            with self._lock:
                if not armed[0]:
                    return
                armed[0] = False
                for inv in reversed(inverses_collected):
                    try:
                        await _run_inv_async(inv)
                    except Exception:
                        pass

        pair = (_dispose, _dispose_async)
        self._inverses.append(pair)
        if self.parent is not None:
            self.parent._inverses.append(pair)

        return _dispose

    def dispose(self):
        """Recover all effects executed under this context in LIFO order (sync)."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            to_run = list(reversed(self._inverses))
            self._inverses.clear()
            for disp_sync, _ in to_run:
                try:
                    disp_sync()
                except Exception:
                    pass

    async def async_dispose(self):
        """Recover all effects executed under this context in LIFO order (async)."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            to_run = list(reversed(self._inverses))
            self._inverses.clear()
            for _, disp_async in to_run:
                try:
                    await disp_async()
                except Exception:
                    pass

    # --- Reactive Coeffects (ctx.provide, ctx.inject, ctx.get) ---

    def _get_realm(self, key: str) -> str:
        """Resolve the realm symbol for a key: k -> rho(k)."""
        return self._isolate.get(key, f"global:{key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Read a dependency value from the coeffect store: sigma(rho(k))."""
        realm = self._get_realm(key)
        val = self._store.get(realm, default)

        # Apply interception metadata if configured
        if key in self._intercept and val is not None:
            meta = self._intercept[key]
            if "proxy" in meta and callable(meta["proxy"]):
                return meta["proxy"](val, self)
        return val

    def provide(self, key: str, value: Any) -> Callable[[], Any]:
        """Provide a service binding under key as a revertible effect: set(k, v)."""
        realm = self._get_realm(key)
        old_val = self._store.get(realm)

        def _forward():
            self._store[realm] = value
            self._notify(key, value, "provided")

            def _inverse():
                # Theorem 63 Provider Withdrawal Guard:
                # A provider cannot withdraw until all dependent active fibers deactivate.
                for fiber in list(self._registry.values()):
                    if key in fiber.dependencies and fiber.state == LifecycleState.ACTIVE:
                        try:
                            fiber.deactivate()
                        except Exception:
                            pass

                if old_val is not None:
                    self._store[realm] = old_val
                    self._notify(key, old_val, "restored")
                else:
                    self._store.pop(realm, None)
                    self._notify(key, None, "withdrawn")

            return _inverse

        return self.effect(_forward)

    def isolate(self, key: str, realm: str | None = None) -> Context:
        """Derive a child context with an isolated realm for the given key."""
        child = self.derive(name=f"{self.name}.iso_{key}")
        target_realm = realm or f"realm:{child.uid}:{key}"
        child._isolate[key] = target_realm
        return child

    def intercept(self, key: str, **metadata) -> None:
        """Attach capability/interception metadata to a coeffect key without triggering reload."""
        if key not in self._intercept:
            self._intercept[key] = {}
        self._intercept[key].update(metadata)

    def derive(self, name: str = "child") -> Context:
        """Create a hierarchical child context sharing store and listeners but tracking own effects."""
        return Context(parent=self, name=name)

    # --- Reactive Dependency Notifications ---

    def on_change(self, key: str, listener: Callable[[str, Any, str], None]) -> Callable[[], None]:
        """Register a reactive listener for coeffect key changes."""
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(listener)

        def _unsub():
            if key in self._listeners and listener in self._listeners[key]:
                self._listeners[key].remove(listener)
        return _unsub

    def _notify(self, key: str, value: Any, action: str):
        """Notify all dependent fibers and listeners of a coeffect mutation."""
        listeners = list(self._listeners.get(key, []))
        for l in listeners:
            try:
                l(key, value, action)
            except Exception:
                pass

        # Wire dynamic fiber lifecycle reactions:
        # Check registered fibers depending on `key`
        for fiber in list(self._registry.values()):
            if key in fiber.dependencies and not fiber.retired:
                if action == "withdrawn" or not fiber.is_satisfied():
                    if fiber.state == LifecycleState.ACTIVE:
                        try:
                            fiber.deactivate()
                        except Exception:
                            pass
                elif action in ("provided", "restored") and fiber.is_satisfied():
                    if fiber.state == LifecycleState.INACTIVE:
                        try:
                            fiber.activate()
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# 3. Fiber Lifecycle State Machine (10 Operational Rules)
# ---------------------------------------------------------------------------

class Fiber:
    """Instantiation of a component (d, p, e) carrying its lifecycle state theta.

    Fields:
    - d: Coeffect dependencies required
    - p: Provisions provided
    - e: Effect function / callback
    - pi: Parent fiber identifier
    - sigma: Fiber-owned coeffect table
    - tau: Retirement flag (False -> active/pending, True -> retired)
    - theta: LifecycleState (Inactive, Reloading, Active, Unloading, Failed)
    """

    def __init__(self, ctx: Context, name: str,
                 dependencies: set[str] | None = None,
                 provisions: set[str] | None = None,
                 apply_fn: Callable[[Context], Any] | None = None):
        self.ctx = ctx.derive(name=f"fiber:{name}")
        self.name = name
        self.uid = f"{name}_{id(self)}"
        self.dependencies = set(dependencies or [])
        self.provisions = set(provisions or [])
        self.apply_fn = apply_fn

        self.state = LifecycleState.INACTIVE
        self.retired = False
        self.error: Exception | None = None
        self._dispose_handle: Callable[[], Any] | None = None
        self._unsub_listeners: list[Callable[[], None]] = []

        # Register fiber into the context hierarchy
        self.ctx._registry[self.uid] = self

    def is_satisfied(self) -> bool:
        """Evaluate the coeffect satisfaction predicate: sigma |= d."""
        for dep in self.dependencies:
            if self.ctx.get(dep) is None:
                return False
        return True

    def activate(self) -> bool:
        """L-Begin -> L-Iter -> L-Finish: Activate the fiber if dependencies are satisfied."""
        if self.state == LifecycleState.ACTIVE:
            return True
        if self.retired:
            return False
        if not self.is_satisfied():
            return False

        self.state = LifecycleState.RELOADING
        try:
            # Execute apply_fn under the fiber context
            if self.apply_fn:
                self._dispose_handle = self.ctx.effect(lambda: self.apply_fn(self.ctx))
            self.state = LifecycleState.ACTIVE
            self.error = None
            return True
        except Exception as err:
            self.state = LifecycleState.FAILED
            self.error = err
            # L-Raise: cleanup on failure
            self.ctx.dispose()
            return False

    def deactivate(self) -> None:
        """L-Leave -> L-Unload: Deactivate the fiber and recover all side effects."""
        if self.state in (LifecycleState.INACTIVE, LifecycleState.UNLOADING):
            return

        self.state = LifecycleState.UNLOADING
        try:
            self.ctx.dispose()
        finally:
            self.state = LifecycleState.INACTIVE

    def retire(self) -> None:
        """O-Retire -> O-Remove: Mark fiber retired and unload it."""
        self.retired = True
        self.deactivate()
        # Remove from registry
        self.ctx._registry.pop(self.uid, None)
        for unsub in self._unsub_listeners:
            try:
                unsub()
            except Exception:
                pass
        self._unsub_listeners.clear()


# ---------------------------------------------------------------------------
# 4. Global Root Context & Agent Tool Registry
# ---------------------------------------------------------------------------

_GLOBAL_CONTEXT = Context(name="global_harness_root")


def get_root_context() -> Context:
    """Get or create the global root Cordis context."""
    return _GLOBAL_CONTEXT


def reset_root_context() -> Context:
    """Reset the global context for testing or session reset."""
    global _GLOBAL_CONTEXT
    _GLOBAL_CONTEXT.dispose()
    _GLOBAL_CONTEXT = Context(name="global_harness_root")
    return _GLOBAL_CONTEXT
