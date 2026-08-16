"""Spatiotemporal Composability Engine (Cordis Paradigm - Complete Implementation).

Formal runtime realization of 'A Programming Paradigm for Spatiotemporal Composability'
(Shi, Zhang, Cui 2026):

1. Revertible Effects (Section 3.1):
   - Every context mutation carries an explicit left inverse g: Gamma -> Gamma.
   - Twisted composition monoid T_Gamma: (f1, g1) o (f2, g2) = (f1 o f2, g2 o g1).
   - Accumulator phi rolls up inverses in LIFO order for complete context recovery.
   - Async Generator Effect Iterators with Step-Boundary Guards & Cancellation Tokens (Algorithm 1).
   - Content Hash-Verified Journaling & Configurable Drift Policy (warn vs. abort).

2. Reactive Coeffects (Section 3.2):
   - Partial dependent coeffect context Sigma: (k: K) -> V_k.
   - Provider Stack & Identity Resolution: key -> Stack[(ProviderUID, Value)].
   - Reactive notification: activating / deactivating / neutral state transitions.
   - Theorem 63 Topological Provider Withdrawal: Exact reverse-topological DFS deactivation
     of transitive dependent fibers before provider unmounting.
   - Scoped Realm Isolation (isolate): 2-layer resolution key -> rho(k) -> sigma(rho(k)).
   - Metadata Interception (intercept): capability mediation without triggering reloads.

3. Fiber Lifecycle Calculus (Section 4):
   - Fibers <d, p, e, pi, sigma, tau, theta> spanning 10 formal operational rules:
     Orchestration: O-Insert, O-Retire, O-Remove
     Lifecycle: L-Begin, L-Iter, L-Finish, L-Divert, L-Raise, L-Leave, L-Unload
   - Dual sync (activate) and async (async_activate) lifecycle execution.

4. Declarative Component Loader (Section 5):
   - ctx.use(ComponentClass, config): declarative lifecycle loader with schema reconciliation.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Coroutine, Generator, Generic, Iterable, Iterator, Optional, TypeVar

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
# 1. Revertible Effects, Hash Journaling & Twisted Monoid Accumulator
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    """Cryptographically verifiable journal entry for context rollback."""
    ts: float
    description: str
    target: str
    pre_hash: str
    post_hash: str
    inverse: Callable[[], Any]
    context_uid: str
    drift_detected: bool = False
    on_drift: str = "warn"  # "warn" | "abort"


class CancellationToken:
    """Cancellation guard for in-flight async effect iterators (Algorithm 1)."""
    def __init__(self):
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        self._cancelled = True


class TwistedMonoid:
    """Twisted composition monoid over context transformations: (f1, g1) o (f2, g2) = (f1 o f2, g2 o g1)."""

    @staticmethod
    def compose_inverses(g1: Callable[[], Any], g2: Callable[[], Any]) -> Callable[[], Any]:
        """Compose two cleanup/inverse functions: g1 is executed after g2 (LIFO)."""
        def _composite():
            res2 = None
            res1 = None
            try:
                _run_inv_sync(g2)
            except Exception:
                pass
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
    """Recursive unified context Gamma_infinity = mu Gamma. Gamma x (Gamma -> Gamma) x Sigma."""

    def __init__(self, parent: Optional[Context] = None, name: str = "root"):
        self.parent = parent
        self.name = name
        self.uid = f"{name}_{id(self)}"

        # @@store: Value store sigma: (r: Realm) -> Typed Value
        self._store: dict[str, Any] = {} if parent is None else parent._store

        # @@providers: Provider Stack: Realm -> list[(ProviderUID, Value)]
        self._providers: dict[str, list[tuple[str, Any]]] = {} if parent is None else parent._providers

        # @@isolate: Realm table rho: Map(Key, Realm)
        self._isolate: dict[str, str] = {} if parent is None else dict(parent._isolate)

        # @@intercept: Metadata table iota: (Key -> Metadata)
        self._intercept: dict[str, dict[str, Any]] = {} if parent is None else copy.deepcopy(parent._intercept)

        # Effect accumulator phi: LIFO list of (sync_dispose, async_dispose) pairs
        self._inverses: list[tuple[Callable[[], Any], Callable[[], Coroutine[Any, Any, None]]]] = []
        self._journal: list[JournalEntry] = [] if parent is None else parent._journal
        self._disposed = False

        # Fiber registry: F_gamma (shared across hierarchy)
        self._registry: dict[str, Fiber] = {} if parent is None else parent._registry

        # Reactive listeners on coeffect keys
        self._listeners: dict[str, list[Callable[[str, Any, str], None]]] = {} if parent is None else parent._listeners

        # Lock for thread safety during concurrent transitions
        self._lock = threading.RLock()

    # --- Revertible Effect Tracking (ctx.effect / ctx.async_effect) ---

    def effect(self, callback: Callable[[], Any | Iterator[Any] | Callable[[], Any]]) -> Callable[[], Any]:
        """Execute a synchronous revertible effect and register its left inverse into the accumulator."""
        with self._lock:
            if self._disposed:
                raise RuntimeError("Cannot execute effect on a disposed context")

            if inspect.iscoroutinefunction(callback) or inspect.iscoroutine(callback) or inspect.isawaitable(callback):
                raise TypeError("Cannot pass an async function, coroutine, or awaitable to sync ctx.effect(). Use 'await ctx.async_effect(...)' instead.")

            armed = [True]
            inverses_collected: list[Callable[[], Any]] = []

            try:
                res = callback()
            except Exception as err:
                self._disposed = False
                raise err

            if inspect.isawaitable(res):
                raise TypeError("ctx.effect() callback returned an awaitable/coroutine. Use 'await ctx.async_effect(...)' instead.")

            if inspect.isgenerator(res) or hasattr(res, "__next__"):
                try:
                    for inv in res:
                        if not armed[0]:
                            break
                        if callable(inv):
                            inverses_collected.append(inv)
                except Exception as iter_err:
                    for inv in reversed(inverses_collected):
                        try:
                            _run_inv_sync(inv)
                        except Exception:
                            pass
                    raise iter_err
            elif callable(res):
                inverses_collected.append(res)

            def _dispose():
                with self._lock:
                    if not armed[0]:
                        return
                    armed[0] = False
                    for inv in reversed(inverses_collected):
                        try:
                            _run_inv_sync(inv)
                        except Exception as err:
                            if isinstance(err, RuntimeError) and "Rollback aborted" in str(err):
                                raise err

            async def _dispose_async():
                with self._lock:
                    if not armed[0]:
                        return
                    armed[0] = False
                    for inv in reversed(inverses_collected):
                        try:
                            await _run_inv_async(inv)
                        except Exception as err:
                            if isinstance(err, RuntimeError) and "Rollback aborted" in str(err):
                                raise err

            pair = (_dispose, _dispose_async)
            self._inverses.append(pair)
            if self.parent is not None:
                self.parent._inverses.append(pair)

            return _dispose

    async def async_effect(self, callback: Callable[[], Any | Coroutine[Any, Any, Any] | AsyncIterator[Any]],
                           *, token: Optional[CancellationToken] = None) -> Callable[[], Any]:
        """Execute an async revertible effect or async generator effect iterator (Algorithm 1)."""
        if self._disposed:
            raise RuntimeError("Cannot execute effect on a disposed context")

        armed = [True]
        inverses_collected: list[Callable[[], Any]] = []

        res = callback() if callable(callback) else callback
        if inspect.isawaitable(res):
            res = await res

        if hasattr(res, "__anext__") or inspect.isasyncgen(res):
            try:
                async for inv in res:
                    if not armed[0] or (token and token.is_cancelled):
                        break
                    if callable(inv):
                        inverses_collected.append(inv)
            except Exception as e:
                for inv in reversed(inverses_collected):
                    try:
                        await _run_inv_async(inv)
                    except Exception:
                        pass
                raise e
        elif callable(res):
            inverses_collected.append(res)
        elif inspect.isgenerator(res) or hasattr(res, "__next__"):
            for inv in res:
                if not armed[0] or (token and token.is_cancelled):
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

    def journal_mutation(self, target: str, pre_content: str, post_content: str,
                         inverse: Callable[[], Any], description: str = "",
                         on_drift: str = "warn") -> JournalEntry:
        """Record a hash-verified mutation into the context journal with configurable drift policy (warn vs abort)."""
        pre_hash = hashlib.sha256(pre_content.encode("utf-8")).hexdigest() if pre_content else ""
        post_hash = hashlib.sha256(post_content.encode("utf-8")).hexdigest() if post_content else ""

        entry = JournalEntry(
            ts=time.time(),
            description=description,
            target=target,
            pre_hash=pre_hash,
            post_hash=post_hash,
            inverse=inverse,
            context_uid=self.uid,
            on_drift=on_drift
        )
        self._journal.append(entry)

        def _verified_inverse():
            p = Path(target)
            if p.exists() and post_hash:
                try:
                    curr_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                    if curr_hash != post_hash:
                        entry.drift_detected = True
                        if on_drift == "abort":
                            raise RuntimeError(f"Rollback aborted: external state drift detected on target '{target}'")
                except Exception as err:
                    if on_drift == "abort" and isinstance(err, RuntimeError):
                        raise err
            return inverse()

        self.effect(lambda: _verified_inverse)
        return entry

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
                except Exception as err:
                    if isinstance(err, RuntimeError) and "Rollback aborted" in str(err):
                        raise err

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
                except Exception as err:
                    if isinstance(err, RuntimeError) and "Rollback aborted" in str(err):
                        raise err

    # --- Reactive Coeffects (ctx.provide, ctx.inject, ctx.get) ---

    def _get_realm(self, key: str) -> str:
        """Resolve the realm symbol for a key: k -> rho(k)."""
        return self._isolate.get(key, f"global:{key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Read a dependency value from the coeffect store: sigma(rho(k))."""
        realm = self._get_realm(key)
        val = self._store.get(realm, default)

        if key in self._intercept and val is not None:
            meta = self._intercept[key]
            if "proxy" in meta and callable(meta["proxy"]):
                return meta["proxy"](val, self)
        return val

    def provide(self, key: str, value: Any, *, provider_id: Optional[str] = None) -> Callable[[], Any]:
        """Provide a service binding with Provider Stack resolution and Theorem 63 topological DFS deactivation."""
        realm = self._get_realm(key)
        provider_uid = provider_id or f"provider_{id(value)}_{time.time_ns()}"

        def _forward():
            self._providers.setdefault(realm, []).append((provider_uid, value))
            self._store[realm] = value
            self._notify(key, value, "provided")

            def _inverse():
                # Theorem 63 Topological Reverse Deactivation
                active_fibers = {f.uid: f for f in self._registry.values() if f.state == LifecycleState.ACTIVE}
                visited: set[str] = set()
                deactivation_order: list[Fiber] = []

                def _dfs(f_uid: str):
                    visited.add(f_uid)
                    fiber = active_fibers.get(f_uid)
                    if not fiber:
                        return
                    for other in active_fibers.values():
                        if other.uid not in visited and any(p in other.dependencies for p in fiber.provisions):
                            _dfs(other.uid)
                    deactivation_order.append(fiber)

                for f in list(active_fibers.values()):
                    if key in f.dependencies and f.uid not in visited:
                        _dfs(f.uid)

                for fiber in deactivation_order:
                    try:
                        fiber.deactivate()
                    except Exception:
                        pass

                # Pop this provider from the stack
                stack = self._providers.get(realm, [])
                stack = [(p_uid, val) for p_uid, val in stack if p_uid != provider_uid]
                self._providers[realm] = stack

                if stack:
                    # Restore previous provider on the stack
                    prev_uid, prev_val = stack[-1]
                    self._store[realm] = prev_val
                    self._notify(key, prev_val, "restored")
                else:
                    self._store.pop(realm, None)
                    self._providers.pop(realm, None)
                    self._notify(key, None, "withdrawn")

            return _inverse

        return self.effect(_forward)

    def isolate(self, key: str, realm: Optional[str] = None) -> Context:
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

    # --- Declarative Component Loader (ctx.use) ---

    def use(self, plugin_callable_or_class: Any, config: Optional[dict[str, Any]] = None) -> Any:
        """Declarative component loader: instantiates, binds coeffects, and tracks component lifecycle."""
        cfg = config or {}
        if inspect.isclass(plugin_callable_or_class):
            instance = plugin_callable_or_class(self, **cfg)
        elif callable(plugin_callable_or_class):
            instance = plugin_callable_or_class(self, **cfg)
        else:
            instance = plugin_callable_or_class

        if hasattr(instance, "dispose") and callable(instance.dispose):
            self.effect(lambda: instance.dispose)
        return instance


# ---------------------------------------------------------------------------
# 3. Fiber Lifecycle State Machine (10 Operational Rules)
# ---------------------------------------------------------------------------

class Fiber:
    """Instantiation of a component (d, p, e) carrying its lifecycle state theta."""

    def __init__(self, ctx: Context, name: str,
                 dependencies: Optional[set[str]] = None,
                 provisions: Optional[set[str]] = None,
                 apply_fn: Optional[Callable[[Context], Any]] = None):
        self.ctx = ctx.derive(name=f"fiber:{name}")
        self.name = name
        self.uid = f"{name}_{id(self)}"
        self.dependencies = set(dependencies or [])
        self.provisions = set(provisions or [])
        self.apply_fn = apply_fn

        self.state = LifecycleState.INACTIVE
        self.retired = False
        self.error: Optional[Exception] = None
        self._dispose_handle: Optional[Callable[[], Any]] = None
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
        """L-Begin -> L-Iter -> L-Finish: Synchronous fiber activation."""
        if self.state == LifecycleState.ACTIVE:
            return True
        if self.retired or not self.is_satisfied():
            return False

        self.state = LifecycleState.RELOADING
        try:
            if self.apply_fn:
                self._dispose_handle = self.ctx.effect(lambda: self.apply_fn(self.ctx))
            self.state = LifecycleState.ACTIVE
            self.error = None
            return True
        except Exception as err:
            self.state = LifecycleState.FAILED
            self.error = err
            self.ctx.dispose()
            return False

    async def async_activate(self) -> bool:
        """L-Begin -> L-Iter -> L-Finish: Asynchronous fiber activation."""
        if self.state == LifecycleState.ACTIVE:
            return True
        if self.retired or not self.is_satisfied():
            return False

        self.state = LifecycleState.RELOADING
        try:
            if self.apply_fn:
                self._dispose_handle = await self.ctx.async_effect(lambda: self.apply_fn(self.ctx))
            self.state = LifecycleState.ACTIVE
            self.error = None
            return True
        except Exception as err:
            self.state = LifecycleState.FAILED
            self.error = err
            await self.ctx.async_dispose()
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

    async def async_deactivate(self) -> None:
        """L-Leave -> L-Unload: Asynchronously deactivate the fiber and recover all side effects."""
        if self.state in (LifecycleState.INACTIVE, LifecycleState.UNLOADING):
            return

        self.state = LifecycleState.UNLOADING
        try:
            await self.ctx.async_dispose()
        finally:
            self.state = LifecycleState.INACTIVE

    def retire(self) -> None:
        """O-Retire -> O-Remove: Mark fiber retired and unload it."""
        self.retired = True
        self.deactivate()
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
