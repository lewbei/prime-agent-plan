"""Capability Registry with Strict Typed Contracts and Verifier Bindings."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import ActionIR, FactTruth, PredicateCondition


class CapabilityNotFoundError(Exception):
    pass


class SchemaMismatchError(Exception):
    pass


class CapabilityHashMismatchError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


def typed_args_equal(args1: List[Any], args2: List[Any]) -> bool:
    """Compare argument lists ensuring exact type identity and value equality."""
    if len(args1) != len(args2):
        return False
    return all(type(a1) is type(a2) and a1 == a2 for a1, a2 in zip(args1, args2))


class ObservationVerifier(BaseModel):
    """Deterministic verifier bound to one predicate, argument tuple and truth polarity."""
    verifier_id: str
    predicate: str
    target_args_mapping: List[Any] = Field(default_factory=list)
    expected_truth: FactTruth = FactTruth.VERIFIED_TRUE
    command_template: List[str] = Field(default_factory=list)
    expected_output_pattern: Optional[str] = None
    json_path: Optional[str] = None
    expected_value: Optional[Any] = None
    timeout_seconds: float = 10.0


class CompensationAction(BaseModel):
    compensation_id: str
    capability_name: str
    parameter_mapping: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 60.0


class CapabilityEntry(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str
    platform: str = "linux"
    required_permissions: List[str] = Field(default_factory=list)
    is_idempotent: bool = False
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[PredicateCondition] = Field(default_factory=list)
    positive_effects: List[PredicateCondition] = Field(default_factory=list)
    negative_effects: List[PredicateCondition] = Field(default_factory=list)
    verifiers: List[ObservationVerifier] = Field(default_factory=list)
    executor_command_template: List[str] = Field(default_factory=list)
    default_compensation: Optional[CompensationAction] = None

    def compute_capability_hash(self) -> str:
        data = {
            "name": self.name,
            "version": self.version,
            "platform": self.platform,
            "required_permissions": sorted(self.required_permissions),
            "is_idempotent": self.is_idempotent,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "preconditions": [
                {
                    "key": p.fact_key,
                    "truth": p.expected_truth.value,
                    "active_until_action_id": p.active_until_action_id,
                }
                for p in self.preconditions
            ],
            "positive_effects": [
                {
                    "key": p.fact_key,
                    "truth": p.expected_truth.value,
                    "active_until_action_id": p.active_until_action_id,
                }
                for p in self.positive_effects
            ],
            "negative_effects": [
                {
                    "key": p.fact_key,
                    "truth": p.expected_truth.value,
                    "active_until_action_id": p.active_until_action_id,
                }
                for p in self.negative_effects
            ],
            "verifiers": [
                {
                    "id": v.verifier_id,
                    "predicate": v.predicate,
                    "target_args_mapping": v.target_args_mapping,
                    "expected_truth": v.expected_truth.value,
                    "cmd": v.command_template,
                    "expected_output_pattern": v.expected_output_pattern,
                    "json_path": v.json_path,
                    "expected_value": v.expected_value,
                    "timeout_seconds": v.timeout_seconds,
                }
                for v in self.verifiers
            ],
            "executor_command_template": self.executor_command_template,
            "default_compensation": (
                {
                    "id": self.default_compensation.compensation_id,
                    "cap": self.default_compensation.capability_name,
                    "param_mapping": self.default_compensation.parameter_mapping,
                    "timeout": self.default_compensation.timeout_seconds,
                }
                if self.default_compensation is not None
                else None
            ),
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CapabilityRegistry(BaseModel):
    capabilities: Dict[str, CapabilityEntry] = Field(default_factory=dict)

    def register(self, entry: CapabilityEntry) -> None:
        self.capabilities[entry.name] = entry

    def get(self, name: str) -> CapabilityEntry:
        if name not in self.capabilities:
            raise CapabilityNotFoundError(f"Capability '{name}' not found in registry.")
        return self.capabilities[name]

    @staticmethod
    def _condition_matches(actual: PredicateCondition, expected: PredicateCondition) -> bool:
        return bool(
            actual.predicate == expected.predicate
            and typed_args_equal(actual.args, expected.args)
            and actual.expected_truth == expected.expected_truth
            and actual.active_until_action_id == expected.active_until_action_id
        )

    def _require_conditions(
        self,
        *,
        action: ActionIR,
        actual: List[PredicateCondition],
        required: List[PredicateCondition],
        label: str,
    ) -> None:
        missing = [
            expected for expected in required
            if not any(self._condition_matches(candidate, expected) for candidate in actual)
        ]
        if missing:
            rendered = ", ".join(
                f"{p.fact_key} ({p.expected_truth.value})" for p in missing
            )
            raise SchemaMismatchError(
                f"Action '{action.action_id}' omits required {label} from capability "
                f"'{action.capability_name}': [{rendered}]."
            )

    def validate_action(self, action: ActionIR) -> None:
        """Validate parameters and require the full registered capability contract.

        Planner-authored ActionIR may add stricter preconditions, but it may not
        omit a capability precondition/effect or invent an effect the capability
        does not declare.  Otherwise the validator/runtime would reason about a
        weaker action than the implementation actually performs.
        """
        cap = self.get(action.capability_name)
        params = action.parameters

        for param_name, spec in cap.input_schema.items():
            if isinstance(spec, dict):
                is_req = spec.get("required", False)
                expected_type = spec.get("type", "any")
            else:
                is_req = False
                expected_type = str(spec)

            if is_req and param_name not in params:
                raise SchemaMismatchError(
                    f"Action '{action.action_id}' missing required parameter '{param_name}' "
                    f"for capability '{cap.name}'."
                )
            if param_name in params and not self._check_type(params[param_name], expected_type):
                val = params[param_name]
                raise SchemaMismatchError(
                    f"Action '{action.action_id}' parameter '{param_name}' has invalid value '{val}'. "
                    f"Expected type '{expected_type}', got '{type(val).__name__}'."
                )

        instantiated_preconditions = [self._instantiate_condition(p, params) for p in cap.preconditions]
        instantiated_positive = [self._instantiate_condition(p, params) for p in cap.positive_effects]
        instantiated_negative = [self._instantiate_condition(p, params) for p in cap.negative_effects]

        self._require_conditions(
            action=action,
            actual=action.preconditions,
            required=instantiated_preconditions,
            label="preconditions",
        )
        self._require_conditions(
            action=action,
            actual=action.positive_effects,
            required=instantiated_positive,
            label="positive effects",
        )
        self._require_conditions(
            action=action,
            actual=action.negative_effects,
            required=instantiated_negative,
            label="negative effects",
        )

        for label, actual, allowed in (
            ("positive effect", action.positive_effects, instantiated_positive),
            ("negative effect", action.negative_effects, instantiated_negative),
        ):
            for condition in actual:
                if not any(self._condition_matches(condition, expected) for expected in allowed):
                    allowed_str = ", ".join(
                        f"{p.fact_key} ({p.expected_truth.value})" for p in allowed
                    ) or "[]"
                    raise SchemaMismatchError(
                        f"Action '{action.action_id}' claims undeclared or mismatched {label} "
                        f"'{condition.fact_key}' ({condition.expected_truth.value}) not supported by "
                        f"capability '{cap.name}'. Allowed instantiated effects: [{allowed_str}]."
                    )

    def _instantiate_condition(self, cond: PredicateCondition, params: Dict[str, Any]) -> PredicateCondition:
        instantiated_args: List[Any] = []
        for arg in cond.args:
            if isinstance(arg, str):
                if arg.startswith("{") and arg.endswith("}"):
                    instantiated_args.append(params.get(arg[1:-1], arg))
                elif arg.startswith("$"):
                    instantiated_args.append(params.get(arg[1:], arg))
                elif arg in params:
                    instantiated_args.append(params[arg])
                else:
                    instantiated_args.append(arg)
            else:
                instantiated_args.append(arg)
        return PredicateCondition(
            predicate=cond.predicate,
            args=instantiated_args,
            expected_truth=cond.expected_truth,
            active_until_action_id=cond.active_until_action_id,
        )

    def _check_type(self, val: Any, type_name: str) -> bool:
        if type_name in ("any", "*"):
            return True
        if type_name == "str":
            return isinstance(val, str)
        if type_name == "int":
            return isinstance(val, int) and not isinstance(val, bool)
        if type_name == "float":
            return isinstance(val, (float, int)) and not isinstance(val, bool)
        if type_name == "bool":
            return isinstance(val, bool)
        if type_name == "list":
            return isinstance(val, list)
        if type_name == "dict":
            return isinstance(val, dict)
        return True

    def compute_registry_hash(self) -> str:
        sorted_caps = sorted(self.capabilities.keys())
        hashes = [f"{k}:{self.capabilities[k].compute_capability_hash()}" for k in sorted_caps]
        combined = "\n".join(hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def verify_capability_compatibility(self, capability_name: str, expected_hash: str) -> bool:
        try:
            cap = self.get(capability_name)
            return cap.compute_capability_hash() == expected_hash
        except CapabilityNotFoundError:
            return False
