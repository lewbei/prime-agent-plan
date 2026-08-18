"""Capability Registry with Strict Typed Contracts and Verifier Bindings."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import ActionIR, PredicateCondition


class CapabilityNotFoundError(Exception):
    """Raised when an requested capability is missing from the registry."""
    pass


class SchemaMismatchError(Exception):
    """Raised when an action's parameters violate the capability input schema."""
    pass


class CapabilityHashMismatchError(Exception):
    """Raised when a capability's implementation or signature does not match its expected hash."""
    pass


class PermissionDeniedError(Exception):
    """Raised when required permissions for a capability are not granted."""
    pass


class ObservationVerifier(BaseModel):
    """Deterministic verifier used to witness world state post-execution."""
    verifier_id: str
    predicate: str
    target_args_mapping: List[str] = Field(default_factory=list)
    command_template: List[str] = Field(default_factory=list)
    expected_output_pattern: Optional[str] = None
    json_path: Optional[str] = None
    expected_value: Optional[Any] = None
    timeout_seconds: float = 10.0


class CompensationAction(BaseModel):
    """Declarative backward compensation step to undo/revert side-effects."""
    compensation_id: str
    capability_name: str
    parameter_mapping: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 60.0


class CapabilityEntry(BaseModel):
    """Declarative typed specification of a capability."""
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
    default_compensation: Optional[CompensationAction] = None

    def compute_capability_hash(self) -> str:
        """Deterministic SHA-256 hash of capability contract."""
        data = {
            "name": self.name,
            "version": self.version,
            "platform": self.platform,
            "required_permissions": sorted(self.required_permissions),
            "is_idempotent": self.is_idempotent,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "preconditions": [
                {"key": p.fact_key, "truth": p.expected_truth.value}
                for p in self.preconditions
            ],
            "positive_effects": [
                {"key": p.fact_key, "truth": p.expected_truth.value}
                for p in self.positive_effects
            ],
            "negative_effects": [
                {"key": p.fact_key, "truth": p.expected_truth.value}
                for p in self.negative_effects
            ],
            "verifiers": [
                {
                    "id": v.verifier_id,
                    "predicate": v.predicate,
                    "cmd": v.command_template,
                }
                for v in self.verifiers
            ],
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CapabilityRegistry(BaseModel):
    """Central store for all validated capabilities and contracts."""
    capabilities: Dict[str, CapabilityEntry] = Field(default_factory=dict)

    def register(self, entry: CapabilityEntry) -> None:
        """Register a new capability entry."""
        self.capabilities[entry.name] = entry

    def get(self, name: str) -> CapabilityEntry:
        """Retrieve capability by name or raise CapabilityNotFoundError."""
        if name not in self.capabilities:
            raise CapabilityNotFoundError(f"Capability '{name}' not found in registry.")
        return self.capabilities[name]

    def validate_action(self, action: ActionIR) -> None:
        """Validate action parameters strictly against registered capability schema."""
        cap = self.get(action.capability_name)
        params = action.parameters

        # Check required fields
        for param_name, spec in cap.input_schema.items():
            if isinstance(spec, dict):
                is_req = spec.get("required", False)
                expected_type = spec.get("type", "any")
            else:
                is_req = False
                expected_type = str(spec)

            if is_req and param_name not in params:
                raise SchemaMismatchError(
                    f"Action '{action.action_id}' missing required parameter '{param_name}' for capability '{cap.name}'."
                )

            if param_name in params:
                val = params[param_name]
                if not self._check_type(val, expected_type):
                    raise SchemaMismatchError(
                        f"Action '{action.action_id}' parameter '{param_name}' has invalid value '{val}'. Expected type '{expected_type}', got '{type(val).__name__}'."
                    )

        # Check declared positive effects against instantiated capability schema
        instantiated_positive = [
            self._instantiate_condition(p, params) for p in cap.positive_effects
        ]
        for pos in action.positive_effects:
            matched = any(
                pos.predicate == cap_p.predicate
                and [str(a) for a in pos.args] == [str(a) for a in cap_p.args]
                and pos.expected_truth == cap_p.expected_truth
                for cap_p in instantiated_positive
            )
            if not matched:
                allowed_str = ", ".join(f"{p.fact_key} ({p.expected_truth.value})" for p in instantiated_positive) or "[]"
                raise SchemaMismatchError(
                    f"Action '{action.action_id}' claims undeclared or mismatched positive effect '{pos.fact_key}' ({pos.expected_truth.value}) not supported by capability '{cap.name}'. Allowed instantiated effects: [{allowed_str}]."
                )

        # Check declared negative effects against instantiated capability schema
        instantiated_negative = [
            self._instantiate_condition(n, params) for n in cap.negative_effects
        ]
        for neg in action.negative_effects:
            matched = any(
                neg.predicate == cap_n.predicate
                and [str(a) for a in neg.args] == [str(a) for a in cap_n.args]
                and neg.expected_truth == cap_n.expected_truth
                for cap_n in instantiated_negative
            )
            if not matched:
                allowed_str = ", ".join(f"{n.fact_key} ({n.expected_truth.value})" for n in instantiated_negative) or "[]"
                raise SchemaMismatchError(
                    f"Action '{action.action_id}' claims undeclared or mismatched negative effect '{neg.fact_key}' ({neg.expected_truth.value}) not supported by capability '{cap.name}'. Allowed instantiated negative effects: [{allowed_str}]."
                )

    def _instantiate_condition(self, cond: PredicateCondition, params: Dict[str, Any]) -> PredicateCondition:
        """Instantiate template variables in predicate arguments with action parameter values."""
        instantiated_args: List[Any] = []
        for arg in cond.args:
            if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
                var_name = arg[1:-1]
                val = params.get(var_name, arg)
                instantiated_args.append(val)
            else:
                instantiated_args.append(arg)
        return PredicateCondition(
            predicate=cond.predicate,
            args=instantiated_args,
            expected_truth=cond.expected_truth,
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
        """Compute aggregate deterministic SHA-256 hash across all capabilities."""
        sorted_caps = sorted(self.capabilities.keys())
        hashes = [f"{k}:{self.capabilities[k].compute_capability_hash()}" for k in sorted_caps]
        combined = "\n".join(hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def verify_capability_compatibility(self, capability_name: str, expected_hash: str) -> bool:
        """Check if capability matches expected cryptographic signature."""
        try:
            cap = self.get(capability_name)
            return cap.compute_capability_hash() == expected_hash
        except CapabilityNotFoundError:
            return False
