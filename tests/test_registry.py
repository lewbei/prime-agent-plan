"""Unit and contract tests for Capability Registry and Typed Contracts."""

import pytest
from plan_mode.ir import FactTruth, PredicateCondition, ActionIR, Provenance, SourceType
from plan_mode.registry import (
    CapabilityEntry,
    ObservationVerifier,
    CompensationAction,
    CapabilityRegistry,
    SchemaMismatchError,
    CapabilityNotFoundError,
    CapabilityHashMismatchError,
)


@pytest.fixture
def sample_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    
    entry = CapabilityEntry(
        name="fs.create_file",
        version="1.0.0",
        description="Creates a new file at specified path with contents",
        platform="linux",
        required_permissions=["filesystem:write"],
        is_idempotent=True,
        input_schema={
            "path": {"type": "str", "required": True},
            "content": {"type": "str", "required": True},
            "mode": {"type": "int", "required": False, "default": 644},
        },
        output_schema={
            "bytes_written": {"type": "int"},
            "sha256": {"type": "str"},
        },
        preconditions=[
            PredicateCondition(predicate="parent_dir_exists", args=["$path"])
        ],
        positive_effects=[
            PredicateCondition(predicate="file_exists", args=["$path"], expected_truth=FactTruth.VERIFIED_TRUE)
        ],
        negative_effects=[],
        verifiers=[
            ObservationVerifier(
                verifier_id="v_file_exists",
                predicate="file_exists",
                target_args_mapping=["path"],
                command_template=["test", "-f", "$path"],
            )
        ],
        default_compensation=CompensationAction(
            compensation_id="fs.delete_file_comp",
            capability_name="fs.delete_file",
            parameter_mapping={"path": "path"},
        ),
    )
    registry.register(entry)
    return registry


def test_registry_registration_and_lookup(sample_registry: CapabilityRegistry):
    cap = sample_registry.get("fs.create_file")
    assert cap.name == "fs.create_file"
    assert cap.version == "1.0.0"
    assert cap.is_idempotent is True
    assert len(cap.verifiers) == 1
    assert cap.default_compensation is not None


def test_registry_compute_hash(sample_registry: CapabilityRegistry):
    h1 = sample_registry.compute_registry_hash()
    assert isinstance(h1, str)
    assert len(h1) == 64
    h2 = sample_registry.compute_registry_hash()
    assert h1 == h2


def test_validate_action_success(sample_registry: CapabilityRegistry):
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    action = ActionIR(
        action_id="act_01",
        capability_name="fs.create_file",
        parameters={"path": "/tmp/test.txt", "content": "hello world"},
        preconditions=[],
        positive_effects=[],
        provenance=prov,
    )
    # Should not raise
    sample_registry.validate_action(action)


def test_validate_action_missing_required_param(sample_registry: CapabilityRegistry):
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    action = ActionIR(
        action_id="act_02",
        capability_name="fs.create_file",
        parameters={"path": "/tmp/test.txt"},  # missing 'content'
        provenance=prov,
    )
    with pytest.raises(SchemaMismatchError) as exc_info:
        sample_registry.validate_action(action)
    assert "content" in str(exc_info.value)


def test_validate_action_invalid_param_type(sample_registry: CapabilityRegistry):
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    action = ActionIR(
        action_id="act_03",
        capability_name="fs.create_file",
        parameters={"path": "/tmp/test.txt", "content": 12345},  # content should be str
        provenance=prov,
    )
    with pytest.raises(SchemaMismatchError) as exc_info:
        sample_registry.validate_action(action)
    assert "Expected type 'str'" in str(exc_info.value)


def test_validate_action_unregistered_capability(sample_registry: CapabilityRegistry):
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    action = ActionIR(
        action_id="act_04",
        capability_name="database.drop_all_tables",
        parameters={},
        provenance=prov,
    )
    with pytest.raises(CapabilityNotFoundError):
        sample_registry.validate_action(action)


def test_verify_capability_hash_tamper_detection(sample_registry: CapabilityRegistry):
    cap = sample_registry.get("fs.create_file")
    valid_hash = cap.compute_capability_hash()
    assert sample_registry.verify_capability_compatibility("fs.create_file", valid_hash) is True
    assert sample_registry.verify_capability_compatibility("fs.create_file", "bad_hash_123") is False
