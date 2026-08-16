"""Tests for Causal Validator & Symbolic Planning Engine."""
from __future__ import annotations

import pytest
from plan_mode.causal_validator import ActionSchema, CausalValidator, PlanAST, PlanParser, Proposition
from plan_mode import verify


def test_proposition_parsing():
    """Verify parsing of propositions with negation and arguments."""
    p1 = Proposition.parse("file_exists(model.bin)")
    assert p1.name == "file_exists"
    assert p1.args == ("model.bin",)
    assert not p1.negated

    p2 = Proposition.parse("not dirty(cache)")
    assert p2.name == "dirty"
    assert p2.args == ("cache",)
    assert p2.negated


def test_plan_parser_constructs_ast():
    """Verify that plan text parses into a typed ActionSchema AST."""
    plan_text = """# Objective
    Goal: Deploy image classification service.

    1. Prepare Data
       Inputs: raw.csv
       Output: clean.csv
       Estimated Time: 20 min

    2. Train Model
       Depends on 1
       Inputs: clean.csv
       Output: model.bin
       Estimated Time: 40 min
    """
    ast = PlanParser.parse_plan(plan_text)
    assert len(ast.actions) == 2
    assert ast.actions[0].id == 1
    assert ast.actions[0].outputs == ["clean.csv"]
    assert ast.actions[1].depends_on == [1]
    assert ast.actions[1].inputs == ["clean.csv"]


def test_causal_validator_detects_unsatisfied_precondition():
    """Verify that a missing precondition produces an unsatisfied_precondition flaw."""
    plan_text = """
    1. Step One: Evaluate
       Requires: trained_model.bin
       Output: eval.json
    """
    ast = PlanParser.parse_plan(plan_text)
    res = CausalValidator.validate(ast)
    assert res["ok"] is False
    assert any(f["type"] == "unsatisfied_precondition" for f in res["flaws"])


def test_causal_validator_detects_clobber_threat():
    """Verify that an intermediate action deleting a condition needed by a later step produces a clobber threat."""
    a1 = ActionSchema(id=1, name="Init DB", add_effects=[Proposition("db_online")])
    a2 = ActionSchema(id=2, name="Reset Cluster", del_effects=[Proposition("db_online")])
    a3 = ActionSchema(id=3, name="Query DB", preconditions=[Proposition("db_online")])

    ast = PlanAST(goal="Test clobber", actions=[a1, a2, a3])
    res = CausalValidator.validate(ast)
    assert res["ok"] is False
    assert any(f["type"] == "clobber_threat" for f in res["flaws"])


def test_causal_validator_success_on_sound_plan():
    """Verify that a sound causal plan passes with 100% clean causal links."""
    plan_text = """
    1. Extract Data
       Output: data.csv
    2. Build Index
       Depends on 1
       Inputs: data.csv
       Output: index.bin
    3. Serve
       Depends on 2
       Inputs: index.bin
       Output: server.log
    """
    ast = PlanParser.parse_plan(plan_text)
    res = CausalValidator.validate(ast)
    assert res["ok"] is True
    assert len(res["causal_links"]) >= 2
    assert res["dead_artifacts"] == ["server.log"]


def test_causal_validator_closed_world_negation():
    """Verify that deleting a proposition satisfies a subsequent 'not p' precondition under closed-world semantics."""
    plan_text = """
    1. Clean Cache
       Deletes: dirty(cache)
       Output: clean.log
    2. Build Fast
       Preconditions: not dirty(cache)
       Output: build.bin
    """
    ast = PlanParser.parse_plan(plan_text)
    # Start with dirty(cache) in initial state
    res = CausalValidator.validate(ast, initial_state={"dirty(cache)"})
    assert res["ok"] is True
    assert "dirty(cache)" not in res["final_state"]


def test_verify_grounded_input_seeding(tmp_path):
    """Verify that verify() seeds CausalValidator with ground_check() verified files."""
    real_file = tmp_path / "existing_config.yaml"
    real_file.write_text("env: prod")

    plan_text = f"""
    1. Task One
       Inputs: {real_file}
       Output: result.json
    """
    v = verify(plan_text, cwd=tmp_path)
    assert v["ok"] is True
    assert not any("unsatisfied_precondition" in err for err in v["errors"])
