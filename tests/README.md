# plan_mode tests

Run: `python3 -m pytest tests/ -q` from the skill root (Python 3.14, pytest >= 9).

- test_rubric.py: rubric v4 structure, regex compilation, corpus-ID
  verification (S8), positive/negative samples, corpus overmatch check,
  S7 score-drift bound (needs PLANNING_CORPUS env, defaults to
  /home/lewbei/deep_learning/planning_paper).
- test_engine.py: verify/simulate on a reference plan, history folding
  (HIPIF 2606.10507), root-cause critique grouping (2509.25370), legacy
  session load compat, offline rule-based search with adaptive escalation
  (LFS 2506.05213).

No network needed; search tests use expansion="rules".
