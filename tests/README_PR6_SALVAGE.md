# PR #6 regression salvage

This branch intentionally ports only audit regressions from superseded PR #6 onto current `main`.

It does **not** import PR #6 production patch layers.

Purpose: identify which PR #6 findings still reproduce after PR #5 was merged, then fix those failures canonically on current `main` before closing PR #6.
