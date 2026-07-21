from __future__ import annotations

import unittest

from slam_icp.loop_closure import select_loop_registrations


def registration(
    source_id: int,
    target_id: int,
    *,
    fitness: float = 0.60,
    rmse: float = 0.20,
    correction_x: float = 0.50,
    correction_y: float = 0.0,
    correction_yaw: float = 2.0,
    degenerate: bool = False,
) -> dict:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "dx": 1.0,
        "dy": 0.0,
        "dyaw_rad": 0.0,
        "fitness": fitness,
        "inlier_rmse_m": rmse,
        "correction_dx": correction_x,
        "correction_dy": correction_y,
        "correction_dyaw_deg": correction_yaw,
        "degenerate": degenerate,
    }


class LoopClosureModeTests(unittest.TestCase):
    def test_auto_registers_every_candidate_and_selects_best_valid_loop(self) -> None:
        candidates = [
            {"source_id": 1, "target_id": 10},
            {"source_id": 2, "target_id": 20},
            {"source_id": 3, "target_id": 30},
        ]
        results = {
            (1, 10): registration(1, 10, fitness=0.55, rmse=0.15),
            (2, 20): registration(2, 20, fitness=0.75, rmse=0.30),
            (3, 30): registration(3, 30, fitness=0.10, rmse=0.20),
        }
        calls: list[tuple[int, int]] = []

        def register_pair(source_id: int, target_id: int) -> dict:
            calls.append((source_id, target_id))
            return results[(source_id, target_id)]

        selected, decisions = select_loop_registrations(
            mode="auto",
            candidates=candidates,
            manual_pairs=[],
            register_pair=register_pair,
        )

        self.assertEqual(calls, [(1, 10), (2, 20), (3, 30)])
        self.assertEqual([(item["source_id"], item["target_id"]) for item in selected], [(2, 20)])
        self.assertEqual(sum(bool(item["accepted"]) for item in decisions), 1)
        self.assertEqual(decisions[0]["rejection_reason"], "valid registration not selected")
        self.assertEqual(decisions[2]["rejection_reason"], "fitness below minimum")

    def test_manual_uses_only_configured_pairs(self) -> None:
        calls: list[tuple[int, int]] = []

        def register_pair(source_id: int, target_id: int) -> dict:
            calls.append((source_id, target_id))
            return registration(source_id, target_id, degenerate=True)

        selected, decisions = select_loop_registrations(
            mode="manual",
            candidates=[{"source_id": 1, "target_id": 2}],
            manual_pairs=[{"source_id": 143, "target_id": 596}],
            register_pair=register_pair,
        )

        self.assertEqual(calls, [(143, 596)])
        self.assertEqual(len(selected), 1)
        self.assertTrue(decisions[0]["accepted"])

    def test_disabled_does_not_register_candidates(self) -> None:
        def register_pair(source_id: int, target_id: int) -> dict:
            self.fail(f"unexpected registration: {source_id} -> {target_id}")

        selected, decisions = select_loop_registrations(
            mode="disabled",
            candidates=[{"source_id": 1, "target_id": 2}],
            manual_pairs=[],
            register_pair=register_pair,
        )

        self.assertEqual(selected, [])
        self.assertEqual(decisions, [])

    def test_auto_continues_when_no_registration_is_valid(self) -> None:
        candidates = [
            {"source_id": 1, "target_id": 10},
            {"source_id": 2, "target_id": 20},
        ]

        def register_pair(source_id: int, target_id: int) -> dict:
            if source_id == 1:
                return registration(source_id, target_id, rmse=0.80)
            raise RuntimeError("scan unavailable")

        selected, decisions = select_loop_registrations(
            mode="auto",
            candidates=candidates,
            manual_pairs=[],
            register_pair=register_pair,
        )

        self.assertEqual(selected, [])
        self.assertFalse(any(item["accepted"] for item in decisions))
        self.assertEqual(decisions[0]["rejection_reason"], "inlier RMSE above maximum")
        self.assertIn("registration failed", decisions[1]["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
