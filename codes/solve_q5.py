from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import atan2

import numpy as np

from smoke_model import (
    SMOKE_SINK_SPEED,
    TARGET_GEOMETRIC_CENTER,
    UAV_MIN_SPEED,
    UAVS,
    cylinder_samples,
    missile_position,
)
from strategy_common import (
    DronePlan,
    ScoreResult,
    clamp_drone,
    fmt_vec,
    one_smoke_plan_from_burst_point,
    rounded_key,
    score_smokes,
    smokes_from_drones,
)


UAV_SET = ("FY1", "FY2", "FY3", "FY4", "FY5")
MISSILES = ("M1", "M2", "M3")


@dataclass(frozen=True)
class ScoredQ5:
    plans: tuple[DronePlan, ...]
    result: ScoreResult


BASELINE_PLANS = (
    DronePlan("FY1", 3.095738, 72.719, 0.000, 1.0, 1.0, 2.693, 2.693, 2.693),
    DronePlan("FY2", -0.735765, 128.976, 12.120, 1.0, 1.0, 2.880, 2.880, 2.880),
    DronePlan("FY3", 2.677945, UAV_MIN_SPEED, 0.000, 1.0, 1.0, 2.500, 2.500, 2.500),
    DronePlan("FY4", -0.777281, 131.534, 7.129, 1.0, 1.0, 9.071, 9.071, 9.071),
    DronePlan("FY5", 2.014078, 127.075, 12.370, 1.0, 1.0, 0.630, 0.630, 0.630),
)


def plan_values(plans: tuple[DronePlan, ...]) -> list[float]:
    values: list[float] = []
    for plan in plans:
        values.extend(
            [
                plan.angle,
                plan.speed,
                plan.release1,
                plan.gap12,
                plan.gap23,
                plan.fuse1,
                plan.fuse2,
                plan.fuse3,
            ]
        )
    return values


def plan_key(plans: tuple[DronePlan, ...]) -> tuple[float, ...]:
    return rounded_key(plan_values(plans))


def smoke_train_from_single(base: DronePlan, gap: float) -> DronePlan:
    base = clamp_drone(base, 1)
    release1 = max(0.0, base.release1 - gap)
    return clamp_drone(
        DronePlan(
            base.uav,
            base.angle,
            base.speed,
            release1,
            gap,
            gap,
            base.fuse1,
            base.fuse1,
            base.fuse1,
        ),
        3,
    )


def single_smoke_seeds(uav: str) -> list[DronePlan]:
    seeds: list[DronePlan] = []

    # Lightweight heuristic: sample a few representative points near each sight line.
    for missile_name in MISSILES:
        for center_time in (5.0, 10.0, 15.0):
            missile = missile_position(missile_name, float(center_time))
            sight = TARGET_GEOMETRIC_CENTER - missile
            for lag in (1.5, 3.5, 5.5):
                burst_time = float(center_time - lag)
                if burst_time <= 0.0:
                    continue
                for frac in (0.015, 0.040, 0.070):
                    center = missile + frac * sight
                    burst = center + np.array([0.0, 0.0, SMOKE_SINK_SPEED * lag])
                    plan = one_smoke_plan_from_burst_point(uav, burst_time, burst)
                    if plan is not None:
                        seeds.append(plan)

    direct_angle = atan2(-UAVS[uav][1], -UAVS[uav][0])
    for speed in (UAV_MIN_SPEED, 90.0, 120.0):
        for release in (0.0, 2.0, 5.0):
            for fuse in (2.5, 4.0, 6.0):
                seeds.append(DronePlan(uav, direct_angle, speed, release, 1.0, 1.0, fuse, 0.0, 0.0))

    dedup: dict[tuple[float, ...], DronePlan] = {}
    for plan in seeds:
        plan = clamp_drone(plan, 1)
        dedup.setdefault(rounded_key([plan.angle, plan.speed, plan.release1, plan.fuse1]), plan)
    return list(dedup.values())


def drone_plan_seeds(uav: str, samples, dt: float, keep: int) -> list[DronePlan]:
    single_scores: list[tuple[float, DronePlan]] = []
    for single in single_smoke_seeds(uav):
        smokes = smokes_from_drones((single,), 1)
        score = score_smokes(MISSILES, smokes, samples, dt).total
        single_scores.append((score, single))
    single_scores.sort(key=lambda item: item[0], reverse=True)

    train_plans: list[DronePlan] = []
    for _, single in single_scores[:4]:
        for gap in (1.0, 2.5):
            train_plans.append(smoke_train_from_single(single, gap))

    scored: list[tuple[float, DronePlan]] = []
    seen: set[tuple[float, ...]] = set()
    for plan in train_plans:
        key = rounded_key(plan_values((plan,)))
        if key in seen:
            continue
        seen.add(key)
        smokes = smokes_from_drones((plan,), 3)
        scored.append((score_smokes(MISSILES, smokes, samples, dt).total, plan))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [plan for _, plan in scored[:keep]]


def score_plans(
    plans: tuple[DronePlan, ...],
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredQ5],
) -> ScoredQ5:
    plans = tuple(clamp_drone(plan, 3) for plan in plans)
    key = plan_key(plans)
    if key not in cache:
        smokes = smokes_from_drones(plans, 3)
        cache[key] = ScoredQ5(plans, score_smokes(MISSILES, smokes, samples, dt))
    return cache[key]


def solve() -> ScoredQ5:
    coarse_samples = cylinder_samples(n_theta=18, n_z=5, n_r=1)
    fine_samples = cylinder_samples(n_theta=72, n_z=9, n_r=2)

    per_uav = [drone_plan_seeds(uav, coarse_samples, 0.80, 2) for uav in UAV_SET]

    coarse_cache: dict[tuple[float, ...], ScoredQ5] = {}
    coarse = [
        score_plans(tuple(combo), coarse_samples, 0.75, coarse_cache)
        for combo in product(*per_uav)
    ]
    coarse.append(score_plans(BASELINE_PLANS, coarse_samples, 0.75, coarse_cache))
    coarse.sort(key=lambda item: item.result.total, reverse=True)

    fine_cache: dict[tuple[float, ...], ScoredQ5] = {}
    fine = [
        score_plans(item.plans, fine_samples, 0.10, fine_cache)
        for item in coarse[:4]
    ]
    fine.append(score_plans(BASELINE_PLANS, fine_samples, 0.10, fine_cache))
    fine.sort(key=lambda item: item.result.total, reverse=True)
    return fine[0]


def main() -> None:
    best = solve()

    print("Question 5")
    print("note = heuristic feasible solution; not a global optimum certificate")
    for plan in best.plans:
        smokes = smokes_from_drones((plan,), 3)
        print(f"{plan.uav}:")
        print(f"  heading_angle = {plan.angle:.6f} rad")
        print(f"  speed = {plan.speed:.3f} m/s")
        for index, smoke in enumerate(smokes, start=1):
            print(f"  smoke {index}:")
            print(f"    release_time = {smoke.release_time:.3f} s")
            print(f"    fuse_delay = {smoke.fuse_delay:.3f} s")
            print(f"    burst_time = {smoke.burst_time:.3f} s")
            print(f"    release_position = {fmt_vec(smoke.release_position)}")
            print(f"    burst_position = {fmt_vec(smoke.burst_position)}")

    print("effective_intervals_by_missile:")
    for missile, intervals in best.result.intervals.items():
        duration = sum(stop - start for start, stop in intervals)
        print(f"{missile}: {duration:.3f} s")
        if intervals:
            for start, stop in intervals:
                print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
        else:
            print("  none")
    print(f"total_effective_duration = {best.result.total:.3f} s")


if __name__ == "__main__":
    main()
