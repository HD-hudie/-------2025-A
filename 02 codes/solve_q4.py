from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import atan2

import numpy as np

from smoke_model import (
    SMOKE_SINK_SPEED,
    TARGET_GEOMETRIC_CENTER,
    UAV_MAX_SPEED,
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


UAV_SET = ("FY1", "FY2", "FY3")
MISSILES = ("M1",)


@dataclass(frozen=True)
class ScoredQ4:
    plans: tuple[DronePlan, ...]
    result: ScoreResult


def plan_values(plans: tuple[DronePlan, ...]) -> list[float]:
    values: list[float] = []
    for plan in plans:
        values.extend([plan.angle, plan.speed, plan.release1, plan.fuse1])
    return values


def plan_key(plans: tuple[DronePlan, ...]) -> tuple[float, ...]:
    return rounded_key(plan_values(plans))


def single_smoke_seeds(uav: str) -> list[DronePlan]:
    seeds: list[DronePlan] = []

    if uav == "FY1":
        seeds.append(DronePlan("FY1", 3.113585, 71.981, 0.802, 1.0, 1.0, 2.823, 0.0, 0.0))

    # Put candidate burst points near the missile-target sight line.
    for center_time in np.linspace(4.0, 12.0, 17):
        missile = missile_position("M1", float(center_time))
        sight = TARGET_GEOMETRIC_CENTER - missile
        for lag in np.linspace(0.5, 6.5, 9):
            burst_time = float(center_time - lag)
            if burst_time <= 0.0:
                continue
            for frac in np.linspace(0.004, 0.070, 12):
                center = missile + float(frac) * sight
                burst = center + np.array([0.0, 0.0, SMOKE_SINK_SPEED * lag])
                plan = one_smoke_plan_from_burst_point(uav, burst_time, burst)
                if plan is not None:
                    seeds.append(plan)

    direct_angle = atan2(-UAVS[uav][1], -UAVS[uav][0])
    for speed in (UAV_MIN_SPEED, 90.0, 120.0, UAV_MAX_SPEED):
        for release in (0.0, 1.0, 2.0, 4.0):
            for fuse in (2.5, 3.5, 5.0):
                seeds.append(DronePlan(uav, direct_angle, speed, release, 1.0, 1.0, fuse, 0.0, 0.0))

    dedup: dict[tuple[float, ...], DronePlan] = {}
    for plan in seeds:
        plan = clamp_drone(plan, 1)
        dedup.setdefault(rounded_key([plan.angle, plan.speed, plan.release1, plan.fuse1]), plan)
    return list(dedup.values())


def score_plans(
    plans: tuple[DronePlan, ...],
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredQ4],
) -> ScoredQ4:
    plans = tuple(clamp_drone(plan, 1) for plan in plans)
    key = plan_key(plans)
    if key not in cache:
        smokes = smokes_from_drones(plans, 1)
        cache[key] = ScoredQ4(plans, score_smokes(MISSILES, smokes, samples, dt))
    return cache[key]


def best_single_plans(uav: str, samples, dt: float, keep: int) -> list[DronePlan]:
    scored: list[ScoredQ4] = []
    cache: dict[tuple[float, ...], ScoredQ4] = {}
    for plan in single_smoke_seeds(uav):
        scored.append(score_plans((plan,), samples, dt, cache))
    scored.sort(key=lambda item: item.result.total, reverse=True)
    return [item.plans[0] for item in scored[:keep]]


def move(plans: tuple[DronePlan, ...], flat_index: int, delta: float) -> tuple[DronePlan, ...]:
    plan_index, field_index = divmod(flat_index, 4)
    plan = plans[plan_index]
    values = [plan.angle, plan.speed, plan.release1, plan.fuse1]
    values[field_index] += delta
    new_plan = DronePlan(plan.uav, values[0], values[1], values[2], 1.0, 1.0, values[3], 0.0, 0.0)
    changed = list(plans)
    changed[plan_index] = clamp_drone(new_plan, 1)
    return tuple(changed)


def improve(
    start: tuple[DronePlan, ...],
    samples,
    dt: float,
    steps: tuple[float, float, float, float],
    cache: dict[tuple[float, ...], ScoredQ4],
) -> ScoredQ4:
    best = score_plans(start, samples, dt, cache)
    improved = True
    while improved:
        improved = False
        for index in range(len(start) * 4):
            step = steps[index % 4]
            for sign in (-1.0, 1.0):
                scored = score_plans(move(best.plans, index, sign * step), samples, dt, cache)
                if scored.result.total > best.result.total + 1e-9:
                    best = scored
                    improved = True
    return best


def solve() -> ScoredQ4:
    coarse_samples = cylinder_samples(n_theta=24, n_z=5, n_r=1)
    medium_samples = cylinder_samples(n_theta=60, n_z=7, n_r=2)
    fine_samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)

    singles = [best_single_plans(uav, coarse_samples, 0.40, 4) for uav in UAV_SET]
    coarse_cache: dict[tuple[float, ...], ScoredQ4] = {}
    coarse = [
        score_plans(tuple(combo), coarse_samples, 0.30, coarse_cache)
        for combo in product(*singles)
    ]
    coarse.sort(key=lambda item: item.result.total, reverse=True)

    medium_cache: dict[tuple[float, ...], ScoredQ4] = {}
    refined = [
        improve(item.plans, medium_samples, 0.12, (0.020, 3.0, 0.20, 0.20), medium_cache)
        for item in coarse[:6]
    ]
    refined.extend(score_plans(item.plans, medium_samples, 0.12, medium_cache) for item in coarse[:6])
    refined.sort(key=lambda item: item.result.total, reverse=True)

    fine_cache: dict[tuple[float, ...], ScoredQ4] = {}
    known_q2_bound = (
        DronePlan("FY1", 3.113585, 71.981, 0.802, 1.0, 1.0, 2.823, 0.0, 0.0),
        DronePlan(
            "FY2",
            atan2(-UAVS["FY2"][1], -UAVS["FY2"][0]),
            UAV_MIN_SPEED,
            0.0,
            1.0,
            1.0,
            2.5,
            0.0,
            0.0,
        ),
        DronePlan(
            "FY3",
            atan2(-UAVS["FY3"][1], -UAVS["FY3"][0]),
            UAV_MIN_SPEED,
            0.0,
            1.0,
            1.0,
            2.5,
            0.0,
            0.0,
        ),
    )
    fine = [
        score_plans(item.plans, fine_samples, 0.01, fine_cache)
        for item in refined[:5]
    ]
    fine.append(score_plans(known_q2_bound, fine_samples, 0.01, fine_cache))
    fine.sort(key=lambda item: item.result.total, reverse=True)
    return fine[0]


def main() -> None:
    best = solve()
    smokes = smokes_from_drones(best.plans, 1)

    print("Question 4")
    for plan, smoke in zip(best.plans, smokes):
        print(f"{plan.uav}:")
        print(f"  heading_angle = {plan.angle:.6f} rad")
        print(f"  speed = {smoke.speed:.3f} m/s")
        print(f"  release_time = {smoke.release_time:.3f} s")
        print(f"  fuse_delay = {smoke.fuse_delay:.3f} s")
        print(f"  burst_time = {smoke.burst_time:.3f} s")
        print(f"  release_position = {fmt_vec(smoke.release_position)}")
        print(f"  burst_position = {fmt_vec(smoke.burst_position)}")

    intervals = best.result.intervals["M1"]
    print("effective_intervals:")
    if intervals:
        for start, stop in intervals:
            print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
    else:
        print("  none")
    print(f"total_effective_duration = {best.result.total:.3f} s")


if __name__ == "__main__":
    main()
