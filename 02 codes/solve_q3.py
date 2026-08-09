from __future__ import annotations

from dataclasses import dataclass

from smoke_model import UAV_MIN_SPEED, cylinder_samples
from strategy_common import (
    DronePlan,
    clamp_drone,
    fmt_vec,
    rounded_key,
    score_smokes,
    smokes_from_drones,
)


UAV = "FY1"
MISSILE = "M1"


@dataclass(frozen=True)
class ScoredPlan:
    plan: DronePlan
    score: float
    intervals: tuple[tuple[float, float], ...]


def plan_key(plan: DronePlan) -> tuple[float, ...]:
    return rounded_key(
        (
            plan.angle,
            plan.speed,
            plan.release1,
            plan.gap12,
            plan.gap23,
            plan.fuse1,
            plan.fuse2,
            plan.fuse3,
        )
    )


def score_plan(
    plan: DronePlan,
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredPlan],
) -> ScoredPlan:
    plan = clamp_drone(plan, 3)
    key = plan_key(plan)
    if key in cache:
        return cache[key]

    # score_smokes uses the shared obscuration model for the three-smoke union.
    smokes = smokes_from_drones((plan,), 3)
    result = score_smokes((MISSILE,), smokes, samples, dt)
    scored = ScoredPlan(plan, result.total, result.intervals[MISSILE])
    cache[key] = scored
    return scored


def seeds() -> list[DronePlan]:
    plans: list[DronePlan] = []

    # Start near the second-question result, then vary release spacing and fuses.
    for angle in (3.08, 3.113585, 3.14):
        for speed in (UAV_MIN_SPEED, 72.0, 80.0):
            for release1 in (0.0, 0.8, 1.5):
                for gap12 in (1.0, 2.5, 4.0):
                    for gap23 in (1.0, 2.5, 4.0):
                        for fuse in (2.8, 3.8, 5.0):
                            plans.append(
                                DronePlan(
                                    UAV,
                                    float(angle),
                                    float(speed),
                                    float(release1),
                                    float(gap12),
                                    float(gap23),
                                    float(fuse),
                                    float(fuse),
                                    float(fuse),
                                )
                            )
    plans.append(
        DronePlan(UAV, 3.113585, 71.981, 0.802, 1.0, 1.0, 2.823, 2.823, 2.823)
    )
    return plans


def move(plan: DronePlan, index: int, delta: float) -> DronePlan:
    values = [
        plan.angle,
        plan.speed,
        plan.release1,
        plan.gap12,
        plan.gap23,
        plan.fuse1,
        plan.fuse2,
        plan.fuse3,
    ]
    values[index] += delta
    return clamp_drone(DronePlan(plan.uav, *values), 3)


def improve(
    start: DronePlan,
    samples,
    dt: float,
    step_sets: list[tuple[float, ...]],
    cache: dict[tuple[float, ...], ScoredPlan],
) -> ScoredPlan:
    best = score_plan(start, samples, dt, cache)
    for steps in step_sets:
        improved = True
        while improved:
            improved = False
            for index, step in enumerate(steps):
                for sign in (-1.0, 1.0):
                    trial = move(best.plan, index, sign * step)
                    scored = score_plan(trial, samples, dt, cache)
                    if scored.score > best.score + 1e-9:
                        best = scored
                        improved = True
    return best


def solve() -> ScoredPlan:
    coarse_samples = cylinder_samples(n_theta=24, n_z=5, n_r=1)
    medium_samples = cylinder_samples(n_theta=48, n_z=7, n_r=2)
    fine_samples = cylinder_samples(n_theta=120, n_z=9, n_r=3)

    coarse_cache: dict[tuple[float, ...], ScoredPlan] = {}
    medium_cache: dict[tuple[float, ...], ScoredPlan] = {}
    fine_cache: dict[tuple[float, ...], ScoredPlan] = {}

    coarse = [score_plan(plan, coarse_samples, 0.50, coarse_cache) for plan in seeds()]
    coarse.sort(key=lambda item: item.score, reverse=True)

    refined = [
        improve(
            item.plan,
            coarse_samples,
            0.35,
            [
                (0.030, 4.0, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30),
                (0.012, 1.5, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15),
            ],
            coarse_cache,
        )
        for item in coarse[:6]
    ]

    medium_pool = [item.plan for item in coarse[:6] + refined]
    medium = [score_plan(plan, medium_samples, 0.12, medium_cache) for plan in medium_pool]
    medium.sort(key=lambda item: item.score, reverse=True)
    medium_refined = [
        improve(
            item.plan,
            medium_samples,
            0.08,
            [
                (0.004, 0.5, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06),
                (0.002, 0.2, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
            ],
            medium_cache,
        )
        for item in medium[:4]
    ]

    fine_pool = [item.plan for item in medium[:4] + medium_refined]
    fine = [score_plan(plan, fine_samples, 0.02, fine_cache) for plan in fine_pool]
    fine.sort(key=lambda item: item.score, reverse=True)
    return fine[0]


def main() -> None:
    best = solve()
    smokes = smokes_from_drones((best.plan,), 3)

    print("Question 3")
    print(f"heading_angle = {best.plan.angle:.6f} rad")
    print(f"speed = {best.plan.speed:.3f} m/s")
    for index, smoke in enumerate(smokes, start=1):
        print(f"smoke {index}:")
        print(f"  release_time = {smoke.release_time:.3f} s")
        print(f"  fuse_delay = {smoke.fuse_delay:.3f} s")
        print(f"  burst_time = {smoke.burst_time:.3f} s")
        print(f"  release_position = {fmt_vec(smoke.release_position)}")
        print(f"  burst_position = {fmt_vec(smoke.burst_position)}")
    print("effective_intervals:")
    if best.intervals:
        for start, stop in best.intervals:
            print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
    else:
        print("  none")
    print(f"total_effective_duration = {best.score:.3f} s")


if __name__ == "__main__":
    main()
