from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import pi, sqrt
import random

import numpy as np

from smoke_model import (
    G,
    UAV_MAX_SPEED,
    UAV_MIN_SPEED,
    UAVS,
    cylinder_samples,
    missile_hit_time,
)
from strategy_common import (
    DronePlan,
    fmt_vec,
    rounded_key,
    score_smokes,
    smokes_from_drones,
)


UAV = "FY1"
MISSILE = "M1"
HIT_TIME = missile_hit_time(MISSILE)
MAX_FUSE = sqrt(2.0 * float(UAVS[UAV][2]) / G)
SEARCH_SEEDS = (17, 43, 89, 131, 173)


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


def plan_vector(plan: DronePlan) -> np.ndarray:
    release2 = plan.release1 + plan.gap12
    release3 = release2 + plan.gap23
    return np.array(
        [
            plan.angle,
            plan.speed,
            plan.release1,
            plan.gap12,
            plan.gap23,
            plan.release1 + plan.fuse1,
            release2 + plan.fuse2,
            release3 + plan.fuse3,
        ],
        dtype=float,
    )


def repair_vector(vec: np.ndarray) -> np.ndarray:
    repaired = np.asarray(vec, dtype=float).copy()
    repaired[0] %= 2.0 * pi
    repaired[1] = np.clip(repaired[1], UAV_MIN_SPEED, UAV_MAX_SPEED)

    repaired[2] = np.clip(repaired[2], 0.0, HIT_TIME - 2.0)
    repaired[3] = np.clip(repaired[3], 1.0, HIT_TIME - repaired[2] - 1.0)
    repaired[4] = np.clip(repaired[4], 1.0, HIT_TIME - repaired[2] - repaired[3])
    releases = (
        repaired[2],
        repaired[2] + repaired[3],
        repaired[2] + repaired[3] + repaired[4],
    )
    for index, release in enumerate(releases, start=5):
        repaired[index] = np.clip(
            repaired[index], release, min(HIT_TIME, release + MAX_FUSE)
        )
    return repaired


def vector_plan(vec: np.ndarray) -> DronePlan:
    vec = repair_vector(vec)
    release1 = float(vec[2])
    release2 = release1 + float(vec[3])
    release3 = release2 + float(vec[4])
    return DronePlan(
        UAV,
        float(vec[0]),
        float(vec[1]),
        release1,
        float(vec[3]),
        float(vec[4]),
        float(vec[5]) - release1,
        float(vec[6]) - release2,
        float(vec[7]) - release3,
    )


def score_plan(
    plan: DronePlan,
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredPlan],
) -> ScoredPlan:
    plan = vector_plan(plan_vector(plan))
    key = (round(dt, 5), *plan_key(plan))
    if key in cache:
        return cache[key]

    smokes = smokes_from_drones((plan,), 3)
    result = score_smokes((MISSILE,), smokes, samples, dt)
    intervals = result.intervals[MISSILE]
    scored = ScoredPlan(plan, result.total, intervals)
    cache[key] = scored
    return scored


def is_better(candidate: ScoredPlan, incumbent: ScoredPlan) -> bool:
    return candidate.score > incumbent.score + 1e-9


def latin_hypercube(pop_size: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    columns = []
    for _ in range(8):
        column = [(index + rng.random()) / pop_size for index in range(pop_size)]
        rng.shuffle(column)
        columns.append(column)

    population = []
    for row in range(pop_size):
        values = [column[row] for column in columns]
        release1 = values[2] * (HIT_TIME - 2.0)
        remaining = HIT_TIME - release1 - 2.0
        gap12 = 1.0 + values[3] * remaining
        gap23 = 1.0 + values[4] * (remaining - gap12 + 1.0)
        releases = (release1, release1 + gap12, release1 + gap12 + gap23)
        bursts = [
            release + values[index + 5] * min(MAX_FUSE, HIT_TIME - release)
            for index, release in enumerate(releases)
        ]
        population.append(
            np.array(
                [
                    2.0 * pi * values[0],
                    UAV_MIN_SPEED
                    + values[1] * (UAV_MAX_SPEED - UAV_MIN_SPEED),
                    release1,
                    gap12,
                    gap23,
                    *bursts,
                ],
                dtype=float,
            )
        )
    return population


def differential_evolution(
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredPlan],
    generations: int,
    pop_size: int,
    seed: int,
) -> ScoredPlan:
    rng = random.Random(seed)
    population = latin_hypercube(pop_size, seed)
    scores = [score_plan(vector_plan(vec), samples, dt, cache) for vec in population]
    best = max(scores, key=lambda item: item.score)

    for _ in range(generations):
        for index in range(pop_size):
            choices = [item for item in range(pop_size) if item != index]
            first, second, third = rng.sample(choices, 3)
            mutant = population[first] + 0.7 * (
                population[second] - population[third]
            )

            trial = population[index].copy()
            forced = rng.randrange(len(trial))
            for dimension in range(len(trial)):
                if dimension == forced or rng.random() < 0.85:
                    trial[dimension] = mutant[dimension]
            trial = repair_vector(trial)

            scored = score_plan(vector_plan(trial), samples, dt, cache)
            if is_better(scored, scores[index]):
                population[index] = trial
                scores[index] = scored
            if is_better(scored, best):
                best = scored
    return best


def improve(
    start: DronePlan,
    samples,
    dt: float,
    step_sets: tuple[tuple[float, ...], ...],
    cache: dict[tuple[float, ...], ScoredPlan],
) -> ScoredPlan:
    best = score_plan(start, samples, dt, cache)
    for steps in step_sets:
        for index, step in enumerate(steps):
            for direction in (-1.0, 1.0):
                trial = plan_vector(best.plan)
                trial[index] += direction * step
                scored = score_plan(vector_plan(trial), samples, dt, cache)
                if is_better(scored, best):
                    best = scored
    return best


def solve() -> ScoredPlan:
    coarse_samples = cylinder_samples(n_theta=24, n_z=5, n_r=1)
    medium_samples = cylinder_samples(n_theta=72, n_z=7, n_r=2)
    fine_samples = cylinder_samples(n_theta=120, n_z=9, n_r=3)

    coarse_cache: dict[tuple[float, ...], ScoredPlan] = {}
    independent = [
        differential_evolution(
            coarse_samples,
            0.20,
            coarse_cache,
            generations=36,
            pop_size=28,
            seed=seed,
        )
        for seed in SEARCH_SEEDS
    ]

    medium_cache: dict[tuple[float, ...], ScoredPlan] = {}
    medium = [
        improve(
            result.plan,
            medium_samples,
            0.04,
            (
                (0.04, 4.0, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40),
                (0.01, 1.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10),
            ),
            medium_cache,
        )
        for result in independent
    ]
    medium.sort(key=lambda item: item.score, reverse=True)

    fine_cache: dict[tuple[float, ...], ScoredPlan] = {}
    return improve(
        medium[0].plan,
        fine_samples,
        0.01,
        (
            (0.004, 0.4, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04),
            (0.001, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
        ),
        fine_cache,
    )


def verify(plan: DronePlan):
    samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)
    smokes = smokes_from_drones((plan,), 3)
    result = score_smokes((MISSILE,), smokes, samples, dt=0.005)
    pairs = []
    for kept in combinations(range(3), 2):
        pair_result = score_smokes(
            (MISSILE,), [smokes[index] for index in kept], samples, dt=0.005
        )
        pairs.append((kept, pair_result.total))
    return result, pairs


def main() -> None:
    best = solve()
    verified, pairs = verify(best.plan)
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
    print("verified_effective_intervals:")
    intervals = verified.intervals[MISSILE]
    if intervals:
        for start, stop in intervals:
            print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
    else:
        print("  none")
    print(f"verified_total_effective_duration = {verified.total:.3f} s")
    print("two-smoke_ablation:")
    for kept, duration in pairs:
        labels = ", ".join(str(index + 1) for index in kept)
        print(
            f"  keep smoke {labels}: {duration:.3f} s "
            f"(three-smoke gain = {verified.total - duration:.3f} s)"
        )


if __name__ == "__main__":
    main()
