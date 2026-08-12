from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import pi, sqrt
import random

import numpy as np

from smoke_model import (
    G,
    SMOKE_SINK_SPEED,
    TARGET_GEOMETRIC_CENTER,
    UAV_MAX_SPEED,
    UAV_MIN_SPEED,
    UAVS,
    cylinder_samples,
    missile_hit_time,
    missile_position,
)
from strategy_common import (
    DronePlan,
    ScoreResult,
    fmt_vec,
    one_smoke_plan_from_burst_point,
    score_smokes,
    smokes_from_drones,
)


UAV_SET = ("FY1", "FY2", "FY3")
MISSILE = "M1"
HIT_TIME = missile_hit_time(MISSILE)
SEARCH_SEEDS = (17, 43, 89, 131, 173)
POP_SIZE = 60
GENERATIONS = 120


@dataclass(frozen=True)
class ScoredQ4:
    plans: tuple[DronePlan, ...]
    result: ScoreResult


def feasible_block(block: np.ndarray, uav: str) -> bool:
    _, speed, release, fuse = block
    max_fuse = sqrt(2.0 * float(UAVS[uav][2]) / G)
    return (
        UAV_MIN_SPEED <= speed <= UAV_MAX_SPEED
        and 0.0 <= release <= HIT_TIME
        and 0.0 <= fuse <= max_fuse
        and release + fuse <= HIT_TIME
    )


def feasible_trial(trial: np.ndarray, target: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial, dtype=float).copy()
    for index, uav in enumerate(UAV_SET):
        offset = 4 * index
        trial[offset] %= 2.0 * pi
        block = trial[offset : offset + 4]
        if not feasible_block(block, uav):
            trial[offset : offset + 4] = target[offset : offset + 4]
    return trial


def vector_plans(vec: np.ndarray) -> tuple[DronePlan, ...]:
    plans = []
    for index, uav in enumerate(UAV_SET):
        offset = 4 * index
        plans.append(
            DronePlan(
                uav,
                float(vec[offset]),
                float(vec[offset + 1]),
                float(vec[offset + 2]),
                1.0,
                1.0,
                float(vec[offset + 3]),
                0.0,
                0.0,
            )
        )
    return tuple(plans)


def score_vector(
    vec: np.ndarray,
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredQ4],
) -> ScoredQ4:
    key = (round(dt, 5), *np.round(vec, 5))
    key = tuple(float(value) for value in key)
    if key not in cache:
        plans = vector_plans(vec)
        result = score_smokes(
            (MISSILE,), smokes_from_drones(plans, 1), samples, dt
        )
        cache[key] = ScoredQ4(plans, result)
    return cache[key]


def sight_line_plan(uav: str, rng: random.Random) -> DronePlan:
    for _ in range(2000):
        center_time = rng.uniform(1.0, HIT_TIME)
        lag = rng.uniform(0.0, min(20.0, center_time))
        burst_time = center_time - lag
        missile = missile_position(MISSILE, center_time)
        sight = TARGET_GEOMETRIC_CENTER - missile
        center = missile + rng.uniform(0.002, 0.12) * sight
        burst = center + np.array([0.0, 0.0, SMOKE_SINK_SPEED * lag])
        plan = one_smoke_plan_from_burst_point(uav, burst_time, burst)
        if plan is not None and plan.release1 + plan.fuse1 <= HIT_TIME:
            return plan
    raise RuntimeError(f"failed to generate a feasible plan for {uav}")


def initial_population(pop_size: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    population = []
    for _ in range(pop_size):
        plans = [sight_line_plan(uav, rng) for uav in UAV_SET]
        population.append(
            np.array(
                [
                    value
                    for plan in plans
                    for value in (plan.angle, plan.speed, plan.release1, plan.fuse1)
                ],
                dtype=float,
            )
        )
    return population


def differential_evolution(
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredQ4],
    generations: int,
    pop_size: int,
    seed: int,
) -> ScoredQ4:
    rng = random.Random(seed)
    population = initial_population(pop_size, seed)
    scores = [score_vector(vector, samples, dt, cache) for vector in population]
    best = max(scores, key=lambda item: item.result.total)

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
                if dimension == forced or rng.random() < 0.9:
                    trial[dimension] = mutant[dimension]
            trial = feasible_trial(trial, population[index])

            scored = score_vector(trial, samples, dt, cache)
            if scored.result.total > scores[index].result.total + 1e-9:
                population[index] = trial
                scores[index] = scored
            if scored.result.total > best.result.total + 1e-9:
                best = scored
    return best


def search_seed(seed: int) -> ScoredQ4:
    samples = cylinder_samples(n_theta=24, n_z=5, n_r=1)
    cache: dict[tuple[float, ...], ScoredQ4] = {}
    return differential_evolution(
        samples,
        0.30,
        cache,
        generations=GENERATIONS,
        pop_size=POP_SIZE,
        seed=seed,
    )


def solve() -> ScoredQ4:
    fine_samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)

    with ProcessPoolExecutor(max_workers=len(SEARCH_SEEDS)) as executor:
        coarse_results = list(executor.map(search_seed, SEARCH_SEEDS))

    fine_cache: dict[tuple[float, ...], ScoredQ4] = {}
    fine_results = [
        score_vector(
            np.array(
                [
                    value
                    for plan in result.plans
                    for value in (plan.angle, plan.speed, plan.release1, plan.fuse1)
                ],
                dtype=float,
            ),
            fine_samples,
            0.01,
            fine_cache,
        )
        for result in coarse_results
    ]
    return max(fine_results, key=lambda item: item.result.total)


def verify(best: ScoredQ4) -> tuple[ScoreResult, tuple[float, ...]]:
    samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)
    smokes = smokes_from_drones(best.plans, 1)
    result = score_smokes((MISSILE,), smokes, samples, dt=0.005)
    individual = tuple(
        score_smokes((MISSILE,), [smoke], samples, dt=0.005).total
        for smoke in smokes
    )
    return result, individual


def main() -> None:
    best = solve()
    verified, individual = verify(best)
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

    print("effective_intervals:")
    intervals = verified.intervals["M1"]
    if intervals:
        for start, stop in intervals:
            print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
    else:
        print("  none")
    print(f"total_effective_duration = {verified.total:.3f} s")
    print(f"individual_durations = {tuple(round(value, 3) for value in individual)}")
    print(f"joint_gain = {verified.total - max(individual):.3f} s")


if __name__ == "__main__":
    main()
