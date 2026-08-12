from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import pi, sqrt
import os

import numpy as np

from smoke_model import (
    G,
    SMOKE_SINK_SPEED,
    TARGET_GEOMETRIC_CENTER,
    UAVS,
    cylinder_samples,
    is_obscured,
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


UAV_SET = ("FY1", "FY2", "FY3", "FY4", "FY5")
MISSILES = ("M1", "M2", "M3")
HIT_LIMIT = max(missile_hit_time(missile) for missile in MISSILES)
SEARCH_SEEDS = (17, 43, 89, 131, 173)
REFINE_STEPS = 16

COARSE_POP_SIZE = 32
COARSE_GENERATIONS = 25
COARSE_CYCLES = 3
MEDIUM_POP_SIZE = 20
MEDIUM_GENERATIONS = 12


@dataclass(frozen=True)
class ScoredQ5:
    vector: np.ndarray
    plans: tuple[DronePlan, ...]
    result: ScoreResult


@dataclass(frozen=True)
class SeedResult:
    seed: int
    vector: np.ndarray
    score: float


def decode_block(uav: str, block: np.ndarray) -> DronePlan:
    u = np.asarray(block, dtype=float)
    if u.shape != (8,) or np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("normalized UAV block must lie in [0, 1]^8")

    angle = 2.0 * pi * float(u[0])
    speed = 70.0 + 70.0 * float(u[1])
    release1 = (HIT_LIMIT - 2.0) * float(u[2])
    release2 = release1 + 1.0 + (HIT_LIMIT - release1 - 2.0) * float(u[3])
    release3 = release2 + 1.0 + (HIT_LIMIT - release2 - 1.0) * float(u[4])
    releases = (release1, release2, release3)

    max_ground_fuse = sqrt(2.0 * float(UAVS[uav][2]) / G)
    fuses = tuple(
        float(u[5 + index]) * min(max_ground_fuse, HIT_LIMIT - release)
        for index, release in enumerate(releases)
    )
    return DronePlan(
        uav,
        angle,
        speed,
        release1,
        release2 - release1,
        release3 - release2,
        fuses[0],
        fuses[1],
        fuses[2],
    )


def decode_vector(vector: np.ndarray) -> tuple[DronePlan, ...]:
    matrix = np.asarray(vector, dtype=float).reshape(len(UAV_SET), 8)
    return tuple(decode_block(uav, matrix[index]) for index, uav in enumerate(UAV_SET))


def encode_block(
    uav: str,
    angle: float,
    speed: float,
    releases: tuple[float, float, float],
    fuses: tuple[float, float, float],
) -> np.ndarray:
    release1, release2, release3 = releases
    denom12 = HIT_LIMIT - release1 - 2.0
    denom23 = HIT_LIMIT - release2 - 1.0
    block = np.array(
        [
            (angle % (2.0 * pi)) / (2.0 * pi),
            (speed - 70.0) / 70.0,
            release1 / (HIT_LIMIT - 2.0),
            0.0 if denom12 <= 1e-12 else (release2 - release1 - 1.0) / denom12,
            0.0 if denom23 <= 1e-12 else (release3 - release2 - 1.0) / denom23,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    max_ground_fuse = sqrt(2.0 * float(UAVS[uav][2]) / G)
    for index, (release, fuse) in enumerate(zip(releases, fuses)):
        limit = min(max_ground_fuse, HIT_LIMIT - release)
        block[5 + index] = 0.0 if limit <= 1e-12 else fuse / limit
    if np.any(block < -1e-10) or np.any(block > 1.0 + 1e-10):
        raise ValueError("physical seed cannot be encoded in the unit cube")
    block[abs(block) < 1e-12] = 0.0
    block[abs(block - 1.0) < 1e-12] = 1.0
    return block


def sight_line_seed_block(uav: str, rng: np.random.Generator) -> np.ndarray:
    for _ in range(3000):
        missile = MISSILES[int(rng.integers(len(MISSILES)))]
        center_time = float(rng.uniform(1.0, missile_hit_time(missile)))
        age = float(rng.uniform(0.0, min(20.0, center_time - 0.05)))
        burst_time = center_time - age
        mpos = missile_position(missile, center_time)
        center = mpos + float(rng.uniform(0.002, 0.12)) * (
            TARGET_GEOMETRIC_CENTER - mpos
        )
        burst = center + np.array([0.0, 0.0, SMOKE_SINK_SPEED * age])
        anchor = one_smoke_plan_from_burst_point(uav, burst_time, burst)
        if anchor is None:
            continue

        anchor_release = anchor.release1
        slots = rng.permutation(3)
        releases: tuple[float, float, float] | None = None
        anchor_slot = -1
        for slot in slots:
            if slot == 0 and anchor_release <= HIT_LIMIT - 2.0:
                release2 = float(rng.uniform(anchor_release + 1.0, HIT_LIMIT - 1.0))
                release3 = float(rng.uniform(release2 + 1.0, HIT_LIMIT))
                releases = (anchor_release, release2, release3)
            elif slot == 1 and 1.0 <= anchor_release <= HIT_LIMIT - 1.0:
                release1 = float(rng.uniform(0.0, anchor_release - 1.0))
                release3 = float(rng.uniform(anchor_release + 1.0, HIT_LIMIT))
                releases = (release1, anchor_release, release3)
            elif slot == 2 and anchor_release >= 2.0:
                release2 = float(rng.uniform(1.0, anchor_release - 1.0))
                release1 = float(rng.uniform(0.0, release2 - 1.0))
                releases = (release1, release2, anchor_release)
            else:
                continue
            anchor_slot = int(slot)
            break
        if releases is None:
            continue

        max_ground_fuse = sqrt(2.0 * float(UAVS[uav][2]) / G)
        fuses = [
            float(rng.uniform(0.0, min(max_ground_fuse, HIT_LIMIT - release)))
            for release in releases
        ]
        fuses[anchor_slot] = anchor.fuse1
        return encode_block(
            uav,
            anchor.angle,
            anchor.speed,
            releases,
            tuple(fuses),
        )
    raise RuntimeError(f"failed to generate a feasible sight-line seed for {uav}")


def initial_vector(rng: np.random.Generator) -> np.ndarray:
    return np.vstack([sight_line_seed_block(uav, rng) for uav in UAV_SET])


def vector_key(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.round(np.asarray(vector).ravel(), 7))


def score_vector(
    vector: np.ndarray,
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredQ5],
) -> ScoredQ5:
    matrix = np.asarray(vector, dtype=float).reshape(len(UAV_SET), 8)
    key = vector_key(matrix)
    if key not in cache:
        plans = decode_vector(matrix)
        smokes = smokes_from_drones(plans, 3)
        cache[key] = ScoredQ5(
            matrix.copy(), plans, score_smokes(MISSILES, smokes, samples, dt)
        )
    return cache[key]


def varied_block(
    current: np.ndarray, uav: str, index: int, rng: np.random.Generator
) -> np.ndarray:
    if index % 2:
        return sight_line_seed_block(uav, rng)
    block = current + rng.normal(0.0, 0.12, size=8)
    outside = (block < 0.0) | (block > 1.0)
    block[outside] = rng.random(int(np.count_nonzero(outside)))
    return block


def evolve_block(
    context: np.ndarray,
    block_index: int,
    samples,
    dt: float,
    rng: np.random.Generator,
    cache: dict[tuple[float, ...], ScoredQ5],
    pop_size: int,
    generations: int,
) -> ScoredQ5:
    uav = UAV_SET[block_index]
    population = np.array(
        [context[block_index]]
        + [
            varied_block(context[block_index], uav, index, rng)
            for index in range(1, pop_size)
        ]
    )

    def evaluate(block: np.ndarray) -> ScoredQ5:
        trial = context.copy()
        trial[block_index] = block
        return score_vector(trial, samples, dt, cache)

    scores = [evaluate(block) for block in population]
    for _ in range(generations):
        for index in range(pop_size):
            choices = np.delete(np.arange(pop_size), index)
            first, second, third = rng.choice(choices, 3, replace=False)
            mutant = population[first] + 0.7 * (
                population[second] - population[third]
            )
            outside = (mutant < 0.0) | (mutant > 1.0)
            mutant[outside] = rng.random(int(np.count_nonzero(outside)))

            cross = rng.random(8) < 0.85
            cross[int(rng.integers(8))] = True
            trial_block = np.where(cross, mutant, population[index])
            scored = evaluate(trial_block)
            if scored.result.total > scores[index].result.total + 1e-9:
                population[index] = trial_block
                scores[index] = scored

    return max(scores, key=lambda item: item.result.total)


def cooperative_search(
    samples,
    dt: float,
    seed: int,
    cycles: int,
    pop_size: int,
    generations: int,
    start: np.ndarray | None = None,
) -> ScoredQ5:
    rng = np.random.default_rng(seed)
    cache: dict[tuple[float, ...], ScoredQ5] = {}
    if start is None:
        starts = [initial_vector(rng) for _ in range(8)]
        best = max(
            (score_vector(vector, samples, dt, cache) for vector in starts),
            key=lambda item: item.result.total,
        )
    else:
        best = score_vector(start, samples, dt, cache)

    context = best.vector.copy()
    for _ in range(cycles):
        for block_index in rng.permutation(len(UAV_SET)):
            best = evolve_block(
                context,
                int(block_index),
                samples,
                dt,
                rng,
                cache,
                pop_size,
                generations,
            )
            context = best.vector.copy()
    return best


def search_seed(seed: int) -> SeedResult:
    samples = cylinder_samples(n_theta=24, n_z=5, n_r=1)
    best = cooperative_search(
        samples,
        0.20,
        seed,
        COARSE_CYCLES,
        COARSE_POP_SIZE,
        COARSE_GENERATIONS,
    )
    return SeedResult(seed, best.vector, best.result.total)


def refine_medium(args: tuple[int, np.ndarray]) -> SeedResult:
    seed, vector = args
    samples = cylinder_samples(n_theta=72, n_z=9, n_r=2)
    best = cooperative_search(
        samples,
        0.05,
        seed + 10_000,
        cycles=1,
        pop_size=MEDIUM_POP_SIZE,
        generations=MEDIUM_GENERATIONS,
        start=vector,
    )
    return SeedResult(seed, best.vector, best.result.total)


def coordinate_refine(start: np.ndarray) -> ScoredQ5:
    samples = cylinder_samples(n_theta=120, n_z=9, n_r=3)
    cache: dict[tuple[float, ...], ScoredQ5] = {}
    best = score_vector(start, samples, 0.02, cache)
    for step in (0.02, 0.005):
        for flat_index in range(best.vector.size):
            row, column = divmod(flat_index, 8)
            for direction in (-1.0, 1.0):
                value = best.vector[row, column] + direction * step
                if not 0.0 <= value <= 1.0:
                    continue
                trial = best.vector.copy()
                trial[row, column] = value
                scored = score_vector(trial, samples, 0.02, cache)
                if scored.result.total > best.result.total + 1e-9:
                    best = scored
    return best


def refine_boundary(
    missile: str,
    smokes,
    samples,
    left: float,
    right: float,
    take_right: bool,
) -> float:
    for _ in range(REFINE_STEPS):
        mid = 0.5 * (left + right)
        if is_obscured(missile, smokes, mid, samples):
            if take_right:
                right = mid
            else:
                left = mid
        elif take_right:
            left = mid
        else:
            right = mid
    return right if take_right else left


def refine_intervals(
    missile: str,
    smokes,
    samples,
    intervals,
    scan_step: float,
) -> tuple[tuple[float, float], ...]:
    hit_time = missile_hit_time(missile)
    refined = []
    for start, stop in intervals:
        start_ref = (
            0.0
            if start <= 0.0
            else refine_boundary(
                missile,
                smokes,
                samples,
                max(0.0, start - scan_step),
                start,
                True,
            )
        )
        stop_ref = (
            hit_time
            if stop >= hit_time
            else refine_boundary(
                missile,
                smokes,
                samples,
                max(0.0, stop - scan_step),
                stop,
                False,
            )
        )
        refined.append((start_ref, stop_ref))
    return tuple(refined)


def verify(vector: np.ndarray) -> ScoredQ5:
    samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)
    plans = decode_vector(vector)
    smokes = smokes_from_drones(plans, 3)
    scanned = score_smokes(MISSILES, smokes, samples, dt=0.005)
    intervals = {
        missile: refine_intervals(
            missile, smokes, samples, scanned.intervals[missile], 0.005
        )
        for missile in MISSILES
    }
    total = sum(stop - start for value in intervals.values() for start, stop in value)
    return ScoredQ5(vector.copy(), plans, ScoreResult(total, intervals))


def check_parameterization() -> None:
    rng = np.random.default_rng(2025)
    for _ in range(1000):
        plans = decode_vector(rng.random((len(UAV_SET), 8)))
        for plan in plans:
            smokes = smokes_from_drones((plan,), 3)
            assert 70.0 <= plan.speed <= 140.0
            assert smokes[1].release_time - smokes[0].release_time >= 1.0 - 1e-9
            assert smokes[2].release_time - smokes[1].release_time >= 1.0 - 1e-9
            assert all(smoke.release_time >= 0.0 for smoke in smokes)
            assert all(smoke.burst_time <= HIT_LIMIT + 1e-9 for smoke in smokes)
            assert all(smoke.burst_position[2] >= -1e-9 for smoke in smokes)


def solve() -> tuple[ScoredQ5, tuple[SeedResult, ...], tuple[float, ...]]:
    workers = min(len(SEARCH_SEEDS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        coarse = tuple(executor.map(search_seed, SEARCH_SEEDS))
    print(
        "coarse_seed_scores = "
        + str(tuple((item.seed, round(item.score, 3)) for item in coarse)),
        flush=True,
    )

    medium_samples = cylinder_samples(n_theta=72, n_z=9, n_r=2)
    medium_cache: dict[tuple[float, ...], ScoredQ5] = {}
    medium_scores = tuple(
        score_vector(item.vector, medium_samples, 0.05, medium_cache).result.total
        for item in coarse
    )
    print(
        "medium_seed_scores = "
        + str(
            tuple(
                (item.seed, round(score, 3))
                for item, score in zip(coarse, medium_scores)
            )
        ),
        flush=True,
    )
    ranked = sorted(
        zip(coarse, medium_scores), key=lambda item: item[1], reverse=True
    )

    refine_inputs = tuple((item.seed, item.vector) for item, _ in ranked[:3])
    with ProcessPoolExecutor(max_workers=len(refine_inputs)) as executor:
        refined = tuple(executor.map(refine_medium, refine_inputs))
    best_medium = max(refined, key=lambda item: item.score)
    print(
        "refined_medium_scores = "
        + str(tuple((item.seed, round(item.score, 3)) for item in refined)),
        flush=True,
    )

    fine = coordinate_refine(best_medium.vector)
    print(f"fine_local_score = {fine.result.total:.3f}", flush=True)
    verified = verify(fine.vector)
    return verified, coarse, medium_scores


def main() -> None:
    check_parameterization()
    best, coarse, medium_scores = solve()

    print("Question 5")
    print(
        f"note = best feasible solution found by {len(SEARCH_SEEDS)} "
        "independent searches"
    )
    print("independent_seed_scores:")
    for item, medium_score in zip(coarse, medium_scores):
        print(
            f"  seed {item.seed}: coarse = {item.score:.3f} s, "
            f"medium = {medium_score:.3f} s"
        )

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

    print("verified_effective_intervals_by_missile:")
    for missile, intervals in best.result.intervals.items():
        duration = sum(stop - start for start, stop in intervals)
        print(f"{missile}: {duration:.3f} s")
        if intervals:
            for start, stop in intervals:
                print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
        else:
            print("  none")
    print(f"verified_total_effective_duration = {best.result.total:.3f} s")


if __name__ == "__main__":
    main()
