from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from smoke_model import (
    SMOKE_SINK_SPEED,
    TARGET_GEOMETRIC_CENTER,
    UAV_MAX_SPEED,
    UAV_MIN_SPEED,
    cylinder_samples,
    missile_position,
)
from strategy_common import (
    DronePlan,
    fmt_vec,
    one_smoke_plan_from_burst_point,
    rounded_key,
    score_smokes,
    smokes_from_drones,
)


UAV = "FY1"
MISSILE = "M1"
MAX_BURST_TIME = 18.0


@dataclass(frozen=True)
class Candidate:
    angle: float
    speed: float
    burst_time: float
    fuse_delay: float

    @property
    def release_time(self) -> float:
        return self.burst_time - self.fuse_delay


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    intervals: tuple[tuple[float, float], ...]


def candidate_key(candidate: Candidate) -> tuple[float, ...]:
    return rounded_key(
        (candidate.angle, candidate.speed, candidate.burst_time, candidate.fuse_delay)
    )


def clamp_candidate(candidate: Candidate) -> Candidate:
    burst_time = float(np.clip(candidate.burst_time, 0.05, MAX_BURST_TIME))
    return Candidate(
        angle=float(candidate.angle),
        speed=float(np.clip(candidate.speed, UAV_MIN_SPEED, UAV_MAX_SPEED)),
        burst_time=burst_time,
        fuse_delay=float(np.clip(candidate.fuse_delay, 0.0, burst_time)),
    )


def plan_from_candidate(candidate: Candidate) -> DronePlan | None:
    candidate = clamp_candidate(candidate)
    if candidate.release_time < -1e-9:
        return None

    return DronePlan(
        UAV,
        candidate.angle,
        candidate.speed,
        max(0.0, candidate.release_time),
        1.0,
        1.0,
        candidate.fuse_delay,
        0.0,
        0.0,
    )


def smoke_from_candidate(candidate: Candidate):
    plan = plan_from_candidate(candidate)
    if plan is None:
        return None
    return smokes_from_drones((plan,), 1)[0]


def candidate_from_burst_point(
    burst_time: float, burst_position: np.ndarray
) -> Candidate | None:
    plan = one_smoke_plan_from_burst_point(UAV, burst_time, burst_position)
    if plan is None:
        return None
    return Candidate(plan.angle, plan.speed, plan.release1 + plan.fuse1, plan.fuse1)


def score_candidate(
    candidate: Candidate,
    samples,
    dt: float,
    cache: dict[tuple[float, ...], ScoredCandidate],
) -> ScoredCandidate:
    candidate = clamp_candidate(candidate)
    key = candidate_key(candidate)
    if key in cache:
        return cache[key]

    plan = plan_from_candidate(candidate)
    if plan is None:
        scored = ScoredCandidate(candidate, -1.0, ())
    else:
        # Shared path: DronePlan -> SmokeRound -> effective_intervals.
        smokes = smokes_from_drones((plan,), 1)
        result = score_smokes((MISSILE,), smokes, samples, dt)
        scored = ScoredCandidate(candidate, result.total, result.intervals[MISSILE])
    cache[key] = scored
    return scored


def seed_candidates() -> list[Candidate]:
    # ponytail: known good seed keeps q2 lightweight; widen this only for a fresh search.
    seeds: list[Candidate] = [
        Candidate(3.113585, 71.981, 3.625, 2.823),
        Candidate(pi, 120.0, 5.1, 3.6),
    ]

    center_times = np.linspace(6.5, 10.5, 9)
    sink_lags = np.linspace(2.0, 5.0, 7)
    line_fractions = np.linspace(0.010, 0.050, 7)

    for center_time in center_times:
        missile = missile_position(MISSILE, float(center_time))
        sight = TARGET_GEOMETRIC_CENTER - missile
        for lag in sink_lags:
            burst_time = float(center_time - lag)
            if burst_time <= 0.0:
                continue
            for frac in line_fractions:
                center = missile + float(frac) * sight
                burst = center + np.array([0.0, 0.0, SMOKE_SINK_SPEED * lag])
                candidate = candidate_from_burst_point(burst_time, burst)
                if candidate is not None:
                    seeds.append(candidate)

    for angle in (3.08, 3.113585, 3.14):
        for speed in (UAV_MIN_SPEED, 80.0, 100.0, 120.0, UAV_MAX_SPEED):
            for burst_time in (3.2, 3.6, 4.0, 4.8, 5.6):
                for fuse_delay in (2.4, 2.8, 3.2, 3.6):
                    if fuse_delay <= burst_time:
                        seeds.append(
                            Candidate(
                                float(angle),
                                float(speed),
                                float(burst_time),
                                float(fuse_delay),
                            )
                        )

    dedup: dict[tuple[float, ...], Candidate] = {}
    for candidate in seeds:
        candidate = clamp_candidate(candidate)
        dedup.setdefault(candidate_key(candidate), candidate)
    return list(dedup.values())


def best_n(
    candidates: list[Candidate],
    samples,
    dt: float,
    n: int,
    cache: dict[tuple[float, ...], ScoredCandidate],
) -> list[ScoredCandidate]:
    scored = [score_candidate(candidate, samples, dt, cache) for candidate in candidates]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:n]


def move(candidate: Candidate, index: int, delta: float) -> Candidate:
    values = [
        candidate.angle,
        candidate.speed,
        candidate.burst_time,
        candidate.fuse_delay,
    ]
    values[index] += delta
    return clamp_candidate(Candidate(*values))


def improve(
    start: Candidate,
    samples,
    dt: float,
    step_sets: list[tuple[float, float, float, float]],
    cache: dict[tuple[float, ...], ScoredCandidate],
) -> ScoredCandidate:
    best = score_candidate(start, samples, dt, cache)

    for steps in step_sets:
        improved = True
        while improved:
            improved = False
            for index, step in enumerate(steps):
                for sign in (-1.0, 1.0):
                    trial = move(best.candidate, index, sign * step)
                    scored = score_candidate(trial, samples, dt, cache)
                    if scored.score > best.score + 1e-9:
                        best = scored
                        improved = True
    return best


def solve() -> ScoredCandidate:
    coarse_samples = cylinder_samples(n_theta=24, n_z=5, n_r=2)
    medium_samples = cylinder_samples(n_theta=72, n_z=9, n_r=3)
    fine_samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)

    coarse_cache: dict[tuple[float, ...], ScoredCandidate] = {}
    medium_cache: dict[tuple[float, ...], ScoredCandidate] = {}
    fine_cache: dict[tuple[float, ...], ScoredCandidate] = {}

    seeds = seed_candidates()
    coarse_top = best_n(seeds, coarse_samples, 0.25, 8, coarse_cache)

    refined = [
        improve(
            item.candidate,
            coarse_samples,
            0.20,
            [
                (0.030, 4.0, 0.30, 0.30),
                (0.015, 2.0, 0.15, 0.15),
                (0.006, 0.8, 0.06, 0.06),
            ],
            coarse_cache,
        )
        for item in coarse_top[:6]
    ]

    medium_pool = [item.candidate for item in coarse_top + refined]
    medium_top = best_n(medium_pool, medium_samples, 0.05, 5, medium_cache)
    medium_refined = [
        improve(
            item.candidate,
            medium_samples,
            0.04,
            [
                (0.004, 0.5, 0.040, 0.040),
                (0.002, 0.25, 0.020, 0.020),
            ],
            medium_cache,
        )
        for item in medium_top[:3]
    ]

    fine_pool = [item.candidate for item in medium_top + medium_refined]
    return best_n(fine_pool, fine_samples, 0.01, 1, fine_cache)[0]


def assert_feasible(smoke) -> None:
    assert UAV_MIN_SPEED <= smoke.speed <= UAV_MAX_SPEED
    assert smoke.release_time >= -1e-9
    assert smoke.fuse_delay >= -1e-9
    assert abs(smoke.burst_time - (smoke.release_time + smoke.fuse_delay)) < 1e-9


def main() -> None:
    best = solve()
    smoke = smoke_from_candidate(best.candidate)
    if smoke is None:
        raise RuntimeError("search returned an infeasible candidate")
    assert_feasible(smoke)

    print("Question 2")
    print(
        "heading = "
        f"({smoke.heading[0]:.6f}, {smoke.heading[1]:.6f}, {smoke.heading[2]:.6f})"
    )
    print(f"heading_angle = {best.candidate.angle:.6f} rad")
    print(f"speed = {smoke.speed:.3f} m/s")
    print(f"release_time = {smoke.release_time:.3f} s")
    print(f"fuse_delay = {smoke.fuse_delay:.3f} s")
    print(f"burst_time = {smoke.burst_time:.3f} s")
    print(f"release_position = {fmt_vec(smoke.release_position)}")
    print(f"burst_position = {fmt_vec(smoke.burst_position)}")
    print("effective_intervals:")
    if best.intervals:
        for start, stop in best.intervals:
            print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
    else:
        print("  none")
    print(f"total_effective_duration = {best.score:.3f} s")


if __name__ == "__main__":
    main()
