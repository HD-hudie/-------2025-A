from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
from typing import Iterable

import numpy as np

from smoke_model import (
    G,
    SMOKE_DURATION,
    SmokeRound,
    UAV_MAX_SPEED,
    UAV_MIN_SPEED,
    UAVS,
    effective_intervals,
    interval_duration,
    missile_hit_time,
)


@dataclass(frozen=True)
class DronePlan:
    uav: str
    angle: float
    speed: float
    release1: float
    gap12: float
    gap23: float
    fuse1: float
    fuse2: float
    fuse3: float


@dataclass(frozen=True)
class ScoreResult:
    total: float
    intervals: dict[str, tuple[tuple[float, float], ...]]


def fmt_vec(v: np.ndarray) -> str:
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"


def heading(angle: float) -> np.ndarray:
    return np.array([cos(angle), sin(angle), 0.0])


def clamp_drone(plan: DronePlan, smoke_count: int) -> DronePlan:
    return DronePlan(
        uav=plan.uav,
        angle=float(plan.angle),
        speed=float(np.clip(plan.speed, UAV_MIN_SPEED, UAV_MAX_SPEED)),
        release1=float(max(0.0, plan.release1)),
        gap12=float(max(1.0, plan.gap12 if smoke_count >= 2 else 1.0)),
        gap23=float(max(1.0, plan.gap23 if smoke_count >= 3 else 1.0)),
        fuse1=float(max(0.0, plan.fuse1)),
        fuse2=float(max(0.0, plan.fuse2 if smoke_count >= 2 else 0.0)),
        fuse3=float(max(0.0, plan.fuse3 if smoke_count >= 3 else 0.0)),
    )


def smokes_from_drone(plan: DronePlan, smoke_count: int) -> list[SmokeRound]:
    plan = clamp_drone(plan, smoke_count)
    releases = [
        plan.release1,
        plan.release1 + plan.gap12,
        plan.release1 + plan.gap12 + plan.gap23,
    ]
    fuses = [plan.fuse1, plan.fuse2, plan.fuse3]
    return [
        SmokeRound(plan.uav, plan.speed, heading(plan.angle), releases[i], fuses[i])
        for i in range(smoke_count)
    ]


def smokes_from_drones(
    plans: Iterable[DronePlan],
    smoke_count: int,
) -> list[SmokeRound]:
    smokes: list[SmokeRound] = []
    for plan in plans:
        smokes.extend(smokes_from_drone(plan, smoke_count))
    return smokes


def feasible_smokes(smokes: Iterable[SmokeRound], missiles: Iterable[str]) -> bool:
    max_hit_time = max(missile_hit_time(missile) for missile in missiles)
    return all(
        UAV_MIN_SPEED <= smoke.speed <= UAV_MAX_SPEED
        and smoke.release_time >= -1e-9
        and smoke.fuse_delay >= -1e-9
        and smoke.burst_time <= max_hit_time + SMOKE_DURATION
        and smoke.burst_position[2] >= -1e-9
        for smoke in smokes
    )


def score_smokes(
    missiles: Iterable[str],
    smokes: Iterable[SmokeRound],
    samples,
    dt: float,
) -> ScoreResult:
    missiles = tuple(missiles)
    smokes = list(smokes)
    if not feasible_smokes(smokes, missiles):
        return ScoreResult(-1.0, {missile: () for missile in missiles})

    intervals = {
        missile: tuple(effective_intervals(missile, smokes, samples, dt=dt))
        for missile in missiles
    }
    total = sum(interval_duration(value) for value in intervals.values())
    return ScoreResult(total, intervals)


def rounded_key(values: Iterable[float], digits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values)


def one_smoke_plan_from_burst_point(
    uav: str,
    burst_time: float,
    burst_position: np.ndarray,
) -> DronePlan | None:
    start = UAVS[uav]
    if burst_time <= 0.0:
        return None

    dx = float(burst_position[0] - start[0])
    dy = float(burst_position[1] - start[1])
    horizontal_distance = float(np.hypot(dx, dy))
    if horizontal_distance <= 1e-9:
        return None

    speed = horizontal_distance / burst_time
    if not (UAV_MIN_SPEED <= speed <= UAV_MAX_SPEED):
        return None

    height_drop = float(start[2] - burst_position[2])
    if height_drop < 0.0:
        return None

    fuse = sqrt(2.0 * height_drop / G)
    release = burst_time - fuse
    if release < 0.0:
        return None

    return DronePlan(uav, atan2(dy, dx), speed, release, 1.0, 1.0, fuse, 0.0, 0.0)
