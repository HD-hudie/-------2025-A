from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# 物理常数与基本几何参数。
# 这里定义了导弹速度、烟幕半径、烟幕下沉速度以及目标圆柱体的尺寸和位置。 
G = 9.8
MISSILE_SPEED = 300.0
UAV_MIN_SPEED = 70.0
UAV_MAX_SPEED = 140.0
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0

FAKE_TARGET = np.array([0.0, 0.0, 0.0])
TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0
TARGET_GEOMETRIC_CENTER = TARGET_CENTER + np.array([0.0, 0.0, TARGET_HEIGHT / 2.0])
WORLD_UP = np.array([0.0, 0.0, 1.0])

MISSILES = {
    "M1": np.array([20000.0, 0.0, 2000.0]),
    "M2": np.array([19000.0, 600.0, 2100.0]),
    "M3": np.array([18000.0, -600.0, 1900.0]),
}

UAVS = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}


def unit(v: np.ndarray) -> np.ndarray:
    # 将任意非零向量归一化，得到方向向量。
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("zero vector has no direction")
    return v / norm


def horizontal_unit(v: np.ndarray) -> np.ndarray:
    return unit(np.array([v[0], v[1], 0.0], dtype=float))


def missile_velocity(missile: str) -> np.ndarray:
    return MISSILE_SPEED * unit(FAKE_TARGET - MISSILES[missile])


def missile_position(missile: str, t: float) -> np.ndarray:
    return MISSILES[missile] + missile_velocity(missile) * t


def missile_hit_time(missile: str) -> float:
    return float(np.linalg.norm(MISSILES[missile] - FAKE_TARGET) / MISSILE_SPEED)


@dataclass(frozen=True)
class MovingFrame:
    origin: np.ndarray
    axes: np.ndarray

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.axes.T


def missile_view_frame(missile: str, t: float) -> MovingFrame:
    # 建立导弹瞬时观察坐标系：x 轴沿着从导弹到目标的视线，便于判断烟幕是否遮挡目标。
    origin = missile_position(missile, t)

    # Local x follows the missile-to-real-target sight line at this instant.
    x_axis = unit(TARGET_GEOMETRIC_CENTER - origin)
    y_raw = np.cross(WORLD_UP, x_axis)
    if np.linalg.norm(y_raw) < 1e-12:
        y_raw = np.cross(np.array([0.0, 1.0, 0.0]), x_axis)
    y_axis = unit(y_raw)
    z_axis = np.cross(x_axis, y_axis)

    return MovingFrame(origin=origin, axes=np.vstack([x_axis, y_axis, z_axis]))


@dataclass(frozen=True)
class SmokeRound:
    # 烟幕弹的运动模型：包含发射点、速度、航向、引信延迟等信息。
    uav: str
    speed: float
    heading: np.ndarray
    release_time: float
    fuse_delay: float

    @property
    def burst_time(self) -> float:
        return self.release_time + self.fuse_delay

    def uav_position(self, t: float) -> np.ndarray:
        return UAVS[self.uav] + self.speed * t * self.heading

    @property
    def release_position(self) -> np.ndarray:
        return self.uav_position(self.release_time)

    @property
    def burst_position(self) -> np.ndarray:
        dt = self.fuse_delay
        pos = self.release_position + self.speed * dt * self.heading
        return pos + np.array([0.0, 0.0, -0.5 * G * dt * dt])

    def is_active(self, t: float) -> bool:
        return self.burst_time <= t <= self.burst_time + SMOKE_DURATION

    def center(self, t: float) -> np.ndarray:
        if not self.is_active(t):
            raise ValueError("smoke round is inactive at this time")
        return self.burst_position + np.array(
            [0.0, 0.0, -SMOKE_SINK_SPEED * (t - self.burst_time)]
        )


@dataclass(frozen=True)
class TargetSamples:
    points: np.ndarray
    normals: np.ndarray


def cylinder_samples(n_theta: int = 120, n_z: int = 9, n_r: int = 4) -> TargetSamples:
    # 将目标圆柱体离散为一组采样点和法向量，后续用于判断是否被烟幕遮挡。
    points: list[list[float]] = []
    normals: list[list[float]] = []
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, TARGET_HEIGHT, n_z)

    # Sample side surface and two caps of the protected cylinder.
    for z in zs:
        for th in thetas:
            c, s = float(np.cos(th)), float(np.sin(th))
            points.append([TARGET_RADIUS * c, TARGET_CENTER[1] + TARGET_RADIUS * s, z])
            normals.append([c, s, 0.0])

    rs = np.linspace(0.0, TARGET_RADIUS, n_r + 1)[1:]
    for z, normal in ((TARGET_HEIGHT, [0.0, 0.0, 1.0]), (0.0, [0.0, 0.0, -1.0])):
        for r in rs:
            for th in thetas:
                points.append(
                    [r * float(np.cos(th)), TARGET_CENTER[1] + r * float(np.sin(th)), z]
                )
                normals.append(normal)
        points.append([0.0, TARGET_CENTER[1], z])
        normals.append(normal)

    return TargetSamples(np.array(points, dtype=float), np.array(normals, dtype=float))


def visible_points(samples: TargetSamples, viewpoint: np.ndarray) -> np.ndarray:
    # 只保留从观察点看过去朝向可见的目标采样点，避免背面点参与遮挡判断。
    sight = viewpoint[None, :] - samples.points
    return samples.points[np.einsum("ij,ij->i", samples.normals, sight) >= -1e-9]


def point_to_segments_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_ends: np.ndarray,
) -> np.ndarray:
    # Compute the shortest distance from smoke center to each sight segment.
    seg = segment_ends - segment_start[None, :]
    denom = np.einsum("ij,ij->i", seg, seg)
    raw = np.einsum("ij,ij->i", point[None, :] - segment_start[None, :], seg) / denom
    lam = np.clip(raw, 0.0, 1.0)
    nearest = segment_start[None, :] + lam[:, None] * seg
    return np.linalg.norm(point[None, :] - nearest, axis=1)


def obscuration_margin(
    missile: str,
    smokes: Iterable[SmokeRound],
    t: float,
    samples: TargetSamples,
    use_visible_surface: bool = True,
) -> float:
    # 计算在当前时刻、当前视角下，烟幕对目标的遮挡裕度；小于等于 0 表示被遮挡。
    mpos = missile_position(missile, t)
    target_points = visible_points(samples, mpos) if use_visible_surface else samples.points
    if len(target_points) == 0:
        return np.inf

    centers = np.array(
        [smoke.center(t) for smoke in smokes if smoke.is_active(t)], dtype=float
    )
    if len(centers) == 0:
        return np.inf

    # Euclidean segment distance is invariant under the old moving-frame
    # rotation.  Evaluate every active smoke and sight segment in one batch.
    segments = target_points - mpos
    denom = np.einsum("ni,ni->n", segments, segments)
    raw = np.einsum("si,ni->sn", centers - mpos, segments) / denom[None, :]
    lam = np.clip(raw, 0.0, 1.0)
    nearest = mpos + lam[:, :, None] * segments[None, :, :]
    best = np.min(np.linalg.norm(centers[:, None, :] - nearest, axis=2), axis=0)
    return float(np.max(best - SMOKE_RADIUS))


def is_obscured(
    missile: str,
    smokes: Iterable[SmokeRound],
    t: float,
    samples: TargetSamples,
    use_visible_surface: bool = True,
) -> bool:
    return obscuration_margin(missile, smokes, t, samples, use_visible_surface) <= 0.0


def effective_intervals(
    missile: str,
    smokes: Iterable[SmokeRound],
    samples: TargetSamples,
    dt: float = 0.01,
    use_visible_surface: bool = True,
) -> list[tuple[float, float]]:
    # 在时间轴上扫描遮挡状态，得到烟幕对导弹视线持续有效的时间区间。
    smokes = list(smokes)
    start = max(0.0, min((s.burst_time for s in smokes), default=0.0))
    stop = min(
        missile_hit_time(missile),
        max((s.burst_time + SMOKE_DURATION for s in smokes), default=0.0),
    )
    if stop <= start:
        return []

    # Scan only the possible active smoke interval.
    times = np.arange(start, stop + 0.5 * dt, dt)
    flags = [
        is_obscured(missile, smokes, float(t), samples, use_visible_surface)
        for t in times
    ]

    intervals: list[tuple[float, float]] = []
    in_interval = False
    begin = 0.0
    for t, ok in zip(times, flags):
        if ok and not in_interval:
            begin = float(t)
            in_interval = True
        elif not ok and in_interval:
            intervals.append((begin, float(t)))
            in_interval = False
    if in_interval:
        intervals.append((begin, float(times[-1] + dt)))
    return intervals


def interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return float(sum(max(0.0, b - a) for a, b in intervals))


def _self_check() -> None:
    starts = np.array([0.0, 0.0, 0.0])
    ends = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    distances = point_to_segments_distance(np.array([5.0, 3.0, 0.0]), starts, ends)
    assert np.allclose(distances, [3.0, 5.0])
    assert np.allclose(horizontal_unit(np.array([-2.0, 0.0, 9.0])), [-1.0, 0.0, 0.0])

    frame = missile_view_frame("M1", 1.0)
    assert np.allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-12)
    assert np.allclose(frame.to_local(frame.origin), np.zeros(3), atol=1e-12)


if __name__ == "__main__":
    _self_check()
    print("smoke_model self-check passed")
