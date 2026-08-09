from __future__ import annotations

from smoke_model import (
    FAKE_TARGET,
    UAVS,
    SmokeRound,
    cylinder_samples,
    effective_intervals,
    horizontal_unit,
    interval_duration,
    missile_view_frame,
)
from strategy_common import fmt_vec


def main() -> None:
    # 题 1 里先按题设固定 UAV 的初始航向和烟幕参数，然后求出其遮挡效果。
    heading = horizontal_unit(FAKE_TARGET - UAVS["FY1"])
    smoke = SmokeRound(
        uav="FY1",
        speed=120.0,
        heading=heading,
        release_time=1.5,
        fuse_delay=3.6,
    )

    # 用更细密的采样点逼近圆柱体目标，提升遮挡判断的近似精度。
    samples = cylinder_samples(n_theta=180, n_z=11, n_r=5)
    intervals = effective_intervals("M1", [smoke], samples, dt=0.01)
    total = interval_duration(intervals)

    frame = missile_view_frame("M1", smoke.burst_time)
    burst_local = frame.to_local(smoke.burst_position)

    # 输出关键几何量和烟幕对目标的有效遮挡时段，便于直接分析题目结果。
    print("Question 1")
    print(f"heading = ({heading[0]:.6f}, {heading[1]:.6f}, {heading[2]:.6f})")
    print(f"speed = {smoke.speed:.3f} m/s")
    print(f"release_time = {smoke.release_time:.3f} s")
    print(f"fuse_delay = {smoke.fuse_delay:.3f} s")
    print(f"burst_time = {smoke.burst_time:.3f} s")
    print(f"release_position = {fmt_vec(smoke.release_position)}")
    print(f"burst_position = {fmt_vec(smoke.burst_position)}")
    print(f"burst_position_in_moving_frame = {fmt_vec(burst_local)}")
    print("effective_intervals:")
    if intervals:
        for start, stop in intervals:
            print(f"  [{start:.3f}, {stop:.3f}] duration = {stop - start:.3f} s")
    else:
        print("  none")
    print(f"total_effective_duration = {total:.3f} s")


if __name__ == "__main__":
    main()
