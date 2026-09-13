"""Command-line smoke check for camera-scale recovery."""

import argparse

from .camera_scale import intrinsics_from_sensor, recover_camera_scale


def main():
    parser = argparse.ArgumentParser(
        description="Recover and display camera pixel-to-mm scale"
    )
    parser.add_argument("--focal-mm", type=float, required=True)
    parser.add_argument("--sensor-mm", type=float, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--distance-mm", type=float, required=True)
    args = parser.parse_args()

    intrinsics = intrinsics_from_sensor(
        args.focal_mm, args.sensor_mm, args.image_width, args.image_height
    )
    result = recover_camera_scale(
        intrinsics=intrinsics, distance_mm=args.distance_mm
    )
    calibration = result["calibration"]
    print(f"method: {result['method']}")
    print(f"fx: {intrinsics.fx:.3f} px")
    print(f"fy: {intrinsics.fy:.3f} px")
    print(f"mm_per_px: {calibration.mm_per_px:.6f}")
    print(f"100 px length: {calibration.px_to_mm(100):.3f} mm")


if __name__ == "__main__":
    main()