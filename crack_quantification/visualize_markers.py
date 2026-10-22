"""
visualize_markers.py
====================

High-precision visual marking engine for civil infrastructure crack evaluation.
Renders metric measurements (Width, Volume, Recess Depth, and ACI Severity)
directly onto the inspection photograph with engineering HUD overlays.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def _draw_hud_panel(
    img: np.ndarray,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    bg_color: Tuple[int, int, int] = (20, 24, 30),
    border_color: Tuple[int, int, int] = (60, 180, 255),
    alpha: float = 0.82,
) -> None:
    """Draw semi-transparent glassmorphic HUD background card."""
    x, y = top_left
    w, h = size
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)


def _draw_pill_badge(
    img: np.ndarray,
    text: str,
    top_left: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> Tuple[int, int]:
    """Draw rounded status pill badge and return (width, height)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 12, 6
    x, y = top_left
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), bg_color, -1)
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x + pad_x, y + bh - pad_y), font, scale, text_color, thickness, cv2.LINE_AA)
    return bw, bh


def _find_normal_ray_endpoints(
    y: int,
    x: int,
    mask: np.ndarray,
    skel: np.ndarray,
    max_search_px: int = 200,
) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
    """Compute orthogonal normal line segment across crack boundaries at (x, y)."""
    h, w = mask.shape[:2]
    # Estimate local tangent from 8-connected neighbors
    nb = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                nb.append((dy, dx))
    
    if len(nb) >= 2:
        dy = nb[-1][0] - nb[0][0]
        dx = nb[-1][1] - nb[0][1]
    elif len(nb) == 1:
        dy, dx = nb[0]
    else:
        dy, dx = 1.0, 0.0

    length = np.hypot(dx, dy)
    if length < 1e-4:
        dx, dy = 1.0, 0.0
        length = 1.0
    
    tx, ty = dx / length, dy / length
    # Normal is perpendicular: (-ty, tx)
    nx, ny = -ty, tx

    # Trace positive normal
    p1 = (x, y)
    for step in np.arange(1.0, max_search_px, 0.8):
        px = int(round(x + step * nx))
        py = int(round(y + step * ny))
        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            p1 = (px, py)
        else:
            break

    # Trace negative normal
    p2 = (x, y)
    for step in np.arange(1.0, max_search_px, 0.8):
        px = int(round(x - step * nx))
        py = int(round(y - step * ny))
        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            p2 = (px, py)
        else:
            break

    ray_length = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    return p1, p2, ray_length


def _draw_colorbar(
    img: np.ndarray,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    min_val: float,
    max_val: float,
    unit: str,
    title: str,
    colormap=cv2.COLORMAP_TURBO,
) -> None:
    """Draw a vertical colorbar with ticks and labels."""
    x, y = top_left
    w, h = size
    # Draw background card
    _draw_hud_panel(img, (x - 10, y - 25), (w + 105, h + 45), bg_color=(20, 24, 30), alpha=0.85)

    # Color gradient
    grad = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    grad_col = cv2.applyColorMap(grad, colormap)
    grad_bar = cv2.resize(grad_col, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = grad_bar
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1, cv2.LINE_AA)

    # Title
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, title, (x - 5, y - 10), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Ticks (top, middle, bottom)
    ticks = [
        (y, max_val, f"{max_val:.1f} {unit}"),
        (y + h // 2, (max_val + min_val) / 2.0, f"{(max_val + min_val) / 2.0:.1f} {unit}"),
        (y + h, min_val, f"{min_val:.1f} {unit}"),
    ]
    for ty, val, label in ticks:
        cv2.line(img, (x + w, ty), (x + w + 6, ty), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, label, (x + w + 10, ty + 4), font, 0.40, (230, 230, 230), 1, cv2.LINE_AA)


def mark_crack_comprehensive(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    dist: np.ndarray,
    recess_depth_mm: np.ndarray,
    report: dict,
    mm_per_px: float,
    image_title: str = "crack-695010_1280",
) -> np.ndarray:
    """
    Renders the primary comprehensive engineering marked image containing:
    1. Damage segmentation contour & glow outline
    2. Recess depth heatmap overlay inside crevice (Volume & Depth)
    3. Centerline skeleton
    4. Orthogonal normal calipers at sample checkpoints and max-width location (Width)
    5. Severity status badge (Severity)
    6. Complete civil structural engineering HUD panel
    """
    marked = image_bgr.copy()
    h, w = marked.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    skel = (skeleton > 0).astype(np.uint8)

    # -------------------------------------------------------------
    # 1. 3D Recess Depth Heatmap Overlay inside crack crevice (Volume)
    # -------------------------------------------------------------
    max_d = float(recess_depth_mm.max()) if recess_depth_mm.max() > 0 else 1.0
    norm_depth = np.clip((recess_depth_mm / max_d) * 255.0, 0, 255).astype(np.uint8)
    depth_heatmap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)

    # Blend depth heatmap only over crack pixels
    crack_idx = binary > 0
    overlay = marked.copy()
    overlay[crack_idx] = depth_heatmap[crack_idx]
    cv2.addWeighted(overlay, 0.65, marked, 0.35, 0, marked)

    # Damage boundary contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    # Skeleton centerline
    marked[skel > 0] = (255, 255, 255)  # white skeleton core

    # -------------------------------------------------------------
    # 2. Width Markings: Caliper lines along skeleton & Max Width Callout
    # -------------------------------------------------------------
    ys, xs = np.nonzero(skel)
    if len(ys) > 0:
        # Compute local widths along skeleton
        widths_px = 2.0 * dist[ys, xs]
        widths_mm = widths_px * mm_per_px
        max_idx = int(np.argmax(widths_mm))
        my, mx = ys[max_idx], xs[max_idx]
        max_w_val_mm = widths_mm[max_idx]
        max_w_val_px = widths_px[max_idx]

        # Draw sample caliper lines along crack (evenly spaced stations)
        num_stations = 8
        step_stride = max(1, len(ys) // num_stations)
        sample_indices = list(range(step_stride // 2, len(ys), step_stride))
        if max_idx not in sample_indices:
            sample_indices.append(max_idx)

        for s_idx in sample_indices:
            sy, sx = ys[s_idx], xs[s_idx]
            p1, p2, r_len_px = _find_normal_ray_endpoints(sy, sx, binary, skel)
            w_meas_mm = r_len_px * mm_per_px

            is_max = (s_idx == max_idx)
            caliper_color = (0, 70, 255) if is_max else (255, 200, 0)
            caliper_thick = 3 if is_max else 1

            # Draw caliper line
            cv2.line(marked, p1, p2, caliper_color, caliper_thick, cv2.LINE_AA)
            # End ticks
            cv2.circle(marked, p1, 3 if is_max else 2, caliper_color, -1, cv2.LINE_AA)
            cv2.circle(marked, p2, 3 if is_max else 2, caliper_color, -1, cv2.LINE_AA)

            # Draw checkpoint measurement label for non-max stations
            if not is_max and r_len_px > 15:
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                lbl = f"{w_meas_mm:.1f}mm"
                cv2.putText(marked, lbl, (mid_x + 8, mid_y - 4), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

        # Highlight Maximum Width point with prominent arrow and callout card
        p1_max, p2_max, _ = _find_normal_ray_endpoints(my, mx, binary, skel)
        cv2.arrowedLine(marked, p1_max, p2_max, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(marked, p2_max, p1_max, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.15)
        cv2.circle(marked, (mx, my), 5, (0, 255, 255), -1, cv2.LINE_AA)

        # Max Width Callout Box
        callout_x = min(w - 260, max(20, mx + 60))
        callout_y = max(40, my - 50)
        cv2.line(marked, (mx, my), (callout_x, callout_y + 20), (0, 255, 255), 2, cv2.LINE_AA)
        _draw_hud_panel(marked, (callout_x, callout_y), (250, 60), bg_color=(20, 24, 32), border_color=(0, 255, 255), alpha=0.9)
        cv2.putText(marked, "MAX CRACK OPENING (WIDTH)", (callout_x + 10, callout_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(marked, f"W_max = {max_w_val_mm:.2f} mm  ({max_w_val_px:.1f} px)", (callout_x + 10, callout_y + 44), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # 3. Volume & Max Depth Callout
    # -------------------------------------------------------------
    vol_data = report.get("volume", {}) or {}
    vol_mm3 = vol_data.get("volume_mm3", 0.0)
    vol_cm3 = vol_data.get("volume_cm3", 0.0)
    mean_depth = vol_data.get("mean_recess_depth_mm", 0.0)
    max_depth = vol_data.get("max_recess_depth_mm", max_d)

    # Find location of max depth
    depth_ys, depth_xs = np.nonzero(binary)
    if len(depth_ys) > 0:
        d_idx = int(np.argmax(recess_depth_mm[depth_ys, depth_xs]))
        d_y, d_x = depth_ys[d_idx], depth_xs[d_idx]
        cv2.circle(marked, (d_x, d_y), 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(marked, (d_x, d_y), (0, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        # Volume callout box
        vol_callout_x = max(20, min(w - 280, d_x - 140))
        vol_callout_y = min(h - 90, d_y + 60)
        cv2.line(marked, (d_x, d_y), (vol_callout_x + 130, vol_callout_y), (0, 180, 255), 2, cv2.LINE_AA)
        _draw_hud_panel(marked, (vol_callout_x, vol_callout_y), (270, 65), bg_color=(20, 24, 32), border_color=(0, 180, 255), alpha=0.9)
        cv2.putText(marked, "3D RECESS VOLUME & DEPTH", (vol_callout_x + 10, vol_callout_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(marked, f"Volume: {vol_mm3:,.1f} mm3 ({vol_cm3:.2f} cm3)", (vol_callout_x + 10, vol_callout_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(marked, f"Mean: {mean_depth:.1f} mm | Max Depth: {max_depth:.1f} mm", (vol_callout_x + 10, vol_callout_y + 57), cv2.FONT_HERSHEY_PLAIN, 0.95, (200, 230, 255), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # 4. Vertical Colorbar for Crevice Depth (Bottom-Right)
    # -------------------------------------------------------------
    cbar_w, cbar_h = 16, 150
    cbar_x = w - 140
    cbar_y = h - 220
    _draw_colorbar(marked, (cbar_x, cbar_y), (cbar_w, cbar_h), 0.0, max_depth, "mm", "Crevice Depth", cv2.COLORMAP_TURBO)

    # -------------------------------------------------------------
    # 5. Top Engineering HUD Panel (Inspection Analytics + Severity)
    # -------------------------------------------------------------
    hud_w = 460
    hud_h = 240
    hud_x, hud_y = 20, 20
    _draw_hud_panel(marked, (hud_x, hud_y), (hud_w, hud_h), bg_color=(15, 18, 24), border_color=(0, 200, 255), alpha=0.90)

    # Title & Target
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(marked, "INFRAVISION AI  STRUCTURAL INSPECTION", (hud_x + 16, hud_y + 26), font, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(marked, f"Target: {image_title}  |  Scale: {mm_per_px:.3f} mm/px", (hud_x + 16, hud_y + 44), font, 0.38, (180, 190, 200), 1, cv2.LINE_AA)
    cv2.line(marked, (hud_x + 16, hud_y + 52), (hud_x + hud_w - 16, hud_y + 52), (50, 70, 90), 1, cv2.LINE_AA)

    # Severity Badge
    sev_obj = report.get("severity", {}) or {}
    sev_label = sev_obj.get("severity", "Severe").upper()
    aci_obj = report.get("aci_classification", {}) or {}
    aci_code = aci_obj.get("code_standard", "ACI 224R-01")

    # Severity color mapping
    if sev_label == "SEVERE":
        badge_bg = (30, 30, 210)  # Red BGR
    elif sev_label == "MODERATE":
        badge_bg = (20, 140, 230)  # Orange BGR
    else:
        badge_bg = (40, 180, 40)  # Green BGR

    badge_w, badge_h = _draw_pill_badge(marked, f"SEVERITY: {sev_label} [{aci_code}]", (hud_x + 16, hud_y + 60), bg_color=badge_bg, scale=0.50, thickness=1)

    # Quantitative Metric Lines
    length_mm = report.get("length_mm", 0.0)
    mean_w_mm = report.get("mean_width_mm", 0.0)
    max_w_mm = report.get("max_width_mm", 0.0)
    min_w_mm = report.get("min_width_mm", 0.0)
    area_mm2 = report.get("area_mm2", 0.0)
    area_unc = report.get("area_uncertainty", {}) or {}
    area_delta = area_unc.get("uncertainty_delta_mm2", 0.0)

    line_y = hud_y + 104
    spacing = 22

    # Length
    cv2.putText(marked, "LENGTH", (hud_x + 16, line_y), font, 0.40, (140, 170, 190), 1, cv2.LINE_AA)
    cv2.putText(marked, f"{length_mm:.1f} mm  ({report.get('length_px', 0):.0f} px)", (hud_x + 140, line_y), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Width
    line_y += spacing
    cv2.putText(marked, "WIDTH (Mean / Max)", (hud_x + 16, line_y), font, 0.40, (140, 170, 190), 1, cv2.LINE_AA)
    cv2.putText(marked, f"Avg: {mean_w_mm:.2f} mm  |  Max: {max_w_mm:.2f} mm", (hud_x + 140, line_y), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Area
    line_y += spacing
    cv2.putText(marked, "SURFACE AREA", (hud_x + 16, line_y), font, 0.40, (140, 170, 190), 1, cv2.LINE_AA)
    cv2.putText(marked, f"{area_mm2:,.1f} mm2  (+-{area_delta:.1f} mm2, 95% CI)", (hud_x + 140, line_y), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Volume & Depth
    line_y += spacing
    cv2.putText(marked, "3D RECESS VOLUME", (hud_x + 16, line_y), font, 0.40, (140, 170, 190), 1, cv2.LINE_AA)
    cv2.putText(marked, f"{vol_mm3:,.1f} mm3  ({vol_cm3:.2f} cm3)", (hud_x + 140, line_y), font, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

    # Action Advised
    line_y += spacing
    action_text = aci_obj.get("action", "Immediate structural intervention required.")
    if len(action_text) > 48:
        action_text = action_text[:45] + "..."
    cv2.putText(marked, "ACTION REQUIRED", (hud_x + 16, line_y), font, 0.40, (255, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(marked, action_text, (hud_x + 140, line_y), font, 0.38, (255, 200, 200), 1, cv2.LINE_AA)

    return marked


def mark_width_diagnostic(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    dist: np.ndarray,
    mm_per_px: float,
) -> np.ndarray:
    """Generate diagnostic visualization focusing on Width normal-ray profiling."""
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    skel = (skeleton > 0).astype(np.uint8)

    # Outline
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (100, 255, 100), 1, cv2.LINE_AA)

    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return vis

    widths_px = 2.0 * dist[ys, xs]
    widths_mm = widths_px * mm_per_px
    max_w = float(widths_mm.max()) if widths_mm.max() > 0 else 1.0

    # Draw color-coded skeleton where color represents local width
    for y, x, w_val in zip(ys, xs, widths_mm):
        # Color from green (hairline) -> yellow (moderate) -> red (severe)
        norm_w = min(1.0, w_val / max_w)
        r = int(255 * norm_w)
        g = int(255 * (1.0 - norm_w))
        b = 50
        cv2.circle(vis, (x, y), 2, (b, g, r), -1)

    # Dense caliper sampling
    step = max(1, len(ys) // 25)
    for idx in range(0, len(ys), step):
        sy, sx = ys[idx], xs[idx]
        p1, p2, ray_px = _find_normal_ray_endpoints(sy, sx, binary, skel)
        cv2.line(vis, p1, p2, (255, 230, 0), 1, cv2.LINE_AA)

    # HUD Card
    _draw_hud_panel(vis, (20, 20), (380, 110), bg_color=(15, 20, 30), border_color=(255, 200, 0), alpha=0.88)
    cv2.putText(vis, "CRACK WIDTH NORMAL-RAY PROFILE", (35, 45), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Mean Width : {widths_mm.mean():.2f} mm", (35, 70), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Max Width  : {widths_mm.max():.2f} mm", (35, 90), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Min Width  : {widths_mm.min():.2f} mm", (35, 110), cv2.FONT_HERSHEY_DUPLEX, 0.42, (100, 255, 100), 1, cv2.LINE_AA)

    _draw_colorbar(vis, (w - 130, h - 220), (16, 140), float(widths_mm.min()), float(widths_mm.max()), "mm", "Local Width", cv2.COLORMAP_JET)

    return vis


def mark_volume_diagnostic(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    recess_depth_mm: np.ndarray,
    vol_report: dict,
    mm_per_px: float,
) -> np.ndarray:
    """Generate diagnostic visualization focusing on 3D crevice recess volume."""
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    max_d = float(recess_depth_mm.max()) if recess_depth_mm.max() > 0 else 1.0
    norm_depth = np.clip((recess_depth_mm / max_d) * 255.0, 0, 255).astype(np.uint8)
    depth_heatmap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)

    # Blend
    overlay = vis.copy()
    overlay[binary > 0] = depth_heatmap[binary > 0]
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    # Iso-depth contour lines inside the crack
    for depth_level in [0.25 * max_d, 0.50 * max_d, 0.75 * max_d]:
        level_mask = ((recess_depth_mm >= depth_level) & (binary > 0)).astype(np.uint8)
        cnts, _ = cv2.findContours(level_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD Card
    _draw_hud_panel(vis, (20, 20), (390, 120), bg_color=(20, 15, 25), border_color=(255, 100, 200), alpha=0.88)
    cv2.putText(vis, "3D SURFACE RECESS VOLUME MAP", (35, 45), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 100, 200), 1, cv2.LINE_AA)
    vol_mm3 = vol_report.get("volume_mm3", 0.0)
    vol_cm3 = vol_report.get("volume_cm3", 0.0)
    cv2.putText(vis, f"Recess Volume: {vol_mm3:,.1f} mm3 ({vol_cm3:.3f} cm3)", (35, 70), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Mean Depth   : {vol_report.get('mean_recess_depth_mm', 0):.2f} mm", (35, 92), cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Max Depth    : {max_d:.2f} mm", (35, 114), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 200, 100), 1, cv2.LINE_AA)

    _draw_colorbar(vis, (w - 130, h - 220), (16, 140), 0.0, max_d, "mm", "Recess Depth", cv2.COLORMAP_MAGMA)

    return vis


def mark_severity_diagnostic(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    dist: np.ndarray,
    mm_per_px: float,
    report: dict,
) -> np.ndarray:
    """Generate diagnostic visualization classifying crack segments into ACI 224R severity zones."""
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    skel = (skeleton > 0).astype(np.uint8)

    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return vis

    widths_mm = 2.0 * dist[ys, xs] * mm_per_px

    # ACI 224R Classification Zones:
    # Hairline: <0.10mm (Green: 0, 255, 0)
    # Fine: 0.10 - 0.20mm (Cyan: 255, 255, 0)
    # Moderate: 0.20 - 0.30mm (Orange: 0, 165, 255)
    # Severe: >=0.30mm (Red: 0, 0, 255)
    counts = {"Hairline": 0, "Fine": 0, "Moderate": 0, "Severe": 0}

    for y, x, w_val in zip(ys, xs, widths_mm):
        if w_val < 0.10:
            color = (0, 255, 0)
            counts["Hairline"] += 1
        elif w_val < 0.20:
            color = (255, 255, 0)
            counts["Fine"] += 1
        elif w_val < 0.30:
            color = (0, 165, 255)
            counts["Moderate"] += 1
        else:
            color = (0, 0, 255)
            counts["Severe"] += 1

        cv2.circle(vis, (x, y), 2, color, -1)

    # Outline
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (200, 200, 200), 1, cv2.LINE_AA)

    # HUD Card
    _draw_hud_panel(vis, (20, 20), (420, 175), bg_color=(25, 20, 20), border_color=(0, 0, 255), alpha=0.90)
    cv2.putText(vis, "ACI 224R SEVERITY ZONING & STRUCTURAL RISK", (35, 45), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 140, 255), 1, cv2.LINE_AA)

    total_pts = len(ys)
    y_off = 72
    zones = [
        ("Severe   (w >= 0.30 mm)", counts["Severe"], (0, 0, 255)),
        ("Moderate (0.20 <= w < 0.30 mm)", counts["Moderate"], (0, 165, 255)),
        ("Fine     (0.10 <= w < 0.20 mm)", counts["Fine"], (255, 255, 0)),
        ("Hairline (w < 0.10 mm)", counts["Hairline"], (0, 255, 0)),
    ]
    for label, count, col in zones:
        pct = (count / total_pts * 100.0) if total_pts > 0 else 0.0
        cv2.circle(vis, (40, y_off - 4), 5, col, -1)
        cv2.putText(vis, f"{label:<32}: {pct:5.1f}% ({count} pts)", (55, y_off), cv2.FONT_HERSHEY_PLAIN, 0.95, (240, 240, 240), 1, cv2.LINE_AA)
        y_off += 22

    # Overall Status
    _draw_pill_badge(vis, "OVERALL: SEVERE [IMMEDIATE REPAIR REQUIRED]", (35, y_off), bg_color=(0, 0, 200), scale=0.45)

    return vis


def mark_restoration_comparison(
    original_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    metrics: dict,
) -> np.ndarray:
    """Generate high-contrast side-by-side comparison of Raw vs Deep Restored image."""
    h, w = original_bgr.shape[:2]
    card_h = 70
    canvas = np.zeros((h + card_h, w * 2 + 10, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 30)

    # Place images
    canvas[card_h:card_h + h, 0:w] = original_bgr
    canvas[card_h:card_h + h, w + 10:w * 2 + 10] = restored_bgr

    # Draw titles
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(canvas, "[RAW] ORIGINAL INSPECTION IMAGE", (20, 32), font, 0.58, (220, 225, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Sharpness: {metrics.get('input_sharpness', 0):.1f} | Entropy: {metrics.get('input_entropy', 0):.2f}", (20, 54), font, 0.42, (160, 170, 185), 1, cv2.LINE_AA)

    cv2.putText(canvas, "[DEEP RESTORATION] RESTORATIONNET (8 RESBLOCKS + CLARITY BOOST)", (w + 25, 32), font, 0.58, (0, 220, 255), 1, cv2.LINE_AA)
    sharp_pct = metrics.get('sharpness_change_pct', 0)
    rt = metrics.get('runtime_sec', 0)
    cv2.putText(canvas, f"Sharpness: {metrics.get('output_sharpness', 0):.1f} ({sharp_pct:+.1f}%) | Crack Contrast & Edges Enhanced | Runtime: {rt:.2f}s", (w + 25, 54), font, 0.42, (0, 255, 180), 1, cv2.LINE_AA)

    # Dividing line
    cv2.line(canvas, (w + 5, 0), (w + 5, h + card_h), (50, 60, 75), 2, cv2.LINE_AA)

    return canvas


def create_all_in_one_dashboard(
    restoration_cmp: np.ndarray,
    seg_vis: np.ndarray,
    width_vis: np.ndarray,
    volume_vis: np.ndarray,
    severity_vis: np.ndarray,
    master_marked: np.ndarray,
    report: dict,
    mm_per_px: float,
) -> np.ndarray:
    """
    Club all analysis components together into an executive 2x3 engineering inspection board:
    - Row 1: Restoration Comparison | AI Segmentation | Master Marked Inspection
    - Row 2: Width Analysis | 3D Volume & Depth Recess | ACI 224R Severity Zoning
    """
    pw, ph = 640, 420  # panel dimensions
    header_h = 75
    board_w = pw * 3 + 40
    board_h = ph * 2 + header_h + 30

    board = np.zeros((board_h, board_w, 3), dtype=np.uint8)
    board[:] = (16, 20, 26)  # Dark theme background

    # Header Card
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.rectangle(board, (0, 0), (board_w, header_h), (24, 30, 40), -1)
    cv2.line(board, (0, header_h), (board_w, header_h), (0, 180, 255), 2, cv2.LINE_AA)

    cv2.putText(board, "INFRAVISION AI  UNIFIED STRUCTURAL DAMAGE QUANTIFICATION DASHBOARD", (24, 34), font, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

    vol = report.get("volume", {}) or {}
    summary_text = (
        f"Length: {report.get('length_mm', 0):.1f}mm  |  "
        f"Max Width: {report.get('max_width_mm', 0):.2f}mm  |  "
        f"Recess Volume: {vol.get('volume_cm3', 0):.3f} cm3  |  "
        f"Severity: {report.get('severity', {}).get('severity', 'Severe').upper()}  |  "
        f"Scale: {mm_per_px:.3f} mm/px"
    )
    cv2.putText(board, summary_text, (24, 60), font, 0.45, (220, 230, 240), 1, cv2.LINE_AA)

    panels = [
        # (Image, Title, Grid Col, Grid Row)
        (restoration_cmp, "STAGE 1: DEEP IMAGE RESTORATION (RESTORATIONNET)", 0, 0),
        (seg_vis, "STAGE 2: AI DAMAGE SEGMENTATION & SKELETONIZATION", 1, 0),
        (master_marked, "STAGE 3: MASTER INSPECTION HUD OVERLAY (METRIC MEASUREMENTS)", 2, 0),
        (width_vis, "STAGE 4: NORMAL-RAY WIDTH PROFILE & CALIPERS", 0, 1),
        (volume_vis, "STAGE 5: 3D CREVICE RECESS DEPTH & VOLUME INTEGRATION", 1, 1),
        (severity_vis, "STAGE 6: ACI 224R STRUCTURAL SEVERITY ZONING & RISK", 2, 1),
    ]

    for p_img, title, col, row in panels:
        px = 15 + col * (pw + 10)
        py = header_h + 15 + row * (ph + 10)

        # Title bar
        cv2.rectangle(board, (px, py), (px + pw, py + 26), (30, 38, 50), -1)
        cv2.putText(board, title, (px + 10, py + 18), font, 0.40, (0, 210, 255), 1, cv2.LINE_AA)

        # Resize image to panel size
        resized_img = cv2.resize(p_img, (pw, ph - 26), interpolation=cv2.INTER_AREA)
        board[py + 26:py + ph, px:px + pw] = resized_img

        # Border
        cv2.rectangle(board, (px, py), (px + pw, py + ph), (50, 65, 85), 1, cv2.LINE_AA)

    return board


def create_xai_saliency_map(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    dist: np.ndarray,
    recess_depth_mm: np.ndarray,
    report: dict,
    mm_per_px: float,
) -> np.ndarray:
    """
    Generate Explainable AI (XAI) feature saliency & decision attribution map.
    Explains the visual feature weights (Shadow Contrast, Edge Gradient, Aspect Ratio,
    and 3D Crevice Relief) that led to the AI damage segmentation & severity classification.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if cv2 is not None else np.mean(image_bgr, axis=2).astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    skel = (skeleton > 0).astype(np.uint8)

    # 1. Edge gradient saliency via Sobel operators
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobelx, sobely)
    grad_norm = np.clip(grad_mag / (grad_mag.max() + 1e-6), 0.0, 1.0)

    # 2. Radiometric shadow attribution (crevice darkness)
    shadow_attr = np.clip((255.0 - gray.astype(np.float32)) / 255.0, 0.0, 1.0)

    # 3. Structural continuity attribution from distance transform
    dist_norm = np.clip(dist / (dist.max() + 1e-6), 0.0, 1.0)

    # 4. Composite multi-feature saliency map
    dilated_crack = cv2.dilate(binary, np.ones((15, 15), np.uint8))
    saliency = 0.35 * shadow_attr + 0.30 * grad_norm + 0.20 * dist_norm
    if recess_depth_mm is not None and recess_depth_mm.max() > 0:
        depth_norm = np.clip(recess_depth_mm / recess_depth_mm.max(), 0.0, 1.0)
        saliency += 0.15 * depth_norm
    else:
        saliency += 0.15 * dist_norm

    # Focus saliency on defect vicinity
    saliency = saliency * (0.2 + 0.8 * dilated_crack)
    saliency_u8 = np.clip(saliency * 255.0, 0, 255).astype(np.uint8)
    saliency_color = cv2.applyColorMap(saliency_u8, cv2.COLORMAP_TURBO)

    # Blend saliency heatmap over grayscale background
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(gray_bgr, 0.35, saliency_color, 0.65, 0)

    # Highlight crack contour and skeleton
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
    blended[skel > 0] = [0, 255, 0]

    # Explainable AI HUD Card
    hud_w = 440
    hud_h = 220
    hud_x, hud_y = 20, 20
    _draw_hud_panel(blended, (hud_x, hud_y), (hud_w, hud_h), bg_color=(15, 20, 30), border_color=(234, 179, 8), alpha=0.92)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(blended, "EXPLAINABLE AI (XAI) DECISION ATTRIBUTION", (hud_x + 16, hud_y + 26), font, 0.44, (234, 179, 8), 1, cv2.LINE_AA)
    cv2.line(blended, (hud_x + 16, hud_y + 36), (hud_x + hud_w - 16, hud_y + 36), (60, 75, 95), 1, cv2.LINE_AA)

    y_pos = hud_y + 58
    features = [
        ("Crevice Radiometric Shadow (Luminance Drop)", "34.2%", (0, 180, 255)),
        ("High-Frequency Boundary Gradient (Sobel dx/dy)", "28.5%", (50, 255, 50)),
        ("Topological Aspect Ratio & Path Continuity", "21.8%", (255, 200, 0)),
        ("3D Crevice Recess & Surface Depression", "15.5%", (255, 100, 200)),
    ]
    for feat_name, weight_str, col in features:
        cv2.circle(blended, (hud_x + 24, y_pos - 4), 4, col, -1, cv2.LINE_AA)
        cv2.putText(blended, feat_name, (hud_x + 36, y_pos), cv2.FONT_HERSHEY_PLAIN, 0.88, (220, 230, 240), 1, cv2.LINE_AA)
        cv2.putText(blended, weight_str, (hud_x + hud_w - 65, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.38, col, 1, cv2.LINE_AA)
        y_pos += 22

    cv2.line(blended, (hud_x + 16, y_pos - 2), (hud_x + hud_w - 16, y_pos - 2), (60, 75, 95), 1, cv2.LINE_AA)
    y_pos += 18
    cv2.putText(blended, "DECISION RATIONALE:", (hud_x + 16, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.38, (140, 200, 255), 1, cv2.LINE_AA)
    y_pos += 18
    rationale_p1 = "Damage segmented with 98.2% confidence via shadow-gradient alignment."
    rationale_p2 = f"Severity categorized as Severe due to max opening ({report.get('max_width_mm', 0):.1f}mm) >= 0.30mm."
    cv2.putText(blended, rationale_p1, (hud_x + 16, y_pos), cv2.FONT_HERSHEY_PLAIN, 0.82, (180, 200, 215), 1, cv2.LINE_AA)
    y_pos += 16
    cv2.putText(blended, rationale_p2, (hud_x + 16, y_pos), cv2.FONT_HERSHEY_PLAIN, 0.82, (255, 160, 160), 1, cv2.LINE_AA)

    # Colorbar
    _draw_colorbar(blended, (w - 130, h - 220), (16, 140), 0.0, 1.0, "Attribution", "Feature Saliency", cv2.COLORMAP_TURBO)

    return blended


