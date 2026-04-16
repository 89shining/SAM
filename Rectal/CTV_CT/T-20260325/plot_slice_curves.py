"""
根据merge_slice_metrics_total绘制患者的slice级别评估图
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ID_PATTERN = re.compile(r"(\d+)")

METHOD_ORDER = [
    "nnunet_all",
    "nnunet_crop",
    "nnunet_crop_SAMmask",
    "nnunet_crop_SAMbox",
]

METHOD_STYLES = {
    "nnunet_all": {"color": "dodgerblue", "marker": "o"},
    "nnunet_crop": {"color": "darkorange", "marker": "s"},
    "nnunet_crop_SAMmask": {"color": "forestgreen", "marker": "^"},
    "nnunet_crop_SAMbox": {"color": "crimson", "marker": "D"},
}


def to_int_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def to_float_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def get_patient_id(row):
    if "case_id" in row:
        case_id = to_int_or_none(row.get("case_id"))
        if case_id is not None:
            return case_id

    for key in ("case", "pred_file", "gt_case_dir"):
        if key in row and str(row.get(key, "")).strip() != "":
            m = ID_PATTERN.search(str(row[key]))
            if m:
                return int(m.group(1))

    raise ValueError(f"Cannot parse patient id from row: {row}")


def normalize_rows(rows):
    out = []
    for row in rows:
        patient_id = get_patient_id(row)

        rel_low = to_int_or_none(row.get("z_relative_to_gt_lower"))
        rel_up = to_int_or_none(row.get("z_relative_to_gt_upper"))
        gt_low = to_int_or_none(row.get("gt_lower_z"))
        gt_up = to_int_or_none(row.get("gt_upper_z"))
        z_abs = to_int_or_none(row.get("z"))
        gt_nonempty = to_int_or_none(row.get("gt_nonempty"))
        if gt_nonempty is None:
            gt_nonempty = 0

        if rel_low is None or rel_up is None:
            if z_abs is not None and gt_low is not None and gt_up is not None:
                rel_low = z_abs - gt_low
                rel_up = z_abs - gt_up
            else:
                continue

        if gt_low is not None and gt_up is not None:
            upper_idx = gt_up - gt_low
        else:
            upper_idx = -rel_up

        out.append(
            {
                "patient_id": patient_id,
                "rel_low": rel_low,
                "rel_up": rel_up,
                "cur_z": rel_low,
                "lower_idx": 0,
                "upper_idx": upper_idx,
                "gt_nonempty": gt_nonempty,
                "dice": to_float_or_none(row.get("dice_2d")),
                "hd95": to_float_or_none(row.get("hd95_2d_mm")),
            }
        )
    return out


def build_map(rows):
    m = {}
    for r in rows:
        key = f"{r['patient_id']}|{r['rel_low']}|{r['rel_up']}"
        m[key] = r
    return m


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_nice_interval(raw_interval: float, is_dice: bool) -> float:
    if is_dice:
        for c in (0.02, 0.05, 0.1, 0.2):
            if raw_interval <= c:
                return c
        return 0.2

    for c in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if raw_interval <= c:
            return c
    return 100.0


def get_y_axis_range(values, is_dice: bool):
    if len(values) == 0:
        if is_dice:
            return 0.0, 1.0, 0.1
        return 0.0, 1.0, 0.2

    vmin = min(values)
    vmax = max(values)
    value_range = vmax - vmin

    if value_range < 1e-9:
        if is_dice:
            pad = 0.03
        else:
            pad = max(0.5, abs(vmax) * 0.1)
    else:
        pad_floor = 0.02 if is_dice else 0.2
        pad = max(value_range * 0.15, pad_floor)

    ymin = vmin - pad
    ymax = vmax + pad

    if is_dice:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
        if (ymax - ymin) < 0.06:
            mid = (ymax + ymin) / 2.0
            ymin = max(0.0, mid - 0.03)
            ymax = min(1.0, mid + 0.03)
    elif (ymax - ymin) < 0.5:
        mid = (ymax + ymin) / 2.0
        ymin = mid - 0.25
        ymax = mid + 0.25

    raw_interval = (ymax - ymin) / 6.0
    if raw_interval <= 0:
        raw_interval = 0.1 if is_dice else 0.5
    interval = get_nice_interval(raw_interval, is_dice=is_dice)

    ymin = math.floor(ymin / interval) * interval
    ymax = math.ceil(ymax / interval) * interval
    if ymax <= ymin:
        ymax = ymin + interval * 2

    if is_dice:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
        if ymax <= ymin:
            ymax = min(1.0, ymin + interval * 2)

    return ymin, ymax, interval


def get_x_ticks(x_min: int, x_max: int, x_interval: int):
    ticks = list(range(x_min, x_max + 1, x_interval))
    if len(ticks) == 0 or ticks[-1] != x_max:
        ticks.append(x_max)
    return ticks


def main():
    parser = argparse.ArgumentParser(description="Plot slice-wise Dice/HD95 curves for each patient.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"D:\SAM\Rectal\CTV\146p\20260325\Slice_index"),
        help="Directory containing 4 slice-level CSV files.",
    )
    args = parser.parse_args()

    paths = {
        "nnunet_all": args.base_dir / "slice_nnunet_all.csv",
        "nnunet_crop": args.base_dir / "slice_nnunet_crop.csv",
        "nnunet_crop_SAMmask": args.base_dir / "slice_nnunet_crop_SAMmask.csv",
        "nnunet_crop_SAMbox": args.base_dir / "slice_nnunet_crop_SAMbox.csv",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV for {name}: {p}")

    data = {}
    for method in METHOD_ORDER:
        rows = read_csv_rows(paths[method])
        norm = normalize_rows(rows)
        m = build_map(norm)
        data[method] = m
        print(f"[Info] {method} rows={len(rows)} normalized={len(norm)} keys={len(m)}")

    gt_crop = {k for k, v in data["nnunet_crop"].items() if int(v["gt_nonempty"]) == 1}
    gt_mask = {k for k, v in data["nnunet_crop_SAMmask"].items() if int(v["gt_nonempty"]) == 1}
    gt_box = {k for k, v in data["nnunet_crop_SAMbox"].items() if int(v["gt_nonempty"]) == 1}

    print(f"[Info] GT keys nnunet_crop = {len(gt_crop)}")
    print(f"[Info] GT keys nnunet_crop_SAMmask = {len(gt_mask)}")
    print(f"[Info] GT keys nnunet_crop_SAMbox = {len(gt_box)}")

    keep_keys = gt_crop & gt_mask & gt_box
    print(f"[Info] keep keys = {len(keep_keys)}")

    patient_map = {}
    for key in keep_keys:
        patient_id = int(key.split("|")[0])
        patient_map.setdefault(patient_id, []).append(key)

    dice_dir = args.base_dir / "Dice_2d"
    hd_dir = args.base_dir / "HD95_2d"
    dice_dir.mkdir(parents=True, exist_ok=True)
    hd_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(patient_map.keys())
    total = len(patient_ids)

    for idx, patient_id in enumerate(patient_ids, start=1):
        keys = patient_map[patient_id]

        rows_for_x = []
        for key in keys:
            ref = data["nnunet_crop"].get(key)
            if ref is None:
                ref = data["nnunet_crop_SAMmask"].get(key)
            if ref is None:
                ref = data["nnunet_crop_SAMbox"].get(key)
            if ref is not None:
                rows_for_x.append(ref)
        if len(rows_for_x) == 0:
            continue

        x_vals = sorted(int(r["cur_z"]) for r in rows_for_x)
        x_min = min(x_vals)
        x_max = max(x_vals)
        if x_max <= x_min:
            x_max = x_min + 1
        x_interval = 4
        x_ticks = get_x_ticks(x_min, x_max, x_interval)

        for metric in ("dice", "hd95"):
            is_dice = metric == "dice"

            all_y = []
            for method in METHOD_ORDER:
                for key in keys:
                    r = data[method].get(key)
                    if r is None:
                        continue
                    v = r[metric]
                    if v is not None:
                        all_y.append(float(v))

            y_min, y_max, y_interval = get_y_axis_range(all_y, is_dice=is_dice)
            y_ticks = []
            cur = y_min
            guard = 0
            while cur <= y_max + 1e-9 and guard < 1000:
                y_ticks.append(round(cur, 10))
                cur += y_interval
                guard += 1

            title_metric = "Dice" if is_dice else "HD95"
            y_title = "Dice coefficient" if is_dice else "HD95 (mm)"
            case_name = f"p_{patient_id:02d}"

            fig, ax = plt.subplots(figsize=(24, 13), dpi=100)
            fig.patch.set_facecolor("white")
            ax.set_facecolor((248 / 255, 248 / 255, 248 / 255))

            for method in METHOD_ORDER:
                pts = []
                for key in keys:
                    r = data[method].get(key)
                    if r is None:
                        continue
                    v = r[metric]
                    if v is None:
                        continue
                    pts.append((int(r["cur_z"]), float(v)))
                pts.sort(key=lambda t: t[0])

                if len(pts) == 0:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(
                    xs,
                    ys,
                    label=method,
                    color=METHOD_STYLES[method]["color"],
                    marker=METHOD_STYLES[method]["marker"],
                    linewidth=3.0,
                    markersize=8,
                )

            ax.set_title(f"Slice-wise {title_metric} Curve (Case {case_name})", fontsize=24, fontweight="bold")
            ax.set_xlabel("Slice position (Lower -> Upper)", fontsize=20, fontweight="bold")
            ax.set_ylabel(y_title, fontsize=20, fontweight="bold")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.grid(True, color=(220 / 255, 220 / 255, 220 / 255), linewidth=1)
            ax.tick_params(axis="both", labelsize=16)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=16)

            plt.tight_layout(rect=(0, 0.04, 1, 1))

            out_dir = dice_dir if is_dice else hd_dir
            suffix = "dice_2d" if is_dice else "hd95_2d"
            out_path = out_dir / f"{case_name}_{suffix}.png"
            fig.savefig(out_path, dpi=100)
            plt.close(fig)

        print(f"[Progress] {idx}/{total} done: p_{patient_id:02d}")

    print(f"[Done] Dice plots: {dice_dir}")
    print(f"[Done] HD95 plots: {hd_dir}")


if __name__ == "__main__":
    main()
