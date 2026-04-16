"""
汇总 slice_wise_index 结果（2 methods）。
"""

import argparse
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ID_PATTERN = re.compile(r"(\d+)")

METHOD_ORDER = [
    "nnunet_crop",
    "nnunet_crop_Pmap",
]


def to_int(value, default=None):
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def to_float(value, default=math.nan):
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_patient_id(row, file_tag):
    if "case_id" in row and str(row["case_id"]).strip() != "":
        return to_int(row["case_id"])

    for key in ("case", "pred_file", "gt_case_dir"):
        if key in row and str(row[key]).strip() != "":
            m = ID_PATTERN.search(str(row[key]))
            if m:
                return int(m.group(1))

    raise ValueError(f"Cannot parse patient id from {file_tag}: {row}")


def read_csv_rows(path: Path):
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows, file_tag):
    out = []
    for row in rows:
        pid = parse_patient_id(row, file_tag)
        rel_low = to_int(row.get("z_relative_to_gt_lower"))
        rel_up = to_int(row.get("z_relative_to_gt_upper"))
        gt_low = to_int(row.get("gt_lower_z"))
        gt_up = to_int(row.get("gt_upper_z"))
        gt_nonempty = to_int(row.get("gt_nonempty"), default=0)

        if rel_low is None or rel_up is None:
            z_abs = to_int(row.get("z"))
            if z_abs is None or gt_low is None or gt_up is None:
                continue
            rel_low = z_abs - gt_low
            rel_up = z_abs - gt_up

        if gt_low is not None and gt_up is not None:
            upper_idx = gt_up - gt_low
        else:
            upper_idx = -rel_up

        out.append(
            {
                "patient_id": pid,
                "rel_low": rel_low,
                "rel_up": rel_up,
                "cur_z": rel_low,
                "lower_idx": 0,
                "upper_idx": upper_idx,
                "gt_nonempty": gt_nonempty,
                "dice": to_float(row.get("dice_2d")),
                "hd95": to_float(row.get("hd95_2d_mm")),
            }
        )
    return out


def build_map(norm_rows):
    m = {}
    for r in norm_rows:
        key = (r["patient_id"], r["rel_low"], r["rel_up"])
        m[key] = r
    return m


def fmt2(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.2f}"


def col_letter(col_idx_1based: int) -> str:
    n = col_idx_1based
    chars = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def make_cell_xml(r: int, c: int, value, style_idx: int) -> str:
    ref = f"{col_letter(c)}{r}"
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style_idx}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" s="{style_idx}" t="inlineStr"><is><t>{text}</t></is></c>'


def build_sheet_xml(table_rows):
    rows_xml = []
    max_col = 8
    for r_idx, row in enumerate(table_rows, start=1):
        style_idx = 1 if r_idx <= 2 else 0
        cells_xml = []
        for c_idx in range(1, max_col + 1):
            v = row[c_idx - 1] if c_idx - 1 < len(row) else ""
            cell_xml = make_cell_xml(r_idx, c_idx, v, style_idx)
            if cell_xml:
                cells_xml.append(cell_xml)
        rows_xml.append(f'<row r="{r_idx}">' + "".join(cells_xml) + "</row>")

    merges = ["A1:A2", "B1:B2", "C1:C2", "D1:D2", "E1:F1", "G1:H1"]
    merge_xml = "".join([f'<mergeCell ref="{m}"/>' for m in merges])

    last_row = max(1, len(table_rows))

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:H{last_row}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols>'
        '<col min="1" max="1" width="10" customWidth="1"/>'
        '<col min="2" max="4" width="12" customWidth="1"/>'
        '<col min="5" max="8" width="18" customWidth="1"/>'
        '</cols>'
        '<sheetData>'
        + "".join(rows_xml)
        + '</sheetData>'
        f'<mergeCells count="{len(merges)}">{merge_xml}</mergeCells>'
        '</worksheet>'
    )


def write_xlsx(table_rows, out_xlsx: Path):
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="total" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )

    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center"/>'
        '</xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    sheet = build_sheet_xml(table_rows)

    with zipfile.ZipFile(out_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


def main():
    parser = argparse.ArgumentParser(description="Merge 2 slice-wise CSVs and export formatted Excel table.")
    parser.add_argument("--base-dir", type=Path, default=Path(r"C:\Users\dell\Desktop\20260401"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\dell\Desktop\20260401\slice_total_2methods.xlsx"),
    )
    args = parser.parse_args()

    paths = {
        "nnunet_crop": args.base_dir / "slice_nnunet_crop.csv",
        "nnunet_crop_Pmap": args.base_dir / "slice_nnunet_crop_Pmap.csv",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV for {name}: {p}")

    data = {}
    for name in METHOD_ORDER:
        rows = read_csv_rows(paths[name])
        norm = normalize_rows(rows, file_tag=name)
        data[name] = build_map(norm)
        print(f"[Info] {name}: raw={len(rows)}, normalized={len(norm)}, unique_keys={len(data[name])}")

    key_sets = []
    for name in METHOD_ORDER:
        gt_keys = {k for k, v in data[name].items() if v["gt_nonempty"] == 1}
        key_sets.append(gt_keys)
        print(f"[Info] {name}: GT keys={len(gt_keys)}")

    keep_keys = set.intersection(*key_sets) if key_sets else set()
    print(f"[Info] Keep keys (intersection of 2 GT layers): {len(keep_keys)}")

    sorted_keys = sorted(list(keep_keys), key=lambda x: (x[0], x[1], x[2]))

    header1 = [
        "patient_id",
        "current_z",
        "lower_idx",
        "upper_idx",
        "dice_2d",
        "",
        "hd95_2d_mm",
        "",
    ]
    header2 = [
        "",
        "",
        "",
        "",
        "nnunet_crop",
        "nnunet_crop_Pmap",
        "nnunet_crop",
        "nnunet_crop_Pmap",
    ]

    table_rows = [header1, header2]

    for key in sorted_keys:
        pid, _, _ = key

        ref = data["nnunet_crop"].get(key)
        if ref is None:
            ref = data["nnunet_crop_Pmap"].get(key)
        if ref is None:
            continue

        row = [pid, ref["cur_z"], ref["lower_idx"], ref["upper_idx"]]

        dice_vals = []
        hd_vals = []
        for name in METHOD_ORDER:
            v = data[name].get(key)
            if v is None:
                dice_vals.append("")
                hd_vals.append("")
            else:
                dice_vals.append(fmt2(v["dice"]))
                hd_vals.append(fmt2(v["hd95"]))

        row.extend(dice_vals)
        row.extend(hd_vals)
        table_rows.append(row)

    write_xlsx(table_rows, args.out)

    print(f"[Done] Saved merged Excel: {args.out}")


if __name__ == "__main__":
    main()
