import json

data = {
    "boxes": [
        {"slice_index": 32, "x0": 240, "y0": 264, "x1": 269, "y1": 292},
        {"slice_index": 37, "x0": 232, "y0": 258, "x1": 274, "y1": 299},
        {"slice_index": 42, "x0": 241, "y0": 269, "x1": 274, "y1": 303}
    ]
}

with open(r"C:\Users\dell\Desktop\boxes.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("保存完成：C:\\Users\\dell\\Desktop\\boxes.json")