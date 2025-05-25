import shutil
from pathlib import Path

# === 設定來源與目標根目錄 ===
source_root = Path("7SCENES")                   # 原始根目錄
target_root = Path("RGB")             # 複製後的根目錄

# 遍歷所有 .color.png 檔案（含子資料夾）
for path in source_root.rglob("*.color.png"):
    # 取得相對路徑（保留原始結構）
    relative_path = path.relative_to(source_root)
    target_path = target_root / relative_path

    # 確保目標資料夾存在
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 複製檔案
    shutil.copy(path, target_path)
    print(f"複製: {path} → {target_path}")

print("✅ 所有 .color.png 檔案已複製並保留原始結構。")
