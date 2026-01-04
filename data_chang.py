import os
import shutil
from pathlib import Path


def reorganize_mobiphone_dataset(raw_root: str, target_root: str, move_files: bool = False):
    """
    重构MOBIPHONE数据集目录结构

    Args:
        raw_root: 原始MOBIPHONE数据集根目录（如包含MOBIPHONE文件夹的路径）
        target_root: 目标数据集根目录（重构后的目录根路径）
        move_files: 是否移动文件（False=复制，True=移动，默认复制避免原数据丢失）
    """
    # 规范化路径
    raw_root = Path(raw_root).resolve()
    target_root = Path(target_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    # 遍历原始目录结构：MOBIPHONE -> 4800 -> 机型 -> speaker* -> 音频文件
    mobiphone_dir = raw_root / "MOBIPHONE" / "4800"
    if not mobiphone_dir.exists():
        raise FileNotFoundError(f"原始数据集路径不存在：{mobiphone_dir}")

    # 遍历所有机型目录（如HTC Desire C、LG G5290等）
    for device_dir in mobiphone_dir.iterdir():
        if not device_dir.is_dir():
            continue  # 跳过非目录文件

        # 提取品牌和机型（如HTC Desire C拆分为品牌HTC，机型Desire C）
        device_name = device_dir.name
        brand = device_name.split()[0] if " " in device_name else device_name
        model = device_name[len(brand) + 1:] if " " in device_name else device_name

        # 遍历当前机型下的所有speaker目录
        for speaker_dir in device_dir.iterdir():
            if not speaker_dir.is_dir() or not speaker_dir.name.startswith("speaker"):
                continue  # 跳过非speaker目录

            # 提取扬声器ID（如speaker1 -> ID=1，speaker10 -> ID=10）
            speaker_id = speaker_dir.name.replace("speaker", "")

            # 构建目标目录：目标根目录/品牌/机型/speaker{ID}
            target_speaker_dir = target_root / brand / model / f"speaker{speaker_id}"
            target_speaker_dir.mkdir(parents=True, exist_ok=True)

            # 复制/移动所有wav音频文件到目标目录
            for wav_file in speaker_dir.glob("*.wav"):
                target_file = target_speaker_dir / wav_file.name
                if move_files:
                    shutil.move(str(wav_file), str(target_file))
                else:
                    shutil.copy2(str(wav_file), str(target_file))  # 保留文件元信息

            print(f"已处理：{device_name} -> speaker{speaker_id} -> {target_speaker_dir}")

    print(f"数据集重构完成！目标路径：{target_root}")


# -------------------------- 执行示例 --------------------------
if __name__ == "__main__":
    # 配置原始数据集路径和目标路径（请根据实际情况修改）
    RAW_DATA_ROOT = r"E:\Grade 1 master\Small-scale Meetings\RASA\MOBIPHONE"  # 原始数据根目录（包含MOBIPHONE文件夹）
    TARGET_DATA_ROOT = r"E:\Grade 1 master\Small-scale Meetings\RASA\MOBIPHONE_CHANGE"  # 重构后的目标目录

    # 执行重构（默认复制文件，如需移动请将move_files设为True）
    try:
        reorganize_mobiphone_dataset(
            raw_root=RAW_DATA_ROOT,
            target_root=TARGET_DATA_ROOT,
            move_files=False  # 建议先复制，验证无误后再改为移动
        )
    except Exception as e:
        print(f"执行出错：{e}")