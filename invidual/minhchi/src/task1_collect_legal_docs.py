"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = {
    "73luat.pdf": {
        "title": "Luật Phòng, chống ma tuý 2021",
        "note": "Luật số 73/2021/QH15",
    },
    "105.signed_02.pdf": {
        "title": "Nghị định 105/2021/NĐ-CP",
        "note": "Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Phòng, chống ma tuý",
    },
    "2026_82_28_2026_NĐ-CP.docx": {
        "title": "Nghị định 82/2026/NĐ-CP",
        "note": "Văn bản pháp luật hiện có trong data/landing/legal/",
    },
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def validate_legal_documents() -> list[Path]:
    """
    Kiểm tra 3 văn bản pháp luật hiện có trong data/landing/legal/.

    Task 1 yêu cầu lưu file gốc PDF/DOCX vào data/landing/legal/.
    Hàm này không tạo dữ liệu mẫu và không ghi đè file, chỉ xác nhận
    các văn bản đã được đặt đúng tên và có nội dung.
    """
    setup_directory()
    valid_files = []
    missing_files = []

    for filename, metadata in LEGAL_DOCUMENTS.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
            print(f"✗ Thiếu file: {filename}")
            continue

        size_kb = filepath.stat().st_size / 1024
        if size_kb <= 1:
            print(f"✗ File quá nhỏ, cần kiểm tra lại: {filename} ({size_kb:.1f} KB)")
            continue

        valid_files.append(filepath)
        print(f"✓ {metadata['title']}")
        print(f"  File: {filename} ({size_kb:.1f} KB)")
        print(f"  Ghi chú: {metadata['note']}")

    if missing_files:
        raise FileNotFoundError(
            "Thiếu văn bản pháp luật trong data/landing/legal/: "
            + ", ".join(missing_files)
        )

    if len(valid_files) < 3:
        raise ValueError("Cần tối thiểu 3 file pháp luật hợp lệ (>1KB).")

    print(f"\n✓ Task 1 hoàn tất: tìm thấy {len(valid_files)} văn bản pháp luật.")
    return valid_files


if __name__ == "__main__":
    validate_legal_documents()
