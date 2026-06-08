"""Create a small offline dataset for end-to-end RAG demos and tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import LANDING_DIR, STANDARDIZED_DIR


LEGAL_DOCS: dict[str, str] = {
    "luat-phong-chong-ma-tuy-2021.doc": """# Luật Phòng, chống ma túy 2021 - trích mẫu học tập

Điều 3. Giải thích từ ngữ.
Ma túy là các chất gây nghiện, chất hướng thần được quy định trong danh mục do Chính phủ ban hành. Người sử dụng trái phép chất ma túy là người dùng chất ma túy mà không được cơ quan có thẩm quyền cho phép. Cơ quan, tổ chức, gia đình và cá nhân có trách nhiệm tham gia phòng, chống ma túy theo quy định của pháp luật.

Điều 4. Nguyên tắc phòng, chống ma túy.
Phòng, chống ma túy phải kết hợp giữa phòng ngừa, ngăn chặn, đấu tranh, xử lý vi phạm và hỗ trợ điều trị, cai nghiện, quản lý sau cai nghiện. Hoạt động phòng, chống ma túy phải tôn trọng quyền con người, bảo đảm bí mật đời tư của người được tư vấn, điều trị và cai nghiện theo quy định.

Điều 28. Các hình thức cai nghiện ma túy.
Cai nghiện ma túy gồm cai nghiện tự nguyện tại gia đình, cai nghiện tự nguyện tại cộng đồng, cai nghiện tự nguyện tại cơ sở cai nghiện ma túy và cai nghiện bắt buộc tại cơ sở cai nghiện ma túy. Việc lựa chọn hình thức cai nghiện phải căn cứ vào tình trạng nghiện, điều kiện của người nghiện và quyết định của cơ quan có thẩm quyền trong trường hợp bắt buộc.

Điều 32. Quản lý sau cai nghiện.
Người hoàn thành cai nghiện được hỗ trợ tái hòa nhập cộng đồng, tư vấn, học nghề, tìm việc làm và được gia đình, cộng đồng phối hợp quản lý, giúp đỡ để hạn chế tái nghiện. Nội dung này chỉ nhằm mục đích học tập về RAG và không thay thế văn bản pháp luật chính thức.
""",
    "bo-luat-hinh-su-2015-dieu-249.doc": """# Bộ luật Hình sự 2015 sửa đổi 2017 - Điều 249 trích mẫu học tập

Điều 249. Tội tàng trữ trái phép chất ma túy.
Người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển hoặc sản xuất trái phép chất ma túy thì tùy loại chất, khối lượng và tình tiết cụ thể có thể bị truy cứu trách nhiệm hình sự. Khung cơ bản thường được mô tả là phạt tù từ 01 năm đến 05 năm đối với một số trường hợp có khối lượng ở mức thấp theo luật định.

Khoản 2 Điều 249.
Phạm tội thuộc trường hợp có tổ chức, phạm tội nhiều lần, lợi dụng chức vụ quyền hạn, hoặc có khối lượng chất ma túy lớn hơn mức khung cơ bản thì hình phạt có thể tăng nặng. Người học cần đối chiếu văn bản chính thức để xác định đúng loại chất, khối lượng và khung hình phạt.

Khoản 3 Điều 249.
Trường hợp khối lượng chất ma túy ở mức rất lớn hoặc có tình tiết đặc biệt nghiêm trọng thì hình phạt có thể cao hơn. Nội dung mẫu này phục vụ truy xuất thông tin pháp luật, không phải tư vấn pháp lý và không khuyến khích bất kỳ hành vi vi phạm nào.

Điều 251. Tội mua bán trái phép chất ma túy.
Hành vi mua bán trái phép chất ma túy bị xử lý nghiêm khắc hơn tàng trữ trong nhiều trường hợp. Hệ thống RAG phải từ chối hướng dẫn mua bán, vận chuyển, sản xuất hoặc che giấu ma túy.
""",
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.doc": """# Nghị định 57/2022/NĐ-CP - danh mục chất ma túy trích mẫu học tập

Điều 1. Phạm vi điều chỉnh.
Nghị định ban hành danh mục các chất ma túy và tiền chất dùng trong quản lý nhà nước. Danh mục được chia thành các nhóm nhằm phục vụ kiểm soát, phòng ngừa và xử lý hành vi vi phạm pháp luật.

Điều 2. Nhóm I.
Nhóm I gồm các chất ma túy tuyệt đối cấm sử dụng trong y học và đời sống xã hội theo quy định pháp luật. Ví dụ thường được nhắc đến trong tài liệu học tập gồm heroin, cocaine, methamphetamine, MDMA và cannabis. Khi trả lời, hệ thống phải nêu rõ đây là thông tin từ context và khuyến nghị kiểm tra văn bản chính thức.

Điều 3. Tiền chất.
Tiền chất là hóa chất được kiểm soát vì có thể bị lợi dụng vào việc sản xuất trái phép chất ma túy. Hệ thống không được cung cấp hướng dẫn điều chế, công thức, quy trình sản xuất hoặc né tránh quản lý.

Điều 4. Cập nhật danh mục.
Danh mục chất ma túy và tiền chất có thể được sửa đổi, bổ sung theo từng giai đoạn. Người dùng cần kiểm tra văn bản mới nhất từ cơ quan có thẩm quyền khi cần áp dụng trong thực tế.
""",
}


NEWS_ARTICLES: list[dict[str, str]] = [
    {
        "slug": "nghe-si-a-bi-xu-phat",
        "title": "Nghệ sĩ A bị xử phạt hành chính trong vụ việc sử dụng chất cấm",
        "url": "https://example.com/news/nghe-si-a-bi-xu-phat",
        "content": """Bài báo mẫu cho biết Nghệ sĩ A bị cơ quan chức năng xử phạt hành chính sau khi có kết quả kiểm tra liên quan đến sử dụng trái phép chất ma túy. Vụ việc được đưa tin như một cảnh báo về trách nhiệm của người nổi tiếng trước công chúng.

Nguồn tin nhấn mạnh cơ quan chức năng không công bố các chi tiết đời tư ngoài phạm vi cần thiết. Bài viết khuyến nghị nghệ sĩ và người quản lý nâng cao nhận thức pháp luật, tránh cổ súy hoặc bình thường hóa hành vi sử dụng chất cấm.""",
    },
    {
        "slug": "chuong-trinh-truyen-hinh-tam-dung",
        "title": "Một chương trình giải trí tạm dừng ghi hình sau vụ việc ma túy",
        "url": "https://example.com/news/chuong-trinh-tam-dung",
        "content": """Bài báo mẫu mô tả một chương trình giải trí tạm dừng ghi hình để rà soát hợp đồng sau khi một khách mời bị điều tra trong vụ việc liên quan đến ma túy. Nhà sản xuất cho biết sẽ phối hợp với cơ quan chức năng và không sử dụng hình ảnh có thể gây tác động tiêu cực đến khán giả trẻ.

Chuyên gia truyền thông nhận định các đơn vị sản xuất cần có quy trình quản trị rủi ro, bao gồm kiểm tra điều khoản đạo đức nghề nghiệp và cơ chế xử lý khi nghệ sĩ vướng vi phạm pháp luật.""",
    },
    {
        "slug": "chien-dich-phong-chong-ma-tuy",
        "title": "Nghệ sĩ tham gia chiến dịch truyền thông phòng chống ma túy",
        "url": "https://example.com/news/chien-dich-phong-chong",
        "content": """Bài báo mẫu ghi nhận nhiều nghệ sĩ tham gia chiến dịch truyền thông phòng chống ma túy tại trường học và không gian công cộng. Nội dung chiến dịch tập trung vào tác hại sức khỏe, hậu quả pháp lý và các kênh hỗ trợ tư vấn cai nghiện.

Ban tổ chức nhấn mạnh thông điệp truyền thông phải dựa trên kiến thức khoa học, không kỳ thị người đang điều trị nghiện và không mô tả chi tiết cách sử dụng chất cấm.""",
    },
    {
        "slug": "quan-ly-nghe-si-sau-khung-hoang",
        "title": "Công ty quản lý nghệ sĩ siết điều khoản đạo đức sau khủng hoảng",
        "url": "https://example.com/news/quan-ly-nghe-si",
        "content": """Bài báo mẫu cho biết một số công ty quản lý nghệ sĩ bổ sung điều khoản đạo đức nghề nghiệp, yêu cầu nghệ sĩ tuân thủ pháp luật về phòng chống ma túy. Điều khoản mới có thể cho phép tạm dừng hợp đồng, hủy lịch diễn hoặc yêu cầu nghệ sĩ tham gia tư vấn phục hồi khi có vi phạm.

Luật sư được phỏng vấn cho rằng hợp đồng dân sự không thay thế trách nhiệm pháp lý nếu có hành vi tàng trữ, mua bán hoặc sử dụng trái phép chất ma túy.""",
    },
    {
        "slug": "bao-chi-va-quyen-rieng-tu",
        "title": "Báo chí cân bằng giữa quyền riêng tư và lợi ích công chúng trong tin ma túy",
        "url": "https://example.com/news/quyen-rieng-tu",
        "content": """Bài báo mẫu phân tích cách báo chí đưa tin về nghệ sĩ liên quan đến ma túy cần cân bằng giữa quyền riêng tư, lợi ích công chúng và nguyên tắc không suy đoán khi chưa có kết luận chính thức. Thông tin nên dựa trên nguồn xác thực và tránh mô tả giật gân.

Chuyên gia đạo đức báo chí khuyến nghị dùng ngôn ngữ trung tính, bảo vệ người chưa thành niên và nhấn mạnh thông điệp phòng ngừa thay vì khai thác đời tư của cá nhân liên quan.""",
    },
]


QA_MARKDOWN = """# Q&A thủ công về pháp luật ma túy

Q: Người dùng hỏi về hình thức cai nghiện ma túy theo Luật Phòng, chống ma túy 2021.
A: Context mẫu nêu bốn hình thức: cai nghiện tự nguyện tại gia đình, tại cộng đồng, tại cơ sở cai nghiện ma túy và cai nghiện bắt buộc tại cơ sở cai nghiện ma túy.

Q: Hệ thống có được hướng dẫn cách sản xuất hoặc mua bán ma túy không?
A: Không. Hệ thống phải từ chối mọi yêu cầu hướng dẫn sản xuất, mua bán, vận chuyển, che giấu hoặc né tránh xử lý pháp luật liên quan đến ma túy.

Q: Khi không có đủ context thì câu trả lời nên làm gì?
A: Câu trả lời phải nói rõ không đủ dữ liệu từ nguồn hiện có và không suy đoán ngoài context.
"""


def ensure_sample_data() -> None:
    """Create local sample data only when the expected files are missing."""
    _write_legal_samples()
    _write_news_samples()
    _write_standardized_samples()


def _write_legal_samples() -> None:
    legal_dir = LANDING_DIR / "legal"
    legal_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in LEGAL_DOCS.items():
        path = legal_dir / filename
        if not path.exists() or path.stat().st_size < 1024:
            path.write_text(content, encoding="utf-8")


def _write_news_samples() -> None:
    news_dir = LANDING_DIR / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    crawled_at = datetime.now(timezone.utc).isoformat()
    for index, article in enumerate(NEWS_ARTICLES, 1):
        path = news_dir / f"article_{index:02d}_{article['slug']}.json"
        if path.exists() and path.stat().st_size > 500:
            continue
        payload = {
            "url": article["url"],
            "title": article["title"],
            "date_crawled": crawled_at,
            "content_markdown": article["content"],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_standardized_samples() -> None:
    legal_std = STANDARDIZED_DIR / "legal"
    news_std = STANDARDIZED_DIR / "news"
    qa_std = STANDARDIZED_DIR / "qa"
    legal_std.mkdir(parents=True, exist_ok=True)
    news_std.mkdir(parents=True, exist_ok=True)
    qa_std.mkdir(parents=True, exist_ok=True)

    for filename, content in LEGAL_DOCS.items():
        path = legal_std / f"{Path(filename).stem}.md"
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    for index, article in enumerate(NEWS_ARTICLES, 1):
        path = news_std / f"article_{index:02d}_{article['slug']}.md"
        if path.exists():
            continue
        markdown = (
            f"# {article['title']}\n\n"
            f"**Source:** {article['url']}\n"
            f"**Crawled:** {datetime.now(timezone.utc).isoformat()}\n\n"
            f"---\n\n{article['content']}\n"
        )
        path.write_text(markdown, encoding="utf-8")

    qa_path = qa_std / "qa_manual.md"
    if not qa_path.exists():
        qa_path.write_text(QA_MARKDOWN, encoding="utf-8")


if __name__ == "__main__":
    ensure_sample_data()
    print("Sample data is ready.")
