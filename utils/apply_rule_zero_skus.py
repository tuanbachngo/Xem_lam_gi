import argparse
import os

import pandas as pd

# ==========================================
# CẤU HÌNH THAM SỐ MẶC ĐỊNH (DEFAULT CONFIG)
# ==========================================
# Người dùng có thể chỉnh sửa các tham số này trực tiếp trong code hoặc truyền qua dòng lệnh (CLI).
DEFAULT_CONFIG = {
    "train_path": "dataset/train.csv",
    "sub_path": "output/submission_outputs/submission.csv",
    "output_path": "post_process_submission/submission_experiment_new.csv",
    # 1. Khoảng thời gian gần (Recent period): tính từ (max_date - recent_days) đến max_date
    # -> GIẢM số ngày này sẽ làm điều kiện lỏng hơn (nhiều SKU không phát sinh giao dịch hơn trong khoảng gần)
    "recent_days": 30,
    # 2. Khoảng thời gian xa (Distant period): tính từ (max_date - distant_days) đến trước (max_date - recent_days)
    "distant_days": 90,
    # 3. Ngưỡng lọc cho Khoảng Gần (Recent Period)
    # -> TĂNG số giao dịch bán cho phép sẽ làm điều kiện lỏng hơn (chấp nhận SKU có vài giao dịch vẫn bị coi là deactive)
    "max_recent_sales_tx": 20,  # Cho phép tối đa số lần bán hàng (Quantity > 0) trong khoảng gần
    "max_recent_returns": 10,  # Cho phép tối đa số lần returns (Quantity < 0) trong khoảng gần
    "max_recent_return_qty": 30,  # Quantity tuyệt đối tối đa của mỗi lần return trong khoảng gần

    # 4. Ngưỡng lọc cho Khoảng Xa (Distant Period)
    # -> TĂNG các thông số này sẽ làm điều kiện lỏng hơn (lọc cả những SKU bán trung bình/khá thay vì chỉ cực kỳ ít)
    # --- Đối với giao dịch Sales (Quantity > 0) ---
    "max_distant_sales_tx": 30,  # Số giao dịch bán tối đa ở khoảng xa (tăng từ 5 lên 15)
    "max_distant_sales_qty_per_tx": 25,  # Quantity tối đa của mỗi giao dịch bán ở khoảng xa (tăng từ 10 lên 30)
    "max_distant_sales_total_qty": 200,  # Tổng quantity bán tối đa ở khoảng xa (tăng từ 30 lên 100)
    # --- Đối với giao dịch Returns (Quantity < 0) ---
    "max_distant_returns_tx": 20,  # Số giao dịch trả lại tối đa ở khoảng xa
    "max_distant_returns_qty_per_tx": 20,  # Quantity tuyệt đối tối đa của mỗi giao dịch trả lại ở khoảng xa
    "max_distant_returns_total_qty": 200,  # Tổng quantity trả lại tuyệt đối tối đa ở khoảng xa
}


def analyze_and_apply_rule(config):
    print("=" * 60)
    print(" BẮT ĐẦU XỬ LÝ DỮ LIỆU & ÁP DỤNG QUY TẮC SET 0 CHO SKUs ")
    print("=" * 60)

    # Kiểm tra sự tồn tại của file train và submission
    if not os.path.exists(config["train_path"]):
        raise FileNotFoundError(
            f"Không tìm thấy file train.csv tại: {config['train_path']}"
        )
    if not os.path.exists(config["sub_path"]):
        raise FileNotFoundError(
            f"Không tìm thấy file submission tại: {config['sub_path']}"
        )

    # --- BƯỚC 1: Load train.csv ---
    print(f"[1/5] Đang đọc file train: {config['train_path']} ...")
    # Chỉ đọc các cột cần thiết để tối ưu hóa tốc độ đọc và tránh cảnh báo kiểu dữ liệu
    train = pd.read_csv(
        config["train_path"], usecols=["Date", "ItemCode", "Quantity"], low_memory=False
    )

    # Chuyển cột Date sang datetime
    train["Date"] = pd.to_datetime(train["Date"])

    # Tạo các cột phụ trợ phục vụ phân tích Sales và Returns
    train["AbsQuantity"] = train["Quantity"].abs()
    train["IsSale"] = train["Quantity"] > 0
    train["IsReturn"] = train["Quantity"] < 0

    print(f"   -> Tổng số dòng trong train.csv: {len(train):,}")
    print(
        f"   -> Số lượng SKUs duy nhất trong train.csv: {train['ItemCode'].nunique():,}"
    )

    # Tìm ngày gần nhất và xa nhất trong tập train
    max_date = train["Date"].max()
    min_date = train["Date"].min()
    print(
        f"   -> Khoảng thời gian dữ liệu train: từ {min_date.strftime('%Y-%m-%d')} đến {max_date.strftime('%Y-%m-%d')}"
    )

    # --- BƯỚC 2: Phân chia khoảng thời gian ---
    recent_start_date = max_date - pd.Timedelta(days=config["recent_days"])
    distant_start_date = max_date - pd.Timedelta(days=config["distant_days"])

    print("[2/5] Phân chia thời gian:")
    print(
        f"   -> Khoảng thời gian gần (Recent Period): từ {recent_start_date.strftime('%Y-%m-%d')} đến {max_date.strftime('%Y-%m-%d')} ({config['recent_days']} ngày gần nhất)"
    )
    print(
        f"   -> Khoảng thời gian xa (Distant Period): từ {distant_start_date.strftime('%Y-%m-%d')} đến trước {recent_start_date.strftime('%Y-%m-%d')} (từ ngày -{config['distant_days']} đến ngày -{config['recent_days']})"
    )

    # Chia dữ liệu train thành các phần tương ứng
    train_recent = train[train["Date"] >= recent_start_date]
    train_distant = train[
        (train["Date"] >= distant_start_date) & (train["Date"] < recent_start_date)
    ]

    # --- BƯỚC 3: Phân tích hành vi mua sắm (Sales & Returns) của từng SKU ---
    print("[3/5] Đang phân tích hành vi của các SKUs...")

    # A. PHÂN TÍCH KHOẢNG THỜI GIAN GẦN (RECENT PERIOD)
    # Thống kê Sales trong khoảng gần
    recent_sales = (
        train_recent[train_recent["IsSale"]]
        .groupby("ItemCode")
        .size()
        .rename("recent_sales_count")
    )
    # Thống kê Returns trong khoảng gần
    recent_returns_stats = (
        train_recent[train_recent["IsReturn"]]
        .groupby("ItemCode")
        .agg(
            recent_returns_count=("Quantity", "count"),
            recent_returns_max_abs_qty=("AbsQuantity", "max"),
        )
    )

    # B. PHÂN TÍCH KHOẢNG THỜI GIAN XA (DISTANT PERIOD)
    # Thống kê Sales trong khoảng xa
    distant_sales_stats = (
        train_distant[train_distant["IsSale"]]
        .groupby("ItemCode")
        .agg(
            distant_sales_count=("Quantity", "count"),
            distant_sales_max_qty=("Quantity", "max"),
            distant_sales_total_qty=("Quantity", "sum"),
        )
    )
    # Thống kê Returns trong khoảng xa
    distant_returns_stats = (
        train_distant[train_distant["IsReturn"]]
        .groupby("ItemCode")
        .agg(
            distant_returns_count=("Quantity", "count"),
            distant_returns_max_abs_qty=("AbsQuantity", "max"),
            distant_returns_total_abs_qty=("AbsQuantity", "sum"),
        )
    )

    # C. GHÉP TẤT CẢ THỐNG KÊ LẠI VÀO MỘT BẢNG CHUNG
    all_train_skus = set(train["ItemCode"].unique())
    stats = pd.DataFrame(index=list(all_train_skus))
    stats.index.name = "ItemCode"

    stats = stats.join(recent_sales, how="left")
    stats = stats.join(recent_returns_stats, how="left")
    stats = stats.join(distant_sales_stats, how="left")
    stats = stats.join(distant_returns_stats, how="left")

    # Điền các giá trị trống bằng 0 (nghĩa là không phát sinh giao dịch nào thuộc nhóm đó)
    stats = stats.fillna(0)

    # D. ÁP DỤNG CÁC ĐIỀU KIỆN LỌC
    # 1. Điều kiện khoảng gần:
    # - Số giao dịch mua ít (<= max_recent_sales_tx)
    # - Nếu có giao dịch trả hàng (returns) thì số lần phải nhỏ (<= max_recent_returns) và lượng trả phải nhỏ (<= max_recent_return_qty)
    cond_recent_sales = stats["recent_sales_count"] <= config["max_recent_sales_tx"]
    cond_recent_returns_cnt = (
        stats["recent_returns_count"] <= config["max_recent_returns"]
    )
    cond_recent_returns_qty = (
        stats["recent_returns_max_abs_qty"] <= config["max_recent_return_qty"]
    )
    cond_recent = cond_recent_sales & cond_recent_returns_cnt & cond_recent_returns_qty

    # 2. Điều kiện khoảng xa:
    # - Giao dịch Sales ít và lượng nhỏ
    cond_distant_sales_cnt = (
        stats["distant_sales_count"] <= config["max_distant_sales_tx"]
    )
    cond_distant_sales_qty = (
        stats["distant_sales_max_qty"] <= config["max_distant_sales_qty_per_tx"]
    )
    cond_distant_sales_tot = (
        stats["distant_sales_total_qty"] <= config["max_distant_sales_total_qty"]
    )

    # - Giao dịch Returns ít và lượng trả nhỏ (tương tự hoặc chặt chẽ hơn Sales)
    cond_distant_returns_cnt = (
        stats["distant_returns_count"] <= config["max_distant_returns_tx"]
    )
    cond_distant_returns_qty = (
        stats["distant_returns_max_abs_qty"] <= config["max_distant_returns_qty_per_tx"]
    )
    cond_distant_returns_tot = (
        stats["distant_returns_total_abs_qty"]
        <= config["max_distant_returns_total_qty"]
    )

    cond_distant = (
        cond_distant_sales_cnt
        & cond_distant_sales_qty
        & cond_distant_sales_tot
        & cond_distant_returns_cnt
        & cond_distant_returns_qty
        & cond_distant_returns_tot
    )

    # Lọc ra các SKUs thỏa mãn cả 2 khoảng thời gian
    zero_skus_df = stats[cond_recent & cond_distant]
    target_skus_to_zero = set(zero_skus_df.index)
    print(
        f"   -> Số SKUs thỏa mãn cả 2 điều kiện từ tập train: {len(target_skus_to_zero):,}"
    )

    # --- BƯỚC 4: Load và xử lý file submission ---
    print(f"[4/5] Đang đọc file submission gốc: {config['sub_path']} ...")
    sub_df = pd.read_csv(config["sub_path"])

    # Xác định các cột dự đoán cần set về 0 (tất cả các cột trừ cột 'id')
    pred_cols = [col for col in sub_df.columns if col != "id"]
    print(
        f"   -> Tìm thấy {len(pred_cols)} cột dự đoán: {pred_cols[0]} ... {pred_cols[-1]}"
    )

    # Kiểm tra xem file submission đầu vào có phải là file trống (tất cả bằng 0) không
    total_pred_sum = sub_df[pred_cols].sum().sum()
    if total_pred_sum == 0:
        print(
            "\n   [CẢNH BÁO] File submission đầu vào của bạn hiện có tổng các giá trị dự đoán bằng 0."
        )
        print(
            "              Nếu đây là file mẫu 'sample_submission.csv', hãy chắc chắn rằng bạn đã truyền đường dẫn"
        )
        print(
            "              tới file kết quả dự đoán thực tế của model bằng đối số --sub_path."
        )
        print("              Ví dụ: --sub_path output/my_predictions.csv\n")

    # Hàm trích xuất SKU từ cột id (loại bỏ suffix _validation hoặc _evaluation ở cuối)
    def extract_sku(row_id):
        row_str = str(row_id)
        for suffix in ["_validation", "_evaluation", "_VALIDATION", "_EVALUATION"]:
            if row_str.endswith(suffix):
                return row_str[: -len(suffix)]
        return row_str

    sub_df["SKU_extracted"] = sub_df["id"].apply(extract_sku)

    # Kiểm tra xem có SKUs nào trong submission không có bất kỳ giao dịch nào trong tập train
    sub_skus = set(sub_df["SKU_extracted"].unique())
    dead_skus_not_in_train = sub_skus - all_train_skus
    print(
        f"   -> Số SKUs trong submission không xuất hiện trong file train: {len(dead_skus_not_in_train):,}"
    )

    # Gộp chung cả hai nhóm SKUs cần set về 0:
    # Nhóm 1: Thỏa mãn 2 điều kiện đề bài
    # Nhóm 2: Không xuất hiện trong tập train
    final_zero_skus = target_skus_to_zero.union(dead_skus_not_in_train)
    print(
        f"   => TỔNG CỘNG số SKUs sẽ bị set giá trị về 0 trong submission: {len(final_zero_skus):,}"
    )

    # Áp dụng thay đổi giá trị về 0
    mask_to_zero = sub_df["SKU_extracted"].isin(final_zero_skus)
    sub_df.loc[mask_to_zero, pred_cols] = 0

    # Xóa cột phụ trợ trước khi lưu file
    sub_df = sub_df.drop(columns=["SKU_extracted"])

    # --- BƯỚC 5: Lưu file submission mới ---
    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = os.path.dirname(config["output_path"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"[5/5] Đang lưu file submission mới đã set 0 tại: {config['output_path']} ..."
    )
    sub_df.to_csv(config["output_path"], index=False)
    print("   -> Lưu file thành công!")

    # --- IN THỐNG KÊ CHI TIẾT ---
    print("\n" + "=" * 60)
    print(" THỐNG KÊ KẾT QUẢ ")
    print("=" * 60)
    print(f"- Tổng số dòng trong file submission: {len(sub_df):,}")
    print(f"- Số dòng bị set về 0: {mask_to_zero.sum():,}")
    print(f"- Tỷ lệ dòng bị set về 0: {mask_to_zero.mean() * 100:.2f}%")

    if len(final_zero_skus) > 0:
        sample_list = list(final_zero_skus)[:10]
        print(f"- Ví dụ 10 SKUs bị set về 0: {', '.join(sample_list)}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script xử lý dữ liệu train.csv và set 0 các SKUs thỏa mãn điều kiện giao dịch kém ở thời gian xa và không giao dịch ở thời gian gần."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=DEFAULT_CONFIG["train_path"],
        help="Đường dẫn tới file train.csv",
    )
    parser.add_argument(
        "--sub_path",
        type=str,
        default=DEFAULT_CONFIG["sub_path"],
        help="Đường dẫn tới file submission.csv cần copy và xử lý",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_CONFIG["output_path"],
        help="Đường dẫn để lưu file submission mới",
    )

    parser.add_argument(
        "--recent_days",
        type=int,
        default=DEFAULT_CONFIG["recent_days"],
        help="Số ngày trong khoảng thời gian gần (ví dụ: 60)",
    )
    parser.add_argument(
        "--distant_days",
        type=int,
        default=DEFAULT_CONFIG["distant_days"],
        help="Số ngày tối đa cho khoảng thời gian xa (ví dụ: 150)",
    )

    parser.add_argument(
        "--max_recent_sales_tx",
        type=int,
        default=DEFAULT_CONFIG["max_recent_sales_tx"],
        help="Số lần bán tối đa cho phép ở khoảng gần",
    )
    parser.add_argument(
        "--max_recent_returns",
        type=int,
        default=DEFAULT_CONFIG["max_recent_returns"],
        help="Số lần returns tối đa cho phép ở khoảng gần",
    )
    parser.add_argument(
        "--max_recent_return_qty",
        type=int,
        default=DEFAULT_CONFIG["max_recent_return_qty"],
        help="Quantity absolute tối đa mỗi lần return ở khoảng gần",
    )

    parser.add_argument(
        "--max_distant_sales_tx",
        type=int,
        default=DEFAULT_CONFIG["max_distant_sales_tx"],
        help="Số giao dịch bán tối đa ở khoảng xa",
    )
    parser.add_argument(
        "--max_distant_sales_qty",
        type=int,
        default=DEFAULT_CONFIG["max_distant_sales_qty_per_tx"],
        help="Quantity tối đa mỗi lần bán ở khoảng xa",
    )
    parser.add_argument(
        "--max_distant_sales_total_qty",
        type=int,
        default=DEFAULT_CONFIG["max_distant_sales_total_qty"],
        help="Tổng quantity bán tối đa ở khoảng xa",
    )

    parser.add_argument(
        "--max_distant_returns_tx",
        type=int,
        default=DEFAULT_CONFIG["max_distant_returns_tx"],
        help="Số giao dịch trả lại tối đa ở khoảng xa",
    )
    parser.add_argument(
        "--max_distant_returns_qty",
        type=int,
        default=DEFAULT_CONFIG["max_distant_returns_qty_per_tx"],
        help="Quantity absolute tối đa mỗi lần trả ở khoảng xa",
    )
    parser.add_argument(
        "--max_distant_returns_total_qty",
        type=int,
        default=DEFAULT_CONFIG["max_distant_returns_total_qty"],
        help="Tổng quantity trả lại absolute tối đa ở khoảng xa",
    )

    args = parser.parse_args()

    config = {
        "train_path": args.train_path,
        "sub_path": args.sub_path,
        "output_path": args.output_path,
        "recent_days": args.recent_days,
        "distant_days": args.distant_days,
        "max_recent_sales_tx": args.max_recent_sales_tx,
        "max_recent_returns": args.max_recent_returns,
        "max_recent_return_qty": args.max_recent_return_qty,
        "max_distant_sales_tx": args.max_distant_sales_tx,
        "max_distant_sales_qty_per_tx": args.max_distant_sales_qty,
        "max_distant_sales_total_qty": args.max_distant_sales_total_qty,
        "max_distant_returns_tx": args.max_distant_returns_tx,
        "max_distant_returns_qty_per_tx": args.max_distant_returns_qty,
        "max_distant_returns_total_qty": args.max_distant_returns_total_qty,
        "only_positive_sales": False,  # Phải đặt bằng False để phân tích cả giao dịch âm (Returns)
    }

    analyze_and_apply_rule(config)
