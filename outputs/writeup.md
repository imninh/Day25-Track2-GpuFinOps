# Write-up — Lab 25: GPU FinOps Optimization

**Author:** Trần Thế Ninh (2A202602001) · GitHub: imninh
**Vai trò:** FinOps Engineer @ NimbusAI · Period: tháng 6/2026

---

## 1. Baseline vs. Optimized

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| Tổng chi tiêu GPU / tháng | $27,133 | $15,016 | **−45%** |
| Inference `$/1M-token` | $6.488 | $1.126 | **−82.6%** |
| Chi phí mua GPU / tháng (on-demand → spot/reserved) | $25,667 | $16,017 | −37.6% |

Con số quan trọng không phải `$/GPU-hr` mà là `$/1M-token`: hai đội có thể trả
cùng `$/GPU-hr`, nhưng đội tối ưu tốt phục vụ nhiều token hơn nhiều. KPI này bắt
buộc đo cả hiệu quả sử dụng, không chỉ giá thuê.

## 2. Phân tích từng đòn bẩy

| Lever | Savings/tháng | Ghi chú |
|---|---|---|
| **Purchasing (spot/reserved)** | $9,650 | Lớn nhất — vì phần lớn chi tiêu GPU là *infrastructure*, không phải token |
| Inference (cascade/cache/batch) | $1,212 | −82.6% trên `$/1M-token` |
| Right-size "util-lies" | $655 | Hạ cấp GPU bị "lie" |
| Kill idle GPUs | $600 | GPU để trống 8h/ngày |

**Vì sao purchasing là lever lớn nhất?** Chi phí hạ tầng GPU chiếm phần áp đảo hóa
đơn. 3 service inference chạy 24/7 vượt điểm hòa vốn 55% → dùng reserved (3yr);
training job có thể gián đoạn → spot + checkpoint. Kết quả: on-demand $25,667 →
$16,017/tháng.

**Vì sao cascade + cache + batch giảm mạnh `$/1M-token`?** 80% request thực chất
"dễ" và chỉ cần model nhỏ (cascade, rẻ ~15×), input đã cache được chiết khấu 90%,
traffic eval gộp batch giảm thêm 50%. Discount stack nhân lên: `0.5 × 0.1 = 0.05`.

## 3. GPU-Util Lie

**GPU bị "lie": `gpu-h100-4` (98% util, MFU ~0.19) và `gpu-a10g-1` (97%, MFU 0.27).**

`nvidia-smi` "GPU-Util" chỉ đo *thời gian GPU có work được queued* — là đồng hồ
"clock đang bận", không phải thước đo hiệu quả. `gpu-h100-4` 98% util nhưng MFU
chỉ ~0.19: tensor cores "bận" chủ yếu vì kernel launch overhead, memory stall và
pipeline bubble. Bạn trả đủ tiền cho 1 giờ H100 ($2.5/h) nhưng chỉ nhận 1/5 FLOPs.

**Tác động tài chính:** phải trả giá H100 trong khi chỉ cần A100. Right-size 2 GPU
này + kill `gpu-h100-5` idle → ~$1,255/tháng (~4.6% hóa đơn), chưa kể hiệu quả
thực tăng khi đặt đúng hardware.

## 4. Các extension đã làm (3 cái)

### Ext 1 — Interruption-aware tier choice + term matching
`recommend_tier()` mới dùng interruption rate riêng từng GPU (H100 5%, A10G 25%)
và `reserved_hourly_rate()` chọn 1yr vs 3yr theo duration thật của job, thay vì
phẳng 5% và luôn cam kết 3yr.
**Kết quả:** savings purchasing 39.1% → 37.6% — **thấp hơn nhưng thực tế hơn**:
A10G spot bị thu hồi 25% (dev-sandbox $203 → $222), A100 rate 12% (train-embed
$1,393 → $1,439). Insight: với mức chiết khấu spot 40–60% hiện tại, spot luôn
thắng dù bị thu hồi nhiều — interruption rate thay đổi *chi phí*, không đổi *tier*.

### Ext 3 — `cache_is_worth_it()`
Caching không miễn phí (tốn chi phí ghi prefix). Hàm tính break-even số lần đọc:
`small model: 2.78 reads`, `large model: 0.56 reads`. Dataset đo được **536 reads/
prefix** → caching chắc chắn đáng giá; M2 chỉ áp discount cache khi hàm trả `True`.
Insight: cache là "no-brainer" khi system prompt dùng chung nhiều.

### Ext 5 — Carbon-aware scheduling
Cost 5 vùng theo `$/kWh` + `gCO2/kWh`, đề xuất vùng theo tiêu chí:
**Rẻ nhất `$`:** us-east-wa · **Sạch nhất CO2:** europe-north1 · **Cân bằng:** europe-north1.
Chuyển job interruptible từ us-east-1 → europe-north1: **tiết kiệm 1,479,450 gCO2e
(92.1%)** và rẻ hơn về điện. Trade-off: vùng sạch nhất có thể xa users (latency).

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Đổi KPI sang `$/1M-token`** — một câu lệnh dashboards + meeting hằng tuần sẽ
   lộ ngay các GPU 20%-MFU đang "nói dối".
2. **Chốt purchasing trước** — chuyển 24/7 inference sang 3yr reserved và
   interruptible training sang spot+checkpoint (lever lớn nhất, ~$9.6k/tháng).
3. **Bật cascade + cache + batch cho inference** — rẻ 15× cho 80% request, kết
   hợp caching (536 reads/prefix) → `$/1M-token` từ $6.49 về $1.13, đồng thời lập
   lịch vùng sạch nhất cho job gián đoạn để giảm carbon song song với chi phí.

---
_Các con số là snapshot tháng 6/2026, cần re-baseline trước khi áp dụng thực tế._