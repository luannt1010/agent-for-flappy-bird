# Flappy Bird — Reinforcement Learning với PyTorch

Project huấn luyện và đánh giá agent chơi Flappy Bird từ hình ảnh bằng **học tăng cường (Reinforcement Learning)**. Môi trường game được xây dựng bằng Pygame, các mô hình học bằng PyTorch.

Bạn có thể chơi bằng bàn phím, xem AI đã huấn luyện chơi, so sánh nhiều thuật toán và kiểm tra khả năng thích nghi khi thay đổi tốc độ, khe hở giữa ống hoặc gió.

## Cài đặt

Yêu cầu **Python 3.10 trở lên**. Chạy các lệnh từ thư mục gốc của project.

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên Linux/macOS:

```bash
source .venv/bin/activate
```

Cài các thư viện:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-benchmark.txt
```

Các thư viện chính gồm `torch`, `numpy`, `pygame`, `opencv-python`, `matplotlib` và `tqdm`. File requirements cũng bổ sung `tzdata` trên Windows để ghi thời gian theo múi giờ Việt Nam.

Nếu dùng GPU NVIDIA, cần bản PyTorch tương thích với CUDA trên máy. Benchmark hỗ trợ `--device cpu`, `--device cuda`, `--device cuda:0` và `--device auto` (tự chọn CUDA khi khả dụng).

**Repo không kèm checkpoint đã huấn luyện.** Hãy huấn luyện trước khi chạy các ví dụ cần `best.pt`.

## Chơi thử

Chơi với môi trường mặc định hoặc thử cách bố trí ống khó hơn:

```bash
python game.py
python game.py --profile surprise
```

| Phím | Chức năng |
| --- | --- |
| Space / ↑ | Bay lên; bắt đầu lượt chơi nếu đang chờ |
| Enter | Bắt đầu hoặc chuyển sang lượt tiếp theo |
| P | Tạm dừng / tiếp tục |
| R | Chơi lại đường ống với seed hiện tại |
| Esc | Thoát |

Cửa sổ hiển thị điểm, điểm cao nhất trong phiên, tổng reward và thời gian sống mô phỏng. Tốc độ mặc định là 30 FPS; dùng `--fps` để đổi tốc độ và `--scale` để chỉnh kích thước cửa sổ. Các lượt hoàn tất được tổng hợp trên terminal; lượt bị khởi động lại hoặc thoát giữa chừng không được tính vào thống kê đó.

### Cho AI chơi

Thay `duong-dan/toi/best.pt` bằng checkpoint **single-agent của bộ benchmark**:

```bash
python game.py --profile surprise --checkpoint "duong-dan/toi/best.pt" --episodes 3
```

Thuật toán và cấu hình agent được đọc từ checkpoint. AI tự bắt đầu, chuyển sang seed tiếp theo sau mỗi lượt và chỉ suy luận khi chơi. `--profile` chọn môi trường trình diễn, có thể khác môi trường đã dùng để huấn luyện.

Chạy không mở cửa sổ, không giới hạn FPS:

```bash
python game.py --checkpoint "duong-dan/toi/best.pt" --headless --episodes 3
```

`game.py` mặc định suy luận trên CPU; thêm `--device cuda` nếu cần. Thống kê phiên chơi phục vụ quan sát agent; quy trình benchmark bên dưới cung cấp đánh giá theo nhiều seed và môi trường.

## Thuật toán và môi trường

### Các phương pháp

| Giá trị `--methods` | Mô tả |
| --- | --- |
| `dqn` | Deep Q-Network làm baseline |
| `double_dqn` | Double DQN |
| `dueling_dqn` | DQN với kiến trúc Dueling |
| `per_dqn` | DQN với Prioritized Experience Replay |
| `rainbow` | Kết hợp Double, Dueling, PER, n-step, NoisyNet và C51 |
| `ppo` | Proximal Policy Optimization |
| `a2c` | Advantage Actor-Critic |
| `sac` | Soft Actor-Critic cho hành động rời rạc |
| `domain_randomization` | Tên tương thích cho DQN với ngẫu nhiên hóa môi trường |
| `curriculum` | Tên tương thích cho DQN với lịch tăng độ khó |
| `multi_agent` | Các DQN độc lập thi đấu sinh tồn trên cùng đường ống |

Dùng `--methods all` để chọn tất cả. Với các learner single-agent, có thể chọn riêng chiến lược huấn luyện bằng `--train-strategy fixed`, `domain_randomization` hoặc `curriculum`.

### Các profile

| Profile | Thay đổi so với `standard` |
| --- | --- |
| `standard` | Khe hở 100 px, tốc độ ống −4 px/bước, không gió |
| `narrow` | Thu hẹp khe hở còn 80 px |
| `fast` | Tăng tốc độ ống lên −6 px/bước |
| `wind` | Thêm gió hướng xuống, hệ số 0.35 |
| `combined` | Kết hợp `narrow`, `fast` và `wind` |
| `surprise` | Giữ vật lý chuẩn; đỉnh khe hở nằm trong khoảng 80–260 px, lệch 80–120 px giữa hai ống liên tiếp |

Domain randomization lấy mẫu vật lý mới ở đầu mỗi episode và yêu cầu `--train-profile standard`. Curriculum tăng dần độ khó tới profile đích; `--curriculum-fraction` xác định phần ngân sách dùng để đạt đích. `surprise` hiện chỉ hỗ trợ chiến lược `fixed`; `multi_agent` chỉ hỗ trợ `standard` / `fixed`.

### Quan sát, hành động và reward

- **Quan sát:** chồng 4 ảnh nhị phân 84 × 84, dạng `uint8`, kích thước `(4, 84, 84)`.
- **Hành động:** `0` để chim rơi theo vật lý; `1` để bay lên.
- **Reward:** `+0.1` ở bước còn sống thông thường, `+1` ở bước vượt ống và `−1` khi va chạm.
- **Score:** số ống vượt qua, được báo cáo riêng với tổng reward.
- **Kết thúc:** va chạm là `terminated`; đạt giới hạn số bước là `truncated`.
- **Thời gian sống:** `episode_steps / 30` giây mô phỏng, độc lập với tốc độ chạy thực tế.

## Huấn luyện và đánh giá

### Huấn luyện một nhóm thuật toán

Ví dụ so sánh DQN, Double DQN, PER-DQN và PPO trên môi trường chuẩn, với 5 training seed:

```bash
python benchmark.py --methods dqn double_dqn per_dqn ppo --seeds 0 1 2 3 4 --train-profile standard --train-steps 1000000 --validation-interval 25000 --validation-episodes 10 --test-episodes 100 --eval-profiles standard narrow fast wind surprise combined --early-stopping-episodes 0 --device auto
```

Mỗi cặp thuật toán/seed có ngân sách tối đa 1.000.000 transition. Benchmark mặc định chạy không mở cửa sổ, ghi tiến trình bằng `tqdm` và tự đánh giá checkpoint tốt nhất sau huấn luyện. Thêm `--no-progress` để tắt thanh tiến trình hoặc `--output-dir` để đổi nơi lưu.

Ví dụ thay đổi chiến lược huấn luyện:

```bash
python benchmark.py --methods double_dqn ppo --seeds 0 --train-profile standard --train-strategy domain_randomization --device auto
python benchmark.py --methods double_dqn ppo --seeds 0 --train-profile narrow --train-strategy curriculum --curriculum-fraction 0.8 --device auto
```

### Các bộ thí nghiệm có sẵn

| Cấu hình | Nội dung | Số lượt huấn luyện mặc định |
| --- | --- | --- |
| [core.json](experiments/core.json) | DQN, Double DQN, PER-DQN, PPO × 2 profile huấn luyện: `standard`, `narrow` | 40 |
| [full_matrix.json](experiments/full_matrix.json) | Cùng 4 thuật toán × 5 profile huấn luyện: `standard`, `narrow`, `fast`, `wind`, `surprise` | 100 |
| [robustness.json](experiments/robustness.json) | Double DQN và PPO với domain randomization hoặc curriculum tới `narrow` | 20 |
| [surprise.json](experiments/surprise.json) | Double DQN và PPO huấn luyện trên `standard` hoặc `surprise` | 20 |

Các cấu hình dùng 5 training seed và ngân sách 1.000.000 transition cho mỗi lượt. `full_matrix.json` đánh giá trên cả 6 profile, bao gồm `combined`.

Xem kế hoạch và kiểm tra cấu hình trước khi chạy:

```bash
python benchmark_suite.py --config experiments/full_matrix.json --dry-run
```

Kiểm tra nhanh luồng huấn luyện, đánh giá và xuất báo cáo trên CPU:

```bash
python benchmark_suite.py --config experiments/core.json --smoke --device cpu
```

Chế độ `--smoke` dùng 1 seed, 32 transition mỗi lượt và episode tối đa 40 bước. Kết quả này chỉ kiểm tra luồng thực thi, không phản ánh chất lượng học của agent.

Chạy bộ thí nghiệm đầy đủ:

```bash
python benchmark_suite.py --config experiments/full_matrix.json --device auto
```

Có thể giới hạn thí nghiệm và ghi đè ngân sách khi cần:

```bash
python benchmark_suite.py --config experiments/core.json --experiments fixed_standard --seeds 0 --train-steps 100000 --device cpu
```

### Đánh giá lại checkpoint

Chế độ `evaluate` không huấn luyện. Khai báo checkpoint theo cú pháp `METHOD:SEED=PATH`; method và training seed phải khớp metadata đã lưu:

```bash
python benchmark.py --mode evaluate --methods dqn --seeds 0 --checkpoint "dqn:0=duong-dan/toi/best.pt" --test-episodes 100 --eval-profiles standard narrow fast wind surprise combined --device cpu
```

Nếu chọn nhiều method hoặc seed, lặp lại `--checkpoint` cho từng cặp. Thêm `--render --render-fps 30` để xem các lượt đánh giá qua cửa sổ game.

## Kết quả và checkpoint

Một lần chạy `benchmark.py` tạo cấu trúc:

```text
benchmark_results/<timestamp>/
├── results.json
├── summary.json
├── comparison_<profile>.png
├── comparison_<profile>.pdf
├── learning_curves.png
└── <method>/seed_<seed>/
    ├── best.pt
    ├── history.json
    └── train_summary.json
```

- `results.json`: cấu hình, phiên bản thư viện, thông tin thiết bị, fingerprint SHA-256 của mã nguồn/assets và kết quả từng lượt.
- `summary.json`: tổng hợp theo thuật toán và profile, gồm score, reward, thời gian sống, tỷ lệ đạt 10/50 ống và tỷ lệ chạm giới hạn episode.
- `best.pt`: checkpoint được chọn bằng **reward trung bình trên validation**, không dùng test để chọn mô hình.
- `history.json`, `train_summary.json`: lịch sử validation, chẩn đoán cập nhật và chi phí huấn luyện.
- Các biểu đồ: so sánh kết quả theo profile và đường cong reward trên validation.

Training, validation và test dùng các nhóm seed tách biệt; validation mặc định bắt đầu từ `1_000_000`, test từ `2_000_000`. Khi tổng hợp nhiều training seed, độ lệch chuẩn được tính trên trung bình của từng seed. Khả năng tái lập chính xác còn phụ thuộc phần cứng, phiên bản PyTorch và cấu hình luồng.

Checkpoint phục vụ lựa chọn mô hình và đánh giá; replay buffer và toàn bộ trạng thái cần để tiếp tục chính xác một lượt huấn luyện không được lưu.

Với `benchmark_suite.py`, kết quả mặc định nằm ở `benchmark_results/suites/<timestamp>/`. File `suite.json` ghi trạng thái từng thí nghiệm; thư mục `comparison/` chứa JSON tổng hợp, biểu đồ, `comparison.csv`, `per_seed.csv` và các bảng chuyển môi trường `transfer_score_mean.csv`, `transfer_score_std.csv`.

Có thể thu thập lại báo cáo từ suite đã chạy, kể cả những thí nghiệm đã hoàn tất trong suite bị gián đoạn:

```bash
python benchmark_suite.py --collect "benchmark_results/suites/thu-muc-suite"
```

Lệnh này chỉ tổng hợp báo cáo đã có. Bộ thu thập kiểm tra tính tương thích của protocol, runtime và fingerprint nguồn, đồng thời từ chối các lượt seed trùng lặp.

## Cấu trúc mã nguồn

```text
.
├── game.py                     # Chơi bằng bàn phím hoặc xem AI chơi
├── benchmark.py                # CLI huấn luyện và đánh giá
├── benchmark_suite.py          # Chạy, tổng hợp các bộ thí nghiệm JSON
├── requirements-benchmark.txt  # Thư viện cho game và benchmark
├── experiments/                # Cấu hình thí nghiệm
├── assets/sprites/             # Hình ảnh game
├── src/
│   ├── benchmark_env.py        # Môi trường RL với seed và giới hạn episode
│   ├── benchmarking.py         # Vòng lặp train/evaluate, checkpoint, thống kê
│   ├── experiment_config.py    # Profile và chiến lược môi trường
│   ├── agent_registry.py       # Đăng ký và khởi tạo agent
│   ├── dqn_agent.py            # DQN, Double, Dueling và PER
│   ├── rainbow.py              # Rainbow DQN
│   ├── ppo.py                  # PPO
│   ├── a2c.py                  # A2C
│   ├── sac.py                  # SAC rời rạc
│   ├── replay.py               # Replay buffer
│   ├── policy_networks.py      # Mạng cho policy/value
│   ├── domain_randomization.py
│   ├── curriculum.py
│   ├── multi_agent.py          # Môi trường thi đấu nhiều agent
│   ├── rl_common.py            # Cấu hình, seed và giao diện agent
│   ├── progress.py             # Hiển thị tiến trình
│   ├── provenance.py           # Fingerprint nguồn và assets
│   └── ...                    # Mạng, game và tiền xử lý DQN gốc
├── train.py                    # Script huấn luyện DQN gốc
└── test.py                     # Script chạy mô hình DQN gốc
```

<!-- ## Các script DQN gốc

`train.py`, `test.py`, `src/deep_q_network.py`, `src/flappy_bird.py` và `src/utils.py` giữ phần triển khai gốc với thông tin tác giả **Viet Nguyen** trong mã nguồn. Để dùng luồng này, cài thêm `tensorboardX` ngoài requirements của benchmark:

```bash
python -m pip install tensorboardX
python train.py --num_iters 2000000 --saved_path trained_models --log_path tensorboard
python test.py --saved_path trained_models
```

Chạy `test.py` sau khi huấn luyện đã tạo `trained_models/flappy_bird`. Đây là script xem agent chơi, không phải bộ kiểm thử tự động. `train.py` xóa thư mục `--log_path` đã tồn tại trước khi ghi log mới.

Định dạng mô hình gốc khác với `best.pt` của benchmark; `game.py` yêu cầu định dạng benchmark. Khi cần đối chiếu mô hình gốc, `benchmark.py` có tùy chọn `--baseline-checkpoint`; mô hình được lưu dưới dạng đối tượng Python cần thêm `--trust-legacy-checkpoint` và chỉ nên được nạp từ nguồn tin cậy. -->

## Xem toàn bộ tùy chọn của các CLI:

```bash
python game.py --help
python benchmark.py --help
python benchmark_suite.py --help
```
