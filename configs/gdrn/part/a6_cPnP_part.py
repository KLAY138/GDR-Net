_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/part/a6_cPnP_part_0422_visble_picture"

# SOLVER = dict(
#     IMS_PER_BATCH=8,
#     LR_SCHEDULER_NAME="flat_and_anneal",
#     ANNEAL_METHOD="cosine",
#     ANNEAL_POINT=0.72,
#     OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
#     WEIGHT_DECAY=0.0,
#     WARMUP_FACTOR=0.001,
#     WARMUP_ITERS=500,
# )
SOLVER = dict(
    IMS_PER_BATCH=12,
    # --- 核心修改：设定总步数为 60,000 ---
    # MAX_ITER=60000,
    MAX_ITER=134400,
    # --- 核心修改：每 2000 步存一个档，防止断电或进程中断 ---
    CHECKPOINT_PERIOD=4000,

    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    # 保持在总进度的 72% 开始降学习率 (即从 43,200 步开始)
    ANNEAL_POINT=0.72,

    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=5e-5, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    # 稍微拉长一点预热，设为 1000 步更稳
    WARMUP_ITERS=1000,
)

# DATASETS = dict(
#     TRAIN=("part_train",),
#     TEST=("part_test",),
# )

SCENE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# 1. 强制关闭所有可能的数量限制
DATALOADER = dict(
    NUM_WORKERS=4,
    FILTER_EMPTY_ANNOTATIONS=False,
    FILTER_VISIB_THR=0.0,
)

# 2. 检查 DATASETS 是否被写死了
# 确保你的 TRAIN 元组里只有你完整的数据集名称
DATASETS = dict(
    TRAIN=("part_train",),
    TEST=("part_test",),
)

# 3. 检查是否有这个隐藏开关（有些版本有）
# 如果有 LIMIT_DATA，强行设为 None
LIMIT_DATA = None

INPUT = dict(
    BG_TYPE="none",        # 彻底关掉那个找不到路径的背景替换
    CHANGE_BG_PROB=0.0,    # 背景切换概率设为 0
    COLOR_AUG_PROB=0.0  # 关掉彩色增强，让收敛更稳
    )
# MODEL = dict(
#     LOAD_DETS_TEST=False,
#     PIXEL_MEAN=[0.0, 0.0, 0.0],
#     PIXEL_STD=[255.0, 255.0, 255.0],
#     CDPN=dict(
#         ROT_HEAD=dict(
#             FREEZE=False,
#             ROT_CLASS_AWARE=False,
#             MASK_CLASS_AWARE=False,
#             XYZ_LW=1.0,
#             REGION_CLASS_AWARE=False,
#             NUM_REGIONS=64,
#         ),
#         PNP_NET=dict(
#             R_ONLY=False,
#             REGION_ATTENTION=True,
#             WITH_2D_COORD=True,
#             ROT_TYPE="allo_rot6d",
#             TRANS_TYPE="centroid_z",
#             PM_NORM_BY_EXTENT=True,
#             PM_R_ONLY=True,
#             CENTROID_LOSS_TYPE="L1",
#             CENTROID_LW=1.0,
#             Z_LOSS_TYPE="L1",
#             Z_LW=1.0,
#         ),
#         TRANS_HEAD=dict(FREEZE=True),
#     ),
# )
MODEL = dict(
    LOAD_DETS_TEST=False,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            XYZ_LW=1.0,
            REGION_CLASS_AWARE=False,
            # --- 核心修改点 1 ---
            NUM_REGIONS=64,# 改回 8 (或 16, 64)，只要 > 2 就能绕过 AssertionError
            REGION_LW=0.0,     # 关键：权重保持 0.0，这样即使数据是 [0, 255] 也不计入总 Loss
        ),
        PNP_NET=dict(
            R_ONLY=False,
            # --- 核心修改点 2 ---
            REGION_ATTENTION=False, # 设为 False，因为无效的 region map 会干扰注意力机制
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
        TRANS_HEAD=dict(FREEZE=True),
        UNCERTAINTY=dict(
            USE_UNCERTAINTY_BRANCH=True,      # 总开关
            UNCERT_USE_HETERO_XYZ_LOSS=True,  # 开启xyz异方差loss
            UNCERT_USE_AS_PNP_FEATURE=True,   # 拼到Patch-PnP输入
            UNCERT_FEATURE_TYPE="weight_map", # 或 "log_var"
            UNCERT_LOGVAR_CLAMP_MIN=-10.0,
            UNCERT_LOGVAR_CLAMP_MAX=4.0,
            UNCERT_LOSS_WEIGHT=1.0,
            UNCERT_INIT_BIAS=0.0,
        ),
    ),
)

# TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="gt")

TEST = dict(
    # --- 核心修改：每 2000 步在测试集上运行一次评估 ---
    # 这样你可以看到精度随时间增长的曲线，判断是否过拟合
    EVAL_PERIOD=4000,

    # --- 核心修改：开启可视化 ---
    # 建议设为 True，这样在 TensorBoard 的 "vis" 选项卡里能看到模型画出的 3D 框
    VIS=True,

    # 保持使用 Ground Truth Bbox，这对于训练阶段验证模型预测能力最稳妥
    TEST_BBOX_TYPE="gt",
)
