_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/part/a6_cPnP_part"

SOLVER = dict(
    IMS_PER_BATCH=8,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=500,
)

DATASETS = dict(
    TRAIN=("part_train",),
    TEST=("part_test",),
)

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
            NUM_REGIONS=64,
        ),
        PNP_NET=dict(
            R_ONLY=False,
            REGION_ATTENTION=True,
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
    ),
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="gt")

