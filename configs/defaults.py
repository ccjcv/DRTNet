from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = ""
_C.SAVE_PSEUDO_LABLES = False # For Stage 2 only
_C.SEED = 0

_C.DATA = CN()
_C.DATA.ROOT = ""
_C.DATA.NUM_CLASSES = 0
_C.DATA.MODE = ""
_C.DATA.PSEUDO_LABEL_FOLDER = []
_C.DATA.BATCH_SIZE = 0
_C.DATA.CROP_SIZE = ()
_C.DATA.AUG=False 

_C.MODEL = CN()
_C.MODEL.WEIGHTS = ""
_C.MODEL.ROI_SIZE = []      # For Stage 1&2
_C.MODEL.GRID_SIZE = 0      # For Stage 1&2
_C.MODEL.BG_THRESHOLD = 0.  # For Stage 2 only
_C.MODEL.FREEZE_BN = False  # For Stage 3
_C.MODEL.LAMBDA = 0.        # For Stage 3 only
_C.MODEL.DAMP = 7           # For Stage 3 only
_C.MODEL.DCRF= []           # For Stage 2&3
_C.MODEL.GAP = False        # For Stage 1
_C.MODEL.LOSS = ""          # For Stage 3 only
_C.MODEL.SCALE =0         # For Stage 3 only

_C.SOLVER = CN()
_C.SOLVER.LR = 0.
_C.SOLVER.MOMENTUM = 0.
_C.SOLVER.WEIGHT_DECAY = 0.
_C.SOLVER.MAX_ITER = 0
_C.SOLVER.MILESTONES = []   # For Stage 1 only
_C.SOLVER.GAMMA = 0.        # For Stage 3 only
_C.SOLVER.STEP_SIZE = 0. # For Stage 3 only

_C.WANDB=CN()
_C.WANDB.MODE = True
_C.WANDB.NAME=""            # For Stage 1 
_C.WANDB.PROJECT=""         # For Stage 1 
_C.WANDB.RESTORE_RUN_PATH="" # For Stage 2
_C.WANDB.RESTORE_NAME="" # For Stage 2
_C.WANDB.CHECKPOINT = ""