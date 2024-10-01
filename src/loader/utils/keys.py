from enum import Enum, auto, IntEnum

class DataSetType(IntEnum):
    DSEC = auto()
    MULTIFLOW2D = auto()
    EVIMO2 = auto()

class DataLoading(Enum):
    FLOW = auto()
    FLOW_TIMESTAMPS = auto()
    FLOW_VALID = auto()
    FILE_INDEX = auto()
    EV_REPR = auto()
    BIN_META = auto()
    IMG = auto()
    IMG_TIMESTAMPS = auto()
    DATASET_TYPE = auto()
    EVENTS = auto()
    POS_EVENTS = auto()
    NEG_EVENTS = auto()
    NUM_POS_EVENTS = auto()
    DEPTH_MASK = auto()
    X_SCALE = auto()
    Y_SCALE = auto()
    ID_MASK = auto()
