from pathlib import Path
import os


################################ MODEL HYPERPARAMETERS ###########################
block_size = 256 # context window for model; can scale to potentially quadratic intermediate VRAM memory space without FlashAttention
# NOTE: If block size is changed, the gutenberg qa dataset will need be rebuilt to accomadate the change in block_size
emb_dims = 256 # Channel dimensions available for total number of attention heads (both encoder and decoder)
memory_slots = 8 # Number of memory slots 
memory_slot_dims = 256 # Number of channels available to each memory slot
head_num = 4 # Number of attention heads per Transformer block
num_blocks = 4 # Number of Decoder Transformer blocks sequentually stacked
num_encoder_blocks = 4 # Number of Encoder Transformer blocks sequentially stacked
memory_pooling_heads = 4 # Number of heads for memory pooling
memory_routing_passes = 3 # Number of routing passes slots get in compressor before forwarding to GRU
memory_write_chunk_size = block_size # The size of each chunk that a single window is broken down to for memory writes; setting to block_size will effectively disable chunking
dropout = 0.2 # Dropout probability; the probability of a neuron to be zeroed out during forward and backpass randomly to prevent overfitting
GUTENBERG_ALLOW_MEMORY_SHRINK = True # Temporary testing parameter allows for weights to be loaded from a model with different memory arrangement
GUTENBERG_INIT_HEAD_NUM = 4  # Actual SOURCE head count
##################################################################################

################################ SHARED TRAINING PARAMETERS #############################
GB = 1 # How many GiB of RAM pool to allocate
POOL_SIZE = GB * 1024**3 # Pool size in bytes  (1 GiB is default)
EVAL_INTERVAL_TOKENS = 20_000_000
CHECKPOINT_INTERVAL_TOKENS = 20_000_000
VALIDATION_BATCHES = 16 # Number of validation batches, each containing batch_size examples.
VALIDATION_SET_SIZE = 0 # Mainly for debug purposes (faster startup the lower the number) should not be used for actual training; set to 0 to bypass
TRAIN_SET_SIZE = 0 # Mainly for debug purposes (faster startup the lower the number) should not be used for actual training; set to 0 to bypass
WARMUP_TOKENS = 110_000_000 # Defines the number of tokens needed for warmup before peak learning rate is reached and cosine decay is applied
MAX_TOKENS = 2_200_000_000 # Defines the max number of tokens to be used for training for cosine decay calculation
MIN_LEARN_RATE = 1e-5 # Defines the minimum learn rate cosine decay will allow it to decay too as training progresses
STOP_AFTER_POOL_CT = 0 # Training session will stop after this many ram pools are consumed (set to 0 to use the whole training set in one session)
TOKEN_STOP_COUNT = 2_200_000_000 # Optional config to stop training after token count is reached; if set to None then training will run over entire training set; training will always exit after most recent checkpoint is created so this number won't be exact stopping point
MIXED_VERIFY_DATA_HASH = True  # False will skip full-content hashing for debug runs/faster startup
# NOTE: Disabling this can lead to dataset mismatches

batch_size = 16 # Number of examples per optimizer batch; Loads VRAM with an approximate linear relationship; larger values means faster training process but more VRAM load
num_epochs = 1 # Number of training cycles to run
learn_rate = 3e-5 # For SGD or AdamW approach; defines the peak learn rate before cosine decay rate drops it lower to 1e-5
##################################################################################

############################### MEMORY TRAINING PARAMETERS ######################################
MEMORY_WINDOWS_PER_SEQUENCE = 5 # Number of windows between fact and question injection formed from training docs; will increase VRAM usage due to longer computation graph for backprop; Minimum required for recall is 2
MEMORY_TEST = False # If true, will divert to the memory-recall training pipeline
MEMORY_MIN_ASSOCIATIONS = 1 # The minimum number of FQAs injected into window sequences
MEMORY_MAX_ASSOCIATIONS = 1 # the maximum number of FQAs injected into window sequences; keep both values the same to prevent randomized injection amounts
RECALL_LOSS_WEIGHT = 1.0 # Extra CE weight applied only to answer-token targets in W4; will pressure the model to favor getting artifical questions correct 
USE_FACT_OFFSET_CURRICULUM = False # Determines if single fact offset will be used; set to True if you want a single fact placed in offsets on set intervals; will ignore MEMORY_MIN/MAX_ASSOCIATION Values
GUTENBERG_DECODER_ONLY = False # Freeze memory system and train decoder only
GUTENBERG_MEMORY_ONLY = False # Freeze the decoder and train memory only
GUTENBERG_EVAL_NOREAD = True # Perform an evaluation with no cross-attention with memory slots; for a control variable to test recall against
GUTENBERG_EVAL_TRAIN_SUBSET = False # Perform an evaluation of Read and NoRead on training sets as well
TRAIN_DIAGNOSTIC_BATCHES = 16 # This is used to breakdown the Normal vs NoRead accuracy of recall on training batches
######################## 
# Seperate LR scalar values for joint training of decoder and recall-memory system
MEMORY_LR_SCALE = 1.0 
DECODER_LR_SCALE = 1.0
########################
GUTENBERG_EVIDENCE_WINDOWS = None # Which window the evidence/data will be used to answer the question asked in W4 example: (2,3) will use evidence from W3 or W2; None will use any window W0->W3
GUTENBERG_SUPPORT_WINDOW = None # Determines where supporting evidence will be injected (W0, W1 etc.) None means no support evidence will be placed; "source" means randomly selected window will have assoicated evidence assist placed in it
GUTENBERG_SUPPORT_LABEL = False  # True: include label; False: evidence only without the support label "Supporting Evidence:..."

# Controls the offset of the fact tokens placement in W0 window
## Format: (tokens seen, minimum offset, maximum offset), inclusive
FACT_OFFSET_SCHEDULE = (
    (0,           0, 0),
    (100_000_000,  0, 8),
    (150_000_000,  0, 16),
    (200_000_000, 0, 32),
    (250_000_000, 0, 64),
    (300_000_000, 0, None),
    
)
ALL_FACTS_SINGLE_QA = True
# If set to True not every fact will have an associated QA (tests associative fact recall)
## NOTE: Will be ignored if USE_FACT_OFFSET_CIRRICULUM = False
##################################################################################

################################### STANDARD (PROSE) TRAINING PARAMETERS #############################
STANDARD_DECODER_ONLY = False # Disables the memory system so decoder is the only one affected by training
######################################################################################################

################################### MIXED TRAINING PARAMETERS #########################################
MIXED_RECALL_RATIO = 0.3       # fraction of optimizer updates, 0..1
MIXED_PROSE_BATCH_SIZE = 16   # recall and validation use existing batch_size
#######################################################################################################

############################### PATH DEFINITIONS #################################
PROJECT_ROOT = Path(__file__).resolve().parent
HF_TOKEN = os.environ.get("HF_TOKEN")

# STANDARD TRAINING
DATA_DIR = PROJECT_ROOT / "train_data" / "gutenberg"
STANDARD_INIT_CHECKPOINT = None  # Fresh training; set a path for weights-only initialization.
STANDARD_CHECKPOINT_DIR = "./checkpoints_standard"

# MEMORY TRAINING (SYNTHETIC FAQS)
MEMORY_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_weight_loading"
MEMORY_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_gutenberg_qa" / "latest.pt"

# MEMORY TRAINING (GUTENBERG CONTEXT CLOZE FAQS)
GUTENBERG_QA_DIR = './gutenberg_qa'  # or './sample_qa' / './gutenberg_qa_natural'
GUTENBERG_CHECKPOINT_DIR = './checkpoints_gutenberg_qa'
GUTENBERG_INIT_CHECKPOINT = None  # Fresh training; set a path for weights-only initialization.

# MIXED TRAINING
MIXED_CHECKPOINT_DIR = './checkpoints_mixed'
MIXED_INIT_CHECKPOINT = None  # Fresh training; set a path for weights-only initialization.
MIXED_ACTIVATION_CHECKPOINTING = True # For mixed training, this will reduce intermediate VRAM usage by recalculating activations instead of holding them in memory
###################################################################################

############################### GENERATION PARAMETERS #############################
MAX_GENERATED_TOKENS = 1000
TEMPERATURE = 0.9 # Temperature (0 to 1.0) effects the distribution, a value closer to 0 sharpens the distribution, while a value close to 1 or over flattens the distribution so rarer tokens get equal selection potentially
TOP_K = 100 # The number of tokens to be sampled for selection on the forward pass
###################################################################################

# Optional portable experiment presets. Unknown names fail fast to catch typos.
# Example: MINIGPT_CONFIG=configs/baseline_8x256.json python training_RNN_mixed.py
if os.environ.get("MINIGPT_CONFIG"):
    import json as _json
    _path = Path(os.environ["MINIGPT_CONFIG"])
    _settings = _json.loads(_path.read_text(encoding="utf-8"))
    if not isinstance(_settings, dict):
        raise ValueError("MINIGPT_CONFIG must contain a JSON object.")
    _unknown = set(_settings) - set(globals())
    if _unknown or any(k.startswith("_") or k in {"os", "Path", "PROJECT_ROOT", "HF_TOKEN"} for k in _settings):
        raise ValueError(f"Unknown or protected config keys: {sorted(_unknown or set(_settings))}")
    globals().update(_settings)
