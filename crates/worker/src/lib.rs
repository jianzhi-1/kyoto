use block_manager::{AllocationError, BlockManager};
use types::{RequestId, TokenId};

#[derive(Debug)]
pub enum WorkerError {
    AllocationError(AllocationError),
    EmptyBatch,
}

#[derive(Debug)]
pub struct PrefillOutput {}

#[derive(Debug)]
pub struct DecodeOutput {}

pub struct PrefillWorker {
    pub block_manager: BlockManager,
    vocab_size: u32,
}

pub struct DecodeWorker {
    pub block_manager: BlockManager,
    vocab_size: u32,
    eos_token_id: TokenId,
}
