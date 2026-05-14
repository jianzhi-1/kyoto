use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use types::{BlockId, ContentHash, RequestId, TokenId};

#[derive(Debug, Clone, PartialEq)]
pub enum BlockState {
    Free,
    Allocated,
    Computed,
}

/// Metadata for a physical block in the KV cache.
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub id: BlockId,
    pub state: BlockState,
    pub ref_count: u32,
    pub content_hash: Option<u64>,
}

impl PhysicalBlock {
    fn new(id: BlockId) -> Self {
        return Self {
            id,
            state: BlockState::Free,
            ref_count: 0,
            content_hash: None,
        };
    }
}

#[derive(Debug)]
pub enum AllocationError {
    OutOfMemory,
    RequestNotFound,
}

pub struct BlockManager {
    blocks: Vec<PhysicalBlock>,
    free_list: VecDeque<BlockId>,
    prefix_cache: HashMap<ContentHash, BlockId>,
    block_tables: HashMap<RequestId, Vec<BlockId>>,
    pub block_size: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let blocks = (0..num_blocks).map(PhysicalBlock::new).collect();
        let free_list = (0..num_blocks).collect();
        return Self {
            blocks,
            free_list,
            prefix_cache: HashMap::new(),
            block_tables: HashMap::new(),
            block_size,
        };
    }

    pub fn num_free_blocks(&self) -> usize {
        return self.free_list.len();
    }

    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        return num_blocks_needed(num_tokens, self.block_size) <= self.num_free_blocks();
    }

    fn hash_tokens(tokens: &[TokenId]) -> u64 {
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        return hasher.finish();
    }

    fn allocate_fresh(&mut self) -> Result<BlockId, AllocationError> {
        let id = self
            .free_list
            .pop_front()
            .ok_or(AllocationError::OutOfMemory)?;
        self.blocks[id].state = BlockState::Allocated;
        self.blocks[id].ref_count = 1;
        self.blocks[id].content_hash = None;
        return Ok(id);
    }

    pub fn allocate(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[TokenId],
    ) -> Result<(), AllocationError> {
        let mut block_table: Vec<BlockId> = Vec::new();
        let num_full_blocks = prompt_tokens.len() / self.block_size;

        for i in 0..num_full_blocks {
            let start = i * self.block_size;
            let end = (i + 1) * self.block_size;
            let hash = Self::hash_tokens(&prompt_tokens[start..end]);
            if let Some(&cached_block_id) = self.prefix_cache.get(&hash) {
                self.blocks[cached_block_id].ref_count += 1;
                block_table.push(cached_block_id);
            } else {
                let block_id = self.allocate_fresh()?;
                block_table.push(block_id);
            }
        }

        if prompt_tokens.len() % self.block_size != 0 {
            let block_id = self.allocate_fresh()?;
            block_table.push(block_id);
        }
        self.block_tables.insert(request_id, block_table);
        return Ok(());
    }

    pub fn mark_computed(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[TokenId],
    ) -> Result<(), AllocationError> {
        let block_table = self
            .block_tables
            .get(&request_id)
            .ok_or(AllocationError::RequestNotFound)?
            .clone();
        let num_full_blocks = prompt_tokens.len() / self.block_size;

        for i in 0..num_full_blocks {
            let block_id = block_table[i];
            let start = i * self.block_size;
            let end = (i + 1) * self.block_size;
            let hash = Self::hash_tokens(&prompt_tokens[start..end]);
            if self.blocks[block_id].state != BlockState::Computed {
                self.blocks[block_id].state = BlockState::Computed;
                self.blocks[block_id].content_hash = Some(hash);
                self.prefix_cache.insert(hash, block_id);
            }
        }

        return Ok(());
    }

    pub fn append_token(&mut self, request_id: RequestId) -> Result<(), AllocationError> {
        // Primarily used for decode.
        let block_table = self
            .block_tables
            .get(&request_id)
            .ok_or(AllocationError::RequestNotFound)?;
        let last_block_id = *block_table.last().ok_or(AllocationError::RequestNotFound)?;
        let last_block = &self.blocks[last_block_id];

        let needs_new_block =
            (last_block.ref_count > 1) || (last_block.state == BlockState::Computed);

        if needs_new_block {
            let new_block_id = self.allocate_fresh()?;
            self.block_tables
                .get_mut(&request_id)
                .ok_or(AllocationError::RequestNotFound)?
                .push(new_block_id);
        }

        return Ok(());
    }

    pub fn free(&mut self, request_id: RequestId) -> () {
        let Some(block_table) = self.block_tables.remove(&request_id) else {
            return;
        };

        for block_id in block_table {
            let block = &mut self.blocks[block_id];
            block.ref_count -= 1;
            if block.ref_count == 0 {
                if let Some(hash) = block.content_hash {
                    if self.prefix_cache.get(&hash) == Some(&block_id) {
                        self.prefix_cache.remove(&hash);
                    }
                }
                block.state = BlockState::Free;
                block.content_hash = None;
                self.free_list.push_back(block_id);
            }
        }
    }

    pub fn get(&self, request_id: RequestId) -> Option<&Vec<BlockId>> {
        return self.block_tables.get(&request_id);
    }
}

fn num_blocks_needed(num_tokens: usize, block_size: usize) -> usize {
    return (num_tokens + block_size - 1) / block_size;
}

#[cfg(test)]
mod tests;
