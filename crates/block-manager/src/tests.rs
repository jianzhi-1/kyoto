use super::*;

#[test]
fn test_basic_allocation() {
    let mut bm = BlockManager::new(10, 4);
    assert_eq!(bm.num_free_blocks(), 10);

    // 5 tokens, block size 4 = 1 full block + 1 partial = 2 blocks
    bm.allocate(1, &[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(bm.num_free_blocks(), 8);
}

#[test]
fn test_exact_block_fit() {
    let mut bm = BlockManager::new(10, 4);

    // 8 tokens, block size 4 = exactly 2 full blocks, no partial
    bm.allocate(1, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(bm.num_free_blocks(), 8);
}

#[test]
fn test_free_returns_blocks() {
    let mut bm = BlockManager::new(10, 4);
    bm.allocate(1, &[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(bm.num_free_blocks(), 8);
    bm.free(1);
    assert_eq!(bm.num_free_blocks(), 10);
}

#[test]
fn test_prefix_cache_hit() {
    let mut bm = BlockManager::new(10, 4);
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8]; // exactly 2 full blocks

    // first request - cold cache, uses 2 fresh blocks
    bm.allocate(1, &prompt).unwrap();
    assert_eq!(bm.num_free_blocks(), 8);

    bm.mark_computed(1, &prompt).unwrap();
    assert_eq!(bm.num_free_blocks(), 8); // mark_computed allocates nothing

    // second request - both blocks are cache hits, no fresh blocks consumed
    bm.allocate(2, &prompt).unwrap();
    assert_eq!(bm.num_free_blocks(), 8);
}

#[test]
fn test_out_of_memory() {
    let mut bm = BlockManager::new(2, 4);

    // 5 tokens = 1 full + 1 partial = 2 blocks, uses all memory
    bm.allocate(1, &[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(bm.num_free_blocks(), 0);

    // no blocks left
    let result = bm.allocate(2, &[1, 2, 3]);
    assert!(matches!(result, Err(AllocationError::OutOfMemory)));
}

#[test]
fn test_shared_blocks_not_freed_early() {
    let mut bm = BlockManager::new(10, 4);
    let shared_prefix = vec![1u32, 2, 3, 4]; // exactly 1 full block

    // request 1 allocates 1 fresh block
    bm.allocate(1, &shared_prefix).unwrap();
    assert_eq!(bm.num_free_blocks(), 9);

    bm.mark_computed(1, &shared_prefix).unwrap();
    assert_eq!(bm.num_free_blocks(), 9);

    // request 2 hits cache - ref_count goes 1→2, no fresh blocks
    bm.allocate(2, &shared_prefix).unwrap();
    assert_eq!(bm.num_free_blocks(), 9);

    // free request 1 - ref_count goes 2→1, block stays alive
    bm.free(1);
    assert_eq!(bm.num_free_blocks(), 9);

    // free request 2 - ref_count goes 1→0, block returns to free list
    bm.free(2);
    assert_eq!(bm.num_free_blocks(), 10);
}

#[test]
fn test_append_token_allocates_when_last_block_computed() {
    let mut bm = BlockManager::new(10, 4);
    let prompt = vec![1u32, 2, 3, 4]; // 1 full block

    bm.allocate(1, &prompt).unwrap();
    bm.mark_computed(1, &prompt).unwrap();
    assert_eq!(bm.num_free_blocks(), 9);

    // last block is Computed - append_token must allocate a fresh one
    bm.append_token(1).unwrap();
    assert_eq!(bm.num_free_blocks(), 8);
}

#[test]
fn test_mark_computed_enters_prefix_cache() {
    let mut bm = BlockManager::new(10, 4);
    let prompt = vec![1u32, 2, 3, 4]; // 1 full block

    bm.allocate(1, &prompt).unwrap();
    assert_eq!(bm.num_free_blocks(), 9);

    bm.mark_computed(1, &prompt).unwrap();

    // now allocate a second request with same prompt
    // should hit cache - free blocks unchanged
    bm.allocate(2, &prompt).unwrap();
    assert_eq!(bm.num_free_blocks(), 9);
}
