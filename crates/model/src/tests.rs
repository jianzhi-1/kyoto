use super::*;
use crate::MockModel;
use types::ModelData;

fn test_data() -> ModelData {
    return ModelData::new(50257, 0, 12, 12, 64);
}

fn seeded_model() -> MockModel {
    return MockModel::new(test_data()).seed(42);
}

#[test]
fn test_prefill_logit_shape() {
    let mut model = seeded_model();
    let output = model.forward(&[1u32, 2, 3, 4], None).unwrap();
    assert_eq!(output.logits.shape, vec![50257]);
    assert_eq!(output.logits.data.len(), 50257);
}

#[test]
fn test_prefill_kv_cache_shape() {
    let mut model = seeded_model();
    let output = model.forward(&[1u32, 2, 3, 4], None).unwrap();
    assert_eq!(output.kv_cache.keys.shape, vec![12, 12, 4, 64]);
    assert_eq!(output.kv_cache.values.shape, vec![12, 12, 4, 64]);
}

#[test]
fn test_prefill_kv_numel() {
    let mut model = seeded_model();
    let tokens = vec![1u32, 2, 3, 4]; // 4 tokens
    let output = model.forward(&tokens, None).unwrap();

    // num_layers=12, num_heads=12, seq_len=4, head_dim=64
    let expected_numel = 12 * 12 * 4 * 64;
    assert_eq!(output.kv_cache.keys.data.len(), expected_numel);
    assert_eq!(output.kv_cache.values.data.len(), expected_numel);
}

#[test]
fn test_decode_extends_kv_cache_seq_len() {
    let mut model = seeded_model();

    let prefill_output = model.forward(&[1u32, 2, 3, 4], None).unwrap();
    assert_eq!(prefill_output.kv_cache.keys.shape[2], 4);

    let decode1 = model
        .forward(&[5u32], Some(prefill_output.kv_cache))
        .unwrap();
    assert_eq!(decode1.kv_cache.keys.shape[2], 5);
    assert_eq!(decode1.kv_cache.keys.data.len(), 12 * 12 * 5 * 64);

    let decode2 = model.forward(&[6u32], Some(decode1.kv_cache)).unwrap();
    assert_eq!(decode2.kv_cache.keys.shape[2], 6);
    assert_eq!(decode2.kv_cache.keys.data.len(), 12 * 12 * 6 * 64);
}

#[test]
fn test_decode_logit_shape_unchanged() {
    let mut model = seeded_model();
    let prefill = model.forward(&[1u32, 2, 3, 4], None).unwrap();
    let decode = model.forward(&[5u32], Some(prefill.kv_cache)).unwrap();

    // logits always shape [vocab_size] regardless of decode vs prefill
    assert_eq!(decode.logits.shape, vec![50257]);
    assert_eq!(decode.logits.data.len(), 50257);
}

#[test]
fn test_deterministic_with_same_seed() {
    let mut model1 = MockModel::new(test_data()).seed(99);
    let mut model2 = MockModel::new(test_data()).seed(99);

    let out1 = model1.forward(&[1u32, 2, 3], None).unwrap();
    let out2 = model2.forward(&[1u32, 2, 3], None).unwrap();

    assert_eq!(out1.logits.data, out2.logits.data);
}

#[test]
fn test_different_seeds_give_different_logits() {
    let mut model1 = MockModel::new(test_data()).seed(1);
    let mut model2 = MockModel::new(test_data()).seed(2);

    let out1 = model1.forward(&[1u32, 2, 3], None).unwrap();
    let out2 = model2.forward(&[1u32, 2, 3], None).unwrap();

    assert_ne!(out1.logits.data, out2.logits.data);
}

#[test]
fn test_empty_input_returns_error() {
    let mut model = seeded_model();
    let result = model.forward(&[], None);
    assert!(matches!(result, Err(ModelError::InvalidInput(_))));
}

#[test]
fn test_config_values() {
    let model = seeded_model();
    assert_eq!(model.data().vocab_size, 50257);
    assert_eq!(model.data().num_layers, 12);
    assert_eq!(model.data().num_heads, 12);
    assert_eq!(model.data().head_dim, 64);
    assert_eq!(model.data().eos_token_id, 0);
}
