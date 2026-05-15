use crate::{KvCache, Model, ModelError, ModelOutput, Tensor};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use types::{ModelData, TokenId};

pub struct MockModel {
    data: ModelData,
    rng: StdRng,
}

impl MockModel {
    pub fn new(data: ModelData) -> Self {
        return Self {
            data,
            rng: StdRng::seed_from_u64(42),
        };
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        return self;
    }

    fn empty_kv_cache(&self) -> KvCache {
        let shape = vec![
            self.data.num_layers,
            self.data.num_heads,
            0,
            self.data.head_dim,
        ];
        return KvCache {
            keys: Tensor::new(vec![], shape.clone()),
            values: Tensor::new(vec![], shape),
        };
    }
}

impl Model for MockModel {
    /// Returns random logits and zeroed KV cache.
    fn forward(
        &mut self,
        input_tokens: &[TokenId],
        kv_cache: Option<KvCache>,
    ) -> Result<ModelOutput, ModelError> {
        if input_tokens.is_empty() {
            return Err(ModelError::InvalidInput("empty input tokens".to_string()));
        }

        let logits_data = (0..self.data.vocab_size)
            .map(|_| self.rng.gen_range(0.0f32..1.0))
            .collect();
        let logits = Tensor::new(logits_data, vec![self.data.vocab_size as usize]);

        let prev_cache = kv_cache.unwrap_or_else(|| self.empty_kv_cache());
        let prev_seq_len = prev_cache.keys.shape[2];
        let new_seq_len = prev_seq_len + input_tokens.len();

        let kv_shape = vec![
            self.data.num_layers,
            self.data.num_heads,
            new_seq_len,
            self.data.head_dim,
        ];
        let kv_numel = kv_shape.iter().product();
        let updated_kv_cache = KvCache {
            keys: Tensor::new(vec![0.0f32; kv_numel], kv_shape.clone()),
            values: Tensor::new(vec![0.0f32; kv_numel], kv_shape),
        };

        return Ok(ModelOutput {
            logits,
            kv_cache: updated_kv_cache,
        });
    }

    fn data(&self) -> &ModelData {
        return &self.data;
    }
}
