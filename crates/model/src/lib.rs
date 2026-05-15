use types::{ModelData, TokenId};
mod mock;
pub use mock::MockModel;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        return Self { data, shape };
    }

    pub fn numel(&self) -> usize {
        return self.shape.iter().product();
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub keys: Tensor,
    pub values: Tensor,
}

#[derive(Debug)]
pub enum ModelError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidInput(String),
    BackendError(String),
}

#[derive(Debug)]
pub struct ModelOutput {
    pub logits: Tensor,
    pub kv_cache: KvCache,
}

pub trait Model {
    /// Generate random logits over vocabulary. Optionality of KV cache signifies decode vs prefill phase.
    fn forward(
        &mut self,
        input_tokens: &[TokenId],
        kv_cache: Option<KvCache>,
    ) -> Result<ModelOutput, ModelError>;

    fn data(&self) -> &ModelData;
}

#[cfg(test)]
mod tests;
