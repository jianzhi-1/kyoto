pub type TokenId = u32;

pub type RequestId = u64;

pub type BlockId = usize;

pub type ContentHash = u64;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    pub prompt_tokens: Vec<TokenId>,
}

impl Request {
    pub fn new(id: RequestId, prompt_tokens: Vec<TokenId>) -> Self {
        return Self { id, prompt_tokens };
    }

    pub fn num_tokens(&self) -> usize {
        return self.prompt_tokens.len();
    }
}

#[derive(Debug, Clone)]
pub struct ModelData {
    pub vocab_size: u32,
    pub eos_token_id: TokenId,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl ModelData {
    pub fn new(
        vocab_size: u32,
        eos_token_id: TokenId,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        return Self {
            vocab_size,
            eos_token_id,
            num_layers,
            num_heads,
            head_dim,
        };
    }
}
