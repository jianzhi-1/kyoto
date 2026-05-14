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
