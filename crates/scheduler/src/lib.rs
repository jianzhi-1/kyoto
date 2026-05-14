use std::collections::VecDeque;
use std::time::Instant;
use types::{Request, RequestId};

#[derive(Debug, Clone)]
pub struct SchedulerRequest {
    pub request: Request,
    pub arrival_time: Instant,
}

#[derive(Debug)]
pub struct PrefillScheduler {
    waiting: VecDeque<Request>,
    running: VecDeque<RequestId>,
    max_batch_tokens: usize,
}

impl PrefillScheduler {
    pub fn new(max_batch_tokens: usize) -> Self {
        return Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            max_batch_tokens,
        };
    }

    pub fn add(&mut self, request: Request) {
        self.waiting.push_back(request);
    }

    pub fn schedule(&mut self) -> Vec<Request> {
        let mut batch: Vec<Request> = Vec::new();
        let mut tokens_scheduled = 0;

        while let Some(request) = self.waiting.front() {
            let request_tokens = request.num_tokens();
            if tokens_scheduled + request_tokens > self.max_batch_tokens {
                break;
            }
            let request = self.waiting.pop_front().unwrap();
            tokens_scheduled += request_tokens;
            self.running.push_back(request.id);
            batch.push(request);
        }

        return batch;
    }

    pub fn complete(&mut self, request_id: RequestId) -> () {
        self.running.retain(|r| {
            return *r != request_id;
        });
    }

    pub fn waiting_len(&self) -> usize {
        return self.waiting.len();
    }

    pub fn running_len(&self) -> usize {
        return self.running.len();
    }
}

#[cfg(test)]
mod tests;
