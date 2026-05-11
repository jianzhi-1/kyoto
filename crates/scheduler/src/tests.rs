use super::*;

#[test]
fn test_single_request_scheduled() {
    let mut scheduler = PrefillScheduler::new(100);
    scheduler.add(Request::new(1, vec![1, 2, 3]));

    let batch = scheduler.schedule();

    assert_eq!(batch.len(), 1);
    assert_eq!(scheduler.waiting_len(), 0);
    assert_eq!(scheduler.running_len(), 1);
}

#[test]
fn test_token_budget_respected() {
    let mut scheduler = PrefillScheduler::new(5);
    scheduler.add(Request::new(1, vec![1, 2, 3]));
    scheduler.add(Request::new(2, vec![4, 5, 6, 7, 8]));

    let batch = scheduler.schedule();

    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].id, 1);
    assert_eq!(scheduler.waiting_len(), 1);
}

#[test]
fn test_complete_removes_from_running() {
    let mut scheduler = PrefillScheduler::new(100);
    scheduler.add(Request::new(1, vec![1, 2, 3]));
    scheduler.schedule();

    assert_eq!(scheduler.running_len(), 1);
    scheduler.complete(1);
    assert_eq!(scheduler.running_len(), 0);
}

#[test]
fn test_arrival_order_preserved() {
    let mut scheduler = PrefillScheduler::new(100);
    scheduler.add(Request::new(1, vec![1]));
    scheduler.add(Request::new(2, vec![2]));
    scheduler.add(Request::new(3, vec![3]));

    let batch = scheduler.schedule();

    assert_eq!(batch[0].id, 1);
    assert_eq!(batch[1].id, 2);
    assert_eq!(batch[2].id, 3);
}
