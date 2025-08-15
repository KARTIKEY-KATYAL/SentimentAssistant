from modules.customer_satisfaction import CustomerSatisfactionTracker


def test_satisfaction_trend():
    tracker = CustomerSatisfactionTracker()
    # Add two windows of feedback
    for r in [2,2,2,2,2]:
        tracker.add_feedback("m1", r)
    for r in [4,4,4,4,4]:
        tracker.add_feedback("m2", r)
    assert tracker.rating_trend(window=5) == "improving"
    avg = tracker.average_rating()
    assert 2.9 < avg < 3.1  # average should be around 3
