# Support Routing Cost Robustness v1

## Pass Counts

| Train mode | Family | Calibration worlds | Passed profiles |
| --- | --- | ---: | ---: |
| cost_aware | validation_loss | 2 | 4 |
| cost_aware | validation_loss | 3 | 4 |
| cost_aware | validation_loss | 5 | 5 |
| cost_aware | conformal_validation_rank | 2 | 4 |
| cost_aware | conformal_validation_rank | 3 | 4 |
| cost_aware | conformal_validation_rank | 5 | 5 |
| cost_aware | router_margin | 2 | 2 |
| cost_aware | router_margin | 3 | 4 |
| cost_aware | router_margin | 5 | 5 |
| cost_aware | support_commutator_reference | 2 | 0 |
| cost_aware | support_commutator_reference | 3 | 1 |
| cost_aware | support_commutator_reference | 5 | 2 |
| default_trained | validation_loss | 2 | 4 |
| default_trained | validation_loss | 3 | 4 |
| default_trained | validation_loss | 5 | 5 |
| default_trained | conformal_validation_rank | 2 | 4 |
| default_trained | conformal_validation_rank | 3 | 4 |
| default_trained | conformal_validation_rank | 5 | 5 |
| default_trained | router_margin | 2 | 2 |
| default_trained | router_margin | 3 | 4 |
| default_trained | router_margin | 5 | 5 |
| default_trained | support_commutator_reference | 2 | 0 |
| default_trained | support_commutator_reference | 3 | 1 |
| default_trained | support_commutator_reference | 5 | 2 |

## Profile Leaders At 3 Calibration Worlds

| Train mode | Cost profile | Leader | Delta mean | Win rate |
| --- | --- | --- | ---: | ---: |
| cost_aware | default | validation_loss | -0.222470 | 0.833 |
| cost_aware | high_structured_risk | validation_loss | -0.222470 | 0.833 |
| cost_aware | high_fallback_delay | validation_loss | -0.598214 | 0.938 |
| cost_aware | cheap_escalation | validation_loss | -0.139286 | 1.000 |
| cost_aware | expensive_escalation | validation_loss | +0.462798 | 0.458 |
| default_trained | default | validation_loss | -0.222470 | 0.833 |
| default_trained | high_structured_risk | validation_loss | -0.222470 | 0.833 |
| default_trained | high_fallback_delay | validation_loss | -0.598214 | 0.938 |
| default_trained | cheap_escalation | validation_loss | -0.139286 | 1.000 |
| default_trained | expensive_escalation | validation_loss | +0.462798 | 0.458 |
