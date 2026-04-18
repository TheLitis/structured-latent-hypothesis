# Regime-Gated Routing v1

| router | threshold | synthetic mse | semi-real mse | pooled mse | pooled gain vs coord | pooled regret | pooled capture ratio | semi-real route rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_world_best | n/a | 0.000956 | 0.000128 | 0.000580 | +0.000064 | 0.000000 | 1.000 | 0.00 |
| router_pairA_curv_hankel_shared_joint | 6.809890 | 0.000986 | 0.000128 | 0.000596 | +0.000047 | 0.000000 | 1.000 | 0.00 |
| router_pairA_curv_hankel_inverted | 6.809890 | 0.001232 | 0.000174 | 0.000751 | -0.000108 | 0.000155 | -2.288 | 1.00 |
| router_pairA_curv_hankel_meta_cpl_control | 0.113760 | 0.000986 | 0.000174 | 0.000617 | +0.000026 | 0.000021 | 0.558 | 1.00 |
| always_coord_pairA_curv_hankel | n/a | 0.001072 | 0.000128 | 0.000643 | +0.000000 | 0.000047 | 0.000 | 0.00 |
| always_structured_pairA_curv_hankel | n/a | 0.001145 | 0.000174 | 0.000704 | -0.000061 | 0.000108 | -1.288 | 1.00 |
| oracle_threshold_pairA_curv_hankel | 6.809890 | 0.000977 | 0.000128 | 0.000591 | +0.000052 | -0.000005 | 1.108 | 0.00 |
| router_pairB_operator_diag_shared_joint | 6.809890 | 0.001013 | 0.000128 | 0.000611 | +0.000032 | 0.000000 | 1.000 | 0.00 |
| router_pairB_operator_diag_inverted | 6.809890 | 0.001180 | 0.000144 | 0.000709 | -0.000066 | 0.000098 | -2.047 | 1.00 |
| router_pairB_operator_diag_meta_cpl_control | 0.113760 | 0.001013 | 0.000144 | 0.000618 | +0.000025 | 0.000007 | 0.774 | 1.00 |
| always_coord_pairB_operator_diag | n/a | 0.001072 | 0.000128 | 0.000643 | +0.000000 | 0.000032 | 0.000 | 0.00 |
| always_structured_pairB_operator_diag | n/a | 0.001121 | 0.000144 | 0.000677 | -0.000034 | 0.000066 | -1.047 | 1.00 |
| oracle_threshold_pairB_operator_diag | 6.809890 | 0.000977 | 0.000128 | 0.000591 | +0.000052 | -0.000020 | 1.620 | 0.00 |
