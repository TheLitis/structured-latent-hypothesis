# Regime-Gated Routing v2

| router | pooled mse | pooled gain vs coord | pooled regret | semi-real mse | semi-real gain vs coord | semi-real route rate | coverage | filtered sign acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared_train_pairA_curv_hankel | 0.000606 | +0.000037 | 0.000010 | 0.000146 | -0.000018 | 0.40 | 1.00 | 0.80+/-0.25 |
| shared_train_pairA_curv_hankel_always_coord | 0.000643 | +0.000000 | 0.000047 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.73+/-0.00 |
| shared_train_pairA_curv_hankel_always_structured | 0.000704 | -0.000061 | 0.000108 | 0.000174 | -0.000046 | 1.00 | 1.00 | 0.27+/-0.00 |
| shared_train_pairA_curv_hankel_inverted_router | 0.000741 | -0.000098 | 0.000145 | 0.000156 | -0.000027 | 0.60 | 1.00 | 0.20+/-0.25 |
| shared_train_pairA_curv_hankel_metadata_control_router | 0.000617 | +0.000026 | 0.000021 | 0.000174 | -0.000046 | 1.00 | 1.00 | 0.55+/-0.00 |
| shared_train_pairA_curv_hankel_random_router | 0.000672 | -0.000029 | 0.000076 | 0.000146 | -0.000018 | 0.40 | 1.00 | 0.51+/-0.12 |
| shared_train_pairB_operator_diag | 0.000617 | +0.000026 | 0.000007 | 0.000135 | -0.000006 | 0.40 | 1.00 | 0.84+/-0.19 |
| shared_train_pairB_operator_diag_always_coord | 0.000643 | +0.000000 | 0.000032 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.67+/-0.00 |
| shared_train_pairB_operator_diag_always_structured | 0.000677 | -0.000034 | 0.000066 | 0.000144 | -0.000016 | 1.00 | 1.00 | 0.33+/-0.00 |
| shared_train_pairB_operator_diag_inverted_router | 0.000702 | -0.000059 | 0.000092 | 0.000138 | -0.000010 | 0.60 | 1.00 | 0.16+/-0.19 |
| shared_train_pairB_operator_diag_metadata_control_router | 0.000618 | +0.000025 | 0.000007 | 0.000144 | -0.000016 | 1.00 | 1.00 | 0.67+/-0.00 |
| shared_train_pairB_operator_diag_random_router | 0.000659 | -0.000016 | 0.000048 | 0.000135 | -0.000006 | 0.40 | 1.00 | 0.51+/-0.09 |
| synthetic_only_train_pairA_curv_hankel | 0.000597 | +0.000046 | 0.000001 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.98+/-0.04 |
| synthetic_only_train_pairA_curv_hankel_always_coord | 0.000643 | +0.000000 | 0.000047 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.73+/-0.00 |
| synthetic_only_train_pairA_curv_hankel_always_structured | 0.000704 | -0.000061 | 0.000108 | 0.000174 | -0.000046 | 1.00 | 1.00 | 0.27+/-0.00 |
| synthetic_only_train_pairA_curv_hankel_inverted_router | 0.000750 | -0.000107 | 0.000154 | 0.000174 | -0.000046 | 1.00 | 1.00 | 0.02+/-0.04 |
| synthetic_only_train_pairA_curv_hankel_metadata_control_router | 0.000617 | +0.000026 | 0.000021 | 0.000174 | -0.000046 | 1.00 | 1.00 | 0.55+/-0.00 |
| synthetic_only_train_pairA_curv_hankel_random_router | 0.000661 | -0.000017 | 0.000065 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.60+/-0.02 |
| synthetic_only_train_pairB_operator_diag | 0.000615 | +0.000029 | 0.000004 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.98+/-0.04 |
| synthetic_only_train_pairB_operator_diag_always_coord | 0.000643 | +0.000000 | 0.000032 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.67+/-0.00 |
| synthetic_only_train_pairB_operator_diag_always_structured | 0.000677 | -0.000034 | 0.000066 | 0.000144 | -0.000016 | 1.00 | 1.00 | 0.33+/-0.00 |
| synthetic_only_train_pairB_operator_diag_inverted_router | 0.000705 | -0.000062 | 0.000095 | 0.000144 | -0.000016 | 1.00 | 1.00 | 0.02+/-0.04 |
| synthetic_only_train_pairB_operator_diag_metadata_control_router | 0.000618 | +0.000025 | 0.000007 | 0.000144 | -0.000016 | 1.00 | 1.00 | 0.67+/-0.00 |
| synthetic_only_train_pairB_operator_diag_random_router | 0.000653 | -0.000010 | 0.000042 | 0.000128 | +0.000000 | 0.00 | 1.00 | 0.57+/-0.01 |
