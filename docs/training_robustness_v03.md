# Prospective Training Robustness v0.3

## Scope

This is a prospectively versioned, validation-only secondary robustness study. It does not modify the registered v0.2 Logistic Regression primary, replace its structural N/A outcomes, evaluate any test split, or support deployment claims.

## Design

Learners: Logistic, HistGB, MLP, Selective logistic. Training budgets: 400, 1000, 5000, 20000 independent clusters. A primary endpoint is defined only when all 6 mechanisms have a one-sided 95% coverage lower bound of at least 0.05. Thresholds use seen validation mechanisms only; evaluation mechanisms remain disjoint from threshold selection.

## Result

At 20,000 clusters, the highest valid-mechanism fraction is 0.833 for full_audit_seen with Logistic. The result diagnoses whether endpoint definition recovers with learner capacity and data budget; it is not a benchmark winner claim.

## Learning-curve summary

curriculum,budget,learner,valid_mechanism_fraction_mean,valid_mechanism_fraction_std,all_mechanisms_valid_seeds,primary_far_mean,primary_far_std,mechanism_coverage_lcb_floor_mean
controlled_seen,400,histogram_gradient_boosting,0.5555555555555555,0.1924500897298752,0,,,0.0002902914292153
controlled_seen,400,logistic_regression,0.3333333333333333,0.0,0,,,0.0
controlled_seen,400,mlp,0.3333333333333333,0.0,0,,,0.0
controlled_seen,400,selective_logistic,0.3333333333333333,0.0,0,,,0.0
controlled_seen,1000,histogram_gradient_boosting,0.5,0.1666666666666666,0,,,0.0
controlled_seen,1000,logistic_regression,0.3333333333333333,0.0,0,,,0.0
controlled_seen,1000,mlp,0.2777777777777777,0.0962250448649376,0,,,0.0
controlled_seen,1000,selective_logistic,0.3333333333333333,0.0,0,,,0.0
controlled_seen,5000,histogram_gradient_boosting,0.3333333333333333,0.0,0,,,0.0
controlled_seen,5000,logistic_regression,0.3333333333333333,0.0,0,,,0.0
controlled_seen,5000,mlp,0.3333333333333333,0.0,0,,,0.0
controlled_seen,5000,selective_logistic,0.3333333333333333,0.0,0,,,0.0
controlled_seen,20000,histogram_gradient_boosting,0.3333333333333333,0.0,0,,,0.0
controlled_seen,20000,logistic_regression,0.3333333333333333,0.0,0,,,0.0
controlled_seen,20000,mlp,0.3333333333333333,0.0,0,,,0.0
controlled_seen,20000,selective_logistic,0.3333333333333333,0.0,0,,,0.0
full_audit_seen,400,histogram_gradient_boosting,0.4444444444444444,0.1924500897298752,0,,,0.0010573793711642
full_audit_seen,400,logistic_regression,0.611111111111111,0.0962250448649376,0,,,0.0040379290430487
full_audit_seen,400,mlp,0.3333333333333333,0.1666666666666666,0,,,0.0
full_audit_seen,400,selective_logistic,0.2222222222222222,0.2545875386086578,0,,,0.0010900367419818
full_audit_seen,1000,histogram_gradient_boosting,0.5,0.1666666666666666,0,,,0.0
full_audit_seen,1000,logistic_regression,0.7777777777777778,0.0962250448649376,0,,,0.0035472071921163
full_audit_seen,1000,mlp,0.1666666666666666,0.0,0,,,7.38668938025208e-05
full_audit_seen,1000,selective_logistic,0.3333333333333333,0.2886751345948128,0,,,0.0010964563152803
full_audit_seen,5000,histogram_gradient_boosting,0.3333333333333333,0.0,0,,,0.0
full_audit_seen,5000,logistic_regression,0.8333333333333334,0.0,0,,,0.0041102333762466
full_audit_seen,5000,mlp,0.1666666666666666,0.0,0,,,0.0
full_audit_seen,5000,selective_logistic,0.6111111111111112,0.3849001794597506,0,,,0.0028298967177246
full_audit_seen,20000,histogram_gradient_boosting,0.3888888888888888,0.0962250448649376,0,,,0.0
full_audit_seen,20000,logistic_regression,0.8333333333333334,0.0,0,,,0.003925815626799
full_audit_seen,20000,mlp,0.2777777777777777,0.0962250448649376,0,,,0.0
full_audit_seen,20000,selective_logistic,0.8333333333333334,0.0,0,,,0.003925815626799
multi_generator,400,histogram_gradient_boosting,0.3888888888888888,0.0962250448649376,0,,,0.0
multi_generator,400,logistic_regression,0.5,0.1666666666666666,0,,,0.0016011405349852
multi_generator,400,mlp,0.3888888888888888,0.0962250448649376,0,,,0.0006805339693841
multi_generator,400,selective_logistic,0.3333333333333333,0.0,0,,,0.0001337988151965
multi_generator,1000,histogram_gradient_boosting,0.4444444444444444,0.0962250448649376,0,,,0.0
multi_generator,1000,logistic_regression,0.5,0.1666666666666666,0,,,0.0020201288974216
multi_generator,1000,mlp,0.3888888888888888,0.0962250448649376,0,,,0.0
multi_generator,1000,selective_logistic,0.3333333333333333,0.0,0,,,0.0001696251720835
multi_generator,5000,histogram_gradient_boosting,0.3333333333333333,0.0,0,,,0.0
multi_generator,5000,logistic_regression,0.3888888888888888,0.0962250448649376,0,,,0.0023420956016535
multi_generator,5000,mlp,0.3333333333333333,0.0,0,,,0.0
multi_generator,5000,selective_logistic,0.3333333333333333,0.0,0,,,0.0
multi_generator,20000,histogram_gradient_boosting,0.3333333333333333,0.0,0,,,0.0
multi_generator,20000,logistic_regression,0.4444444444444444,0.0962250448649376,0,,,0.0022992909121957
multi_generator,20000,mlp,0.3333333333333333,0.0,0,,,0.0
multi_generator,20000,selective_logistic,0.3333333333333333,0.0,0,,,0.0


## Interpretation boundary

The sequential and institutional generators are mechanistically distinct but remain repository-authored controlled generators. They improve generator sensitivity analysis but do not replace practitioner labels or independent external validation.