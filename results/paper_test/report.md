# One-Time Paper Test Report

This report was generated once from the frozen paper-test partition. The separate community-hidden partition was not evaluated.

## Controlled authorization

rule,coverage,far,alr,review_rate,moc,worst_profile_certification_volume,raw_far_rank,certified_rank
Lifecycle Checklist,0.29340717299578056,0.14704296243034334,0.0,0.4290084388185654,0.6289568115105588,0.3,4.0,1.0
Cost-Aware Gate,0.3667721518987342,0.0637043428242738,0.2454702329594478,0.0,0.010282369871049735,0.0,1.0,2.0
Confidence Gate,0.38154008438818565,0.09870058059165054,0.25006911805363563,0.0,0.002726405632699636,0.0,2.0,2.0
Uncertainty Gate,0.3857594936708861,0.10910582444626743,0.24788077659283567,0.0,0.009649314003849637,0.0,3.0,2.0
Risk Filter,0.6350738396624472,0.6097500207623952,0.20870359604683997,0.0,5.968474501831947,0.0,5.0,2.0
Hard Role Gate,0.75,0.6557665260196905,0.0,0.0,0.0,0.0,6.0,2.0
Direct Prior,1.0,0.6558016877637131,0.25,0.0,0.0,0.0,7.0,2.0
No Action,0.0,,,0.0,31.202105717706235,0.0,8.0,2.0


## Provenance authorization

rule,coverage,far,alr,direct_leakage_rate,indirect_leakage_rate,safe_delegation_coverage,review_rate
Provenance Hard Gate,0.59125,0.6739957716701903,0.0,0.0,0.0,0.6984553391537945,0.1831
Provenance Learned Gate,0.60315,0.6675785459670065,0.017823095415734062,0.0,0.017823095415734062,0.6739422431161853,0.09955
Lifecycle Checklist,0.3345,0.14559043348281017,0.1654708520179372,0.0,0.1654708520179372,0.9986568166554735,0.2959
EPV Adapter,0.2974,0.039172831203765975,0.16896435776731675,0.0,0.16896435776731675,0.9986568166554735,0.0
Hard Role Gate,0.87865,0.6741592215330336,0.17168383315313265,0.0,0.17168383315313265,1.0,0.12135
Soft Penalty,0.4065,0.1979089790897909,0.23788437884378844,0.08536285362853628,0.15252152521525214,0.9993284083277367,0.0
Shared Threshold,0.37135,0.1221219873434765,0.2498990170997711,0.09882859835734482,0.15107041874242627,0.9993284083277367,0.0
No Role Gate,1.0,0.6737,0.25,0.09915,0.15085,1.0,0.0
No Action,0.0,,,,,0.0,0.0


## Validation-to-paper rank stability

{
  "certified_rank": {
    "rank": "certified_rank",
    "rules": 8,
    "spearman": 0.9999999999999999
  },
  "raw_far_rank": {
    "rank": "raw_far_rank",
    "rules": 8,
    "spearman": 1.0
  }
}

## Interpretation boundary

These are controlled and provenance paper-test results. They are not public market deployment evidence, practitioner labels, trading profitability, or a global ranking of execution rules. Any divergence from validation is retained.