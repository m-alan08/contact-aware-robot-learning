# h2_001a summary (generated)

Gates: {"L1_nominal_success_A": 1.0, "L1_teacher_recovery": 1.0, "L1_pass": true, "L3_nominal_success_A": 1.0, "L3_teacher_recovery": 1.0, "L3_pass": true}

## Primary family (L=1, recoverable, load+push; Holm over P1-P4)

| | contrast | estimate [95% CI] | episodes-only CI | seeds-only CI | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| P1 | B_refresh - A_cached | -0.009 [-0.029, +0.010] | [-0.023, +0.005] | [-0.019, -0.001] | 1 | equivalent within delta |
| P2 | C_null - A_cached | -0.045 [-0.069, -0.024] | [-0.062, -0.029] | [-0.055, -0.036] | 0 | negative (<= -delta) |
| P3 | F_gate - A_cached | +0.001 [-0.003, +0.007] | [-0.002, +0.005] | [+0.000, +0.003] | 1 | equivalent within delta |
| P4 | F_gate - G_random | +0.001 [-0.007, +0.008] | [-0.004, +0.006] | [-0.003, +0.003] | 1 | equivalent within delta |
| P5 | F_gate - A_cached (nominal) | +0.000 [+0.000, +0.000] | | | | non-inferior: True |

## Mechanisms

```
{
 "L1": {
  "M1": {
   "established": true,
   "fc_err_A_minus_B": "+0.063 [+0.053, +0.073]",
   "A_disturbed": "+0.122 [+0.114, +0.131]",
   "A_nominal_window": "+0.018 [+0.017, +0.019]"
  },
  "M2": {
   "level": "weak",
   "nominal_action_sensitivity_null": "+0.001 [+0.001, +0.002]",
   "nominal_action_sensitivity_refresh_mean": 0.007248299598121467,
   "C_minus_A_nominal_success": "+0.000 [+0.000, +0.000]",
   "C_minus_A_recovery": "-0.045 [-0.069, -0.024]"
  },
  "M3": {
   "established": false,
   "refresh_change": "+0.003 [+0.003, +0.003]",
   "refresh_interference": "-0.001 [-0.002, -0.001]",
   "refresh_under_grip": "-0.002 [-0.002, -0.001]",
   "null_change": "+0.003 [+0.002, +0.003]",
   "null_interference": "-0.001 [-0.001, -0.001]",
   "null_under_grip": "-0.001 [-0.001, -0.001]",
   "shuffle_change": "+0.004 [+0.004, +0.005]",
   "shuffle_interference": "-0.001 [-0.001, -0.001]",
   "shuffle_under_grip": "-0.000 [-0.001, -0.000]",
   "oracle_change": "+0.012 [+0.011, +0.013]",
   "oracle_interference": "-0.005 [-0.007, -0.003]",
   "oracle_under_grip": "-0.003 [-0.006, -0.001]"
  },
  "M4": {
   "established": false,
   "B_minus_A_recovery": "-0.009 [-0.029, +0.010]"
  },
  "OOD": {
   "cached_nominal_probes_frac_above_nominal_q99": 0.0272,
   "cached_disturbed_probes_frac_above_nominal_q99": 0.19164912280701754,
   "refresh_nominal_probes_frac_above_nominal_q99": 0.2112,
   "refresh_disturbed_probes_frac_above_nominal_q99": 0.17687719298245613,
   "null_nominal_probes_frac_above_nominal_q99": 0.0302,
   "null_disturbed_probes_frac_above_nominal_q99": 0.19536842105263158,
   "shuffle_nominal_probes_frac_above_nominal_q99": 0.264,
   "shuffle_disturbed_probes_frac_above_nominal_q99": 0.219859649122807,
   "oracle_nominal_probes_frac_above_nominal_q99": 0.2367,
   "oracle_disturbed_probes_frac_above_nominal_q99": 0.4270877192982456
  }
 },
 "L3": {
  "M1": {
   "established": true,
   "fc_err_A_minus_B": "+0.023 [+0.015, +0.030]",
   "A_disturbed": "+0.147 [+0.134, +0.160]",
   "A_nominal_window": "+0.019 [+0.018, +0.020]"
  },
  "M2": {
   "level": "absent",
   "nominal_action_sensitivity_null": "+0.002 [+0.001, +0.002]",
   "nominal_action_sensitivity_refresh_mean": 0.007157618646873406,
   "C_minus_A_nominal_success": "+0.000 [+0.000, +0.000]",
   "C_minus_A_recovery": "-0.026 [-0.051, -0.002]"
  },
  "M3": {
   "established": false,
   "refresh_change": "+0.002 [+0.002, +0.003]",
   "refresh_interference": "-0.001 [-0.002, -0.001]",
   "refresh_under_grip": "-0.001 [-0.002, -0.001]",
   "null_change": "+0.002 [+0.002, +0.002]",
   "null_interference": "-0.001 [-0.001, -0.001]",
   "null_under_grip": "-0.001 [-0.001, -0.000]",
   "shuffle_change": "+0.004 [+0.004, +0.005]",
   "shuffle_interference": "-0.001 [-0.001, -0.000]",
   "shuffle_under_grip": "-0.000 [-0.001, +0.000]",
   "oracle_change": "+0.011 [+0.010, +0.012]",
   "oracle_interference": "-0.003 [-0.005, -0.000]",
   "oracle_under_grip": "-0.002 [-0.005, +0.001]"
  },
  "M4": {
   "established": false,
   "B_minus_A_recovery": "-0.025 [-0.055, +0.005]"
  },
  "OOD": {
   "cached_nominal_probes_frac_above_nominal_q99": 0.0227,
   "cached_disturbed_probes_frac_above_nominal_q99": 0.09564912280701754,
   "refresh_nominal_probes_frac_above_nominal_q99": 0.1887,
   "refresh_disturbed_probes_frac_above_nominal_q99": 0.0907719298245614,
   "null_nominal_probes_frac_above_nominal_q99": 0.0205,
   "null_disturbed_probes_frac_above_nominal_q99": 0.10291228070175439,
   "shuffle_nominal_probes_frac_above_nominal_q99": 0.1437,
   "shuffle_disturbed_probes_frac_above_nominal_q99": 0.11610526315789474,
   "oracle_nominal_probes_frac_above_nominal_q99": 0.1948,
   "oracle_disturbed_probes_frac_above_nominal_q99": 0.34182456140350875
  }
 }
}
```

**H2 verdict (L=1, protocol §9): contradicted**

## Reactive / live-only

```
{
 "L1_R_reactive_vs_F_gate": {
  "recovery": "+0.091 [+0.058, +0.125]",
  "nominal": "+0.000 [+0.000, +0.000]",
  "matches": true
 },
 "L1_D_live_only_vs_F_gate": {
  "recovery": "+0.003 [-0.014, +0.019]",
  "nominal": "+0.000 [+0.000, +0.000]",
  "matches": true
 },
 "L3_R_reactive_vs_F_gate": {
  "recovery": "+0.251 [+0.201, +0.301]",
  "nominal": "+0.000 [+0.000, +0.000]",
  "matches": true
 },
 "L3_D_live_only_vs_F_gate": {
  "recovery": "+0.030 [+0.001, +0.058]",
  "nominal": "+0.000 [+0.000, +0.000]",
  "matches": true
 }
}
```

## False interventions

```
{
 "L1": {
  "episodes_with_trigger": 0.236,
  "triggers_per_step": 0.00295,
  "disturbed_first_trigger_delay_steps_median": 12.0,
  "disturbed_episodes_with_trigger": 0.059333333333333335
 },
 "L3": {
  "episodes_with_trigger": 0.232,
  "triggers_per_step": 0.0029,
  "disturbed_first_trigger_delay_steps_median": 11.0,
  "disturbed_episodes_with_trigger": 0.2826666666666667
 }
}
```