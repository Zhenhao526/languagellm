# Community merge aggregate

parents=36, children=432

## Parent endpoints

| population | community | all natural | all permuted | message gap |
|---|---:|---:|---:|---:|
| aligned | 0 | 0.584 [0.507, 0.661] | 0.018 [-0.023, 0.059] | 0.566 [0.476, 0.656] |
| aligned | 1 | 0.584 [0.507, 0.661] | 0.018 [-0.023, 0.059] | 0.566 [0.476, 0.656] |
| conflict | 0 | 0.584 [0.507, 0.661] | 0.018 [-0.023, 0.059] | 0.566 [0.476, 0.656] |
| conflict | 1 | 0.584 [0.507, 0.661] | 0.018 [-0.023, 0.059] | 0.566 [0.476, 0.656] |

## Child endpoints

| population | visibility | adaptation | support | role | combo natural | value natural | all message gap | fresh consistency | community hamming | functional combo/value |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| aligned | hidden | coadapt | full | alternating | 0.514 [0.300, 0.727] | 0.428 [0.302, 0.554] | 0.398 [0.303, 0.494] | 1.000 [1.000, 1.000] | 0.309 [0.151, 0.466] | 3/9 ; 1/9 |
| aligned | hidden | coadapt | full | sender_only | 0.812 [0.620, 1.004] | 0.758 [0.633, 0.884] | 0.743 [0.655, 0.830] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 6/9 ; 7/9 |
| aligned | hidden | coadapt | heldout_combo | alternating | -0.250 [-0.250, -0.250] | 0.450 [0.307, 0.593] | 0.433 [0.364, 0.502] | 1.000 [1.000, 1.000] | 0.222 [0.150, 0.295] | 0/9 ; 2/9 |
| aligned | hidden | coadapt | heldout_combo | sender_only | -0.250 [-0.250, -0.250] | 0.664 [0.499, 0.828] | 0.636 [0.528, 0.743] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/9 ; 6/9 |
| aligned | hidden | coadapt | heldout_value | alternating | 0.415 [0.053, 0.777] | -0.250 [-0.250, -0.250] | 0.371 [0.291, 0.451] | 1.000 [1.000, 1.000] | 0.198 [0.110, 0.285] | 5/9 ; 0/9 |
| aligned | hidden | coadapt | heldout_value | sender_only | 0.504 [0.121, 0.888] | -0.250 [-0.250, -0.250] | 0.467 [0.370, 0.564] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 5/9 ; 0/9 |
| aligned | hidden | fresh_only | full | alternating | 0.349 [0.108, 0.590] | 0.393 [0.260, 0.525] | 0.364 [0.257, 0.472] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 3/9 ; 1/9 |
| aligned | hidden | fresh_only | full | sender_only | 0.755 [0.554, 0.955] | 0.667 [0.536, 0.798] | 0.678 [0.598, 0.758] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 5/9 ; 4/9 |
| aligned | hidden | fresh_only | heldout_combo | alternating | -0.057 [-0.205, 0.092] | 0.375 [0.283, 0.468] | 0.384 [0.311, 0.457] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/9 ; 0/9 |
| aligned | hidden | fresh_only | heldout_combo | sender_only | 0.137 [-0.160, 0.434] | 0.579 [0.482, 0.677] | 0.572 [0.450, 0.694] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1/9 ; 3/9 |
| aligned | hidden | fresh_only | heldout_value | alternating | 0.382 [0.059, 0.704] | -0.139 [-0.185, -0.093] | 0.279 [0.175, 0.383] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 4/9 ; 0/9 |
| aligned | hidden | fresh_only | heldout_value | sender_only | 0.479 [0.158, 0.800] | -0.031 [-0.122, 0.060] | 0.357 [0.201, 0.512] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 3/9 ; 0/9 |
| aligned | visible | coadapt | full | alternating | 0.148 [0.016, 0.279] | 0.156 [0.088, 0.225] | 0.142 [0.063, 0.221] | 0.535 [0.461, 0.609] | 0.333 [0.175, 0.492] | 0/9 ; 0/9 |
| aligned | visible | coadapt | full | sender_only | 0.398 [0.125, 0.671] | 0.465 [0.291, 0.639] | 0.394 [0.263, 0.524] | 0.432 [0.361, 0.503] | 0.000 [0.000, 0.000] | 3/9 ; 3/9 |
| aligned | visible | coadapt | heldout_combo | alternating | -0.250 [-0.250, -0.250] | 0.178 [0.089, 0.267] | 0.159 [0.086, 0.232] | 0.570 [0.496, 0.644] | 0.222 [0.141, 0.303] | 0/9 ; 0/9 |
| aligned | visible | coadapt | heldout_combo | sender_only | -0.250 [-0.250, -0.250] | 0.465 [0.265, 0.666] | 0.386 [0.287, 0.484] | 0.502 [0.410, 0.594] | 0.000 [0.000, 0.000] | 0/9 ; 3/9 |
| aligned | visible | coadapt | heldout_value | alternating | 0.202 [-0.039, 0.442] | -0.250 [-0.250, -0.250] | 0.160 [0.091, 0.229] | 0.685 [0.599, 0.771] | 0.235 [0.167, 0.302] | 1/9 ; 0/9 |
| aligned | visible | coadapt | heldout_value | sender_only | 0.328 [-0.009, 0.666] | -0.250 [-0.250, -0.250] | 0.323 [0.211, 0.435] | 0.640 [0.599, 0.681] | 0.000 [0.000, 0.000] | 3/9 ; 0/9 |
| aligned | visible | fresh_only | full | alternating | 0.043 [-0.072, 0.159] | 0.107 [0.048, 0.167] | 0.092 [0.011, 0.172] | 0.562 [0.487, 0.636] | 0.000 [0.000, 0.000] | 0/9 ; 0/9 |
| aligned | visible | fresh_only | full | sender_only | 0.293 [0.035, 0.551] | 0.374 [0.221, 0.527] | 0.258 [0.152, 0.364] | 0.449 [0.365, 0.532] | 0.000 [0.000, 0.000] | 2/9 ; 1/9 |
| aligned | visible | fresh_only | heldout_combo | alternating | -0.057 [-0.205, 0.092] | 0.098 [0.038, 0.157] | 0.092 [0.013, 0.170] | 0.576 [0.499, 0.653] | 0.000 [0.000, 0.000] | 0/9 ; 0/9 |
| aligned | visible | fresh_only | heldout_combo | sender_only | 0.137 [-0.160, 0.434] | 0.311 [0.186, 0.435] | 0.237 [0.126, 0.347] | 0.484 [0.412, 0.555] | 0.000 [0.000, 0.000] | 1/9 ; 0/9 |
| aligned | visible | fresh_only | heldout_value | alternating | 0.024 [-0.116, 0.165] | -0.139 [-0.185, -0.093] | 0.058 [-0.028, 0.144] | 0.671 [0.594, 0.748] | 0.000 [0.000, 0.000] | 0/9 ; 0/9 |
| aligned | visible | fresh_only | heldout_value | sender_only | 0.049 [-0.124, 0.223] | -0.031 [-0.122, 0.060] | 0.128 [-0.015, 0.271] | 0.644 [0.582, 0.706] | 0.000 [0.000, 0.000] | 0/9 ; 0/9 |
| conflict | hidden | coadapt | full | alternating | 0.131 [-0.030, 0.291] | 0.206 [0.125, 0.286] | 0.209 [0.163, 0.254] | 1.000 [1.000, 1.000] | 1.926 [1.875, 1.977] | 0/9 ; 0/9 |
| conflict | hidden | coadapt | full | sender_only | 0.352 [0.062, 0.643] | 0.449 [0.271, 0.627] | 0.350 [0.225, 0.475] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 2/9 ; 2/9 |
| conflict | hidden | coadapt | heldout_combo | alternating | -0.250 [-0.250, -0.250] | 0.213 [0.080, 0.347] | 0.213 [0.175, 0.250] | 1.000 [1.000, 1.000] | 1.926 [1.875, 1.977] | 0/9 ; 0/9 |
| conflict | hidden | coadapt | heldout_combo | sender_only | -0.250 [-0.250, -0.250] | 0.408 [0.260, 0.555] | 0.371 [0.281, 0.461] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 0/9 ; 2/9 |
| conflict | hidden | coadapt | heldout_value | alternating | 0.095 [-0.086, 0.277] | -0.250 [-0.250, -0.250] | 0.165 [0.112, 0.218] | 1.000 [1.000, 1.000] | 1.901 [1.834, 1.969] | 0/9 ; 0/9 |
| conflict | hidden | coadapt | heldout_value | sender_only | 0.220 [-0.062, 0.502] | -0.250 [-0.250, -0.250] | 0.290 [0.200, 0.380] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 1/9 ; 0/9 |
| conflict | hidden | fresh_only | full | alternating | 0.070 [-0.079, 0.220] | 0.149 [0.075, 0.223] | 0.133 [0.075, 0.190] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | hidden | fresh_only | full | sender_only | 0.228 [-0.085, 0.542] | 0.315 [0.150, 0.479] | 0.210 [0.106, 0.315] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 2/9 ; 2/9 |
| conflict | hidden | fresh_only | heldout_combo | alternating | -0.100 [-0.227, 0.026] | 0.145 [0.048, 0.242] | 0.154 [0.103, 0.205] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | hidden | fresh_only | heldout_combo | sender_only | 0.051 [-0.201, 0.303] | 0.251 [0.126, 0.375] | 0.217 [0.091, 0.344] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 1/9 ; 0/9 |
| conflict | hidden | fresh_only | heldout_value | alternating | -0.019 [-0.136, 0.098] | -0.134 [-0.168, -0.100] | 0.089 [0.028, 0.150] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | hidden | fresh_only | heldout_value | sender_only | 0.010 [-0.112, 0.131] | -0.019 [-0.088, 0.049] | 0.080 [-0.031, 0.191] | 1.000 [1.000, 1.000] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | visible | coadapt | full | alternating | 0.142 [0.000, 0.283] | 0.182 [0.122, 0.243] | 0.146 [0.122, 0.170] | 0.465 [0.382, 0.548] | 1.938 [1.886, 1.991] | 0/9 ; 0/9 |
| conflict | visible | coadapt | full | sender_only | 0.259 [0.046, 0.473] | 0.416 [0.305, 0.528] | 0.330 [0.214, 0.446] | 0.325 [0.269, 0.381] | 2.000 [2.000, 2.000] | 1/9 ; 1/9 |
| conflict | visible | coadapt | heldout_combo | alternating | -0.250 [-0.250, -0.250] | 0.183 [0.058, 0.308] | 0.140 [0.101, 0.180] | 0.516 [0.442, 0.591] | 1.951 [1.912, 1.989] | 0/9 ; 0/9 |
| conflict | visible | coadapt | heldout_combo | sender_only | -0.250 [-0.250, -0.250] | 0.438 [0.272, 0.604] | 0.333 [0.254, 0.412] | 0.387 [0.322, 0.452] | 2.000 [2.000, 2.000] | 0/9 ; 2/9 |
| conflict | visible | coadapt | heldout_value | alternating | 0.154 [-0.066, 0.373] | -0.250 [-0.250, -0.250] | 0.135 [0.108, 0.162] | 0.628 [0.571, 0.684] | 1.914 [1.853, 1.974] | 1/9 ; 0/9 |
| conflict | visible | coadapt | heldout_value | sender_only | 0.214 [-0.053, 0.481] | -0.250 [-0.250, -0.250] | 0.277 [0.192, 0.362] | 0.531 [0.505, 0.557] | 2.000 [2.000, 2.000] | 2/9 ; 0/9 |
| conflict | visible | fresh_only | full | alternating | 0.069 [-0.061, 0.199] | 0.132 [0.078, 0.185] | 0.083 [0.054, 0.112] | 0.481 [0.401, 0.562] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | visible | fresh_only | full | sender_only | 0.193 [-0.031, 0.416] | 0.344 [0.225, 0.464] | 0.213 [0.113, 0.314] | 0.335 [0.280, 0.391] | 2.000 [2.000, 2.000] | 1/9 ; 1/9 |
| conflict | visible | fresh_only | heldout_combo | alternating | -0.100 [-0.227, 0.026] | 0.137 [0.053, 0.221] | 0.087 [0.056, 0.117] | 0.514 [0.440, 0.588] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | visible | fresh_only | heldout_combo | sender_only | 0.051 [-0.201, 0.303] | 0.277 [0.153, 0.400] | 0.198 [0.093, 0.303] | 0.354 [0.292, 0.415] | 2.000 [2.000, 2.000] | 1/9 ; 0/9 |
| conflict | visible | fresh_only | heldout_value | alternating | 0.057 [-0.106, 0.220] | -0.134 [-0.168, -0.100] | 0.069 [0.025, 0.113] | 0.648 [0.586, 0.710] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |
| conflict | visible | fresh_only | heldout_value | sender_only | 0.030 [-0.120, 0.179] | -0.019 [-0.088, 0.049] | 0.109 [0.006, 0.213] | 0.525 [0.482, 0.567] | 2.000 [2.000, 2.000] | 0/9 ; 0/9 |

## Paired contrasts

| contrast | factors | heldout combo | heldout value | extra |
|---|---|---:|---:|---|
| conflict_minus_aligned | visibility=hidden, adaptation=fresh_only, support=full, role=alternating | -0.278 [-0.536, -0.021] | -0.244 [-0.388, -0.099] | message_gap: -0.232 [-0.350, -0.113] |
| conflict_minus_aligned | visibility=hidden, adaptation=fresh_only, support=full, role=sender_only | -0.526 [-0.830, -0.223] | -0.353 [-0.489, -0.216] | message_gap: -0.468 [-0.586, -0.349] |
| conflict_minus_aligned | visibility=hidden, adaptation=fresh_only, support=heldout_combo, role=alternating | -0.044 [-0.105, 0.017] | -0.230 [-0.323, -0.138] | message_gap: -0.230 [-0.296, -0.163] |
| conflict_minus_aligned | visibility=hidden, adaptation=fresh_only, support=heldout_combo, role=sender_only | -0.086 [-0.208, 0.037] | -0.328 [-0.405, -0.252] | message_gap: -0.355 [-0.421, -0.288] |
| conflict_minus_aligned | visibility=hidden, adaptation=fresh_only, support=heldout_value, role=alternating | -0.401 [-0.682, -0.120] | 0.005 [-0.026, 0.036] | message_gap: -0.190 [-0.250, -0.129] |
| conflict_minus_aligned | visibility=hidden, adaptation=fresh_only, support=heldout_value, role=sender_only | -0.469 [-0.777, -0.162] | 0.012 [-0.048, 0.072] | message_gap: -0.277 [-0.372, -0.182] |
| conflict_minus_aligned | visibility=hidden, adaptation=coadapt, support=full, role=alternating | -0.383 [-0.565, -0.200] | -0.222 [-0.353, -0.092] | message_gap: -0.190 [-0.290, -0.089] |
| coadapt_minus_fresh_only | population=conflict, visibility=hidden, support=full, role=alternating | 0.060 [0.009, 0.111] | 0.057 [0.004, 0.110] | community_hamming: -0.074 [-0.125, -0.023] |
| conflict_minus_aligned | visibility=hidden, adaptation=coadapt, support=full, role=sender_only | -0.460 [-0.730, -0.189] | -0.309 [-0.457, -0.161] | message_gap: -0.392 [-0.531, -0.254] |
| coadapt_minus_fresh_only | population=conflict, visibility=hidden, support=full, role=sender_only | 0.124 [0.011, 0.237] | 0.135 [0.072, 0.197] | community_hamming: 0.000 [0.000, 0.000] |
| conflict_minus_aligned | visibility=hidden, adaptation=coadapt, support=heldout_combo, role=alternating | 0.000 [0.000, 0.000] | -0.237 [-0.352, -0.122] | message_gap: -0.220 [-0.266, -0.175] |
| coadapt_minus_fresh_only | population=conflict, visibility=hidden, support=heldout_combo, role=alternating | -0.150 [-0.276, -0.023] | 0.069 [0.013, 0.124] | community_hamming: -0.074 [-0.125, -0.023] |
| conflict_minus_aligned | visibility=hidden, adaptation=coadapt, support=heldout_combo, role=sender_only | 0.000 [0.000, 0.000] | -0.256 [-0.352, -0.161] | message_gap: -0.265 [-0.309, -0.221] |
| coadapt_minus_fresh_only | population=conflict, visibility=hidden, support=heldout_combo, role=sender_only | -0.301 [-0.553, -0.049] | 0.157 [0.054, 0.259] | community_hamming: 0.000 [0.000, 0.000] |
| conflict_minus_aligned | visibility=hidden, adaptation=coadapt, support=heldout_value, role=alternating | -0.320 [-0.544, -0.095] | 0.000 [0.000, 0.000] | message_gap: -0.206 [-0.289, -0.122] |
| coadapt_minus_fresh_only | population=conflict, visibility=hidden, support=heldout_value, role=alternating | 0.114 [-0.002, 0.230] | -0.116 [-0.150, -0.082] | community_hamming: -0.099 [-0.166, -0.031] |
| conflict_minus_aligned | visibility=hidden, adaptation=coadapt, support=heldout_value, role=sender_only | -0.285 [-0.566, -0.003] | 0.000 [0.000, 0.000] | message_gap: -0.177 [-0.271, -0.083] |
| coadapt_minus_fresh_only | population=conflict, visibility=hidden, support=heldout_value, role=sender_only | 0.210 [-0.038, 0.458] | -0.231 [-0.299, -0.162] | community_hamming: 0.000 [0.000, 0.000] |
| conflict_minus_aligned | visibility=visible, adaptation=fresh_only, support=full, role=alternating | 0.026 [-0.035, 0.087] | 0.024 [-0.063, 0.112] | message_gap: -0.008 [-0.066, 0.050] |
| visible_minus_hidden | population=conflict, adaptation=fresh_only, support=full, role=alternating | -0.001 [-0.131, 0.128] | -0.017 [-0.105, 0.070] | consistency: -0.519 [-0.599, -0.438] |
| conflict_minus_aligned | visibility=visible, adaptation=fresh_only, support=full, role=sender_only | -0.100 [-0.275, 0.074] | -0.030 [-0.123, 0.063] | message_gap: -0.045 [-0.107, 0.018] |
| visible_minus_hidden | population=conflict, adaptation=fresh_only, support=full, role=sender_only | -0.035 [-0.209, 0.138] | 0.030 [-0.099, 0.158] | consistency: -0.665 [-0.720, -0.609] |
| conflict_minus_aligned | visibility=visible, adaptation=fresh_only, support=heldout_combo, role=alternating | -0.044 [-0.105, 0.017] | 0.039 [-0.027, 0.105] | message_gap: -0.005 [-0.063, 0.053] |
| visible_minus_hidden | population=conflict, adaptation=fresh_only, support=heldout_combo, role=alternating | 0.000 [0.000, 0.000] | -0.008 [-0.080, 0.064] | consistency: -0.486 [-0.560, -0.412] |
| conflict_minus_aligned | visibility=visible, adaptation=fresh_only, support=heldout_combo, role=sender_only | -0.086 [-0.208, 0.037] | -0.034 [-0.112, 0.045] | message_gap: -0.039 [-0.127, 0.050] |
| visible_minus_hidden | population=conflict, adaptation=fresh_only, support=heldout_combo, role=sender_only | 0.000 [0.000, 0.000] | 0.026 [-0.097, 0.149] | consistency: -0.646 [-0.708, -0.585] |
| conflict_minus_aligned | visibility=visible, adaptation=fresh_only, support=heldout_value, role=alternating | 0.032 [-0.055, 0.120] | 0.005 [-0.026, 0.036] | message_gap: 0.011 [-0.046, 0.068] |
| visible_minus_hidden | population=conflict, adaptation=fresh_only, support=heldout_value, role=alternating | 0.076 [-0.006, 0.158] | 0.000 [0.000, 0.000] | consistency: -0.352 [-0.414, -0.290] |
| conflict_minus_aligned | visibility=visible, adaptation=fresh_only, support=heldout_value, role=sender_only | -0.020 [-0.157, 0.118] | 0.012 [-0.048, 0.072] | message_gap: -0.018 [-0.105, 0.068] |
| visible_minus_hidden | population=conflict, adaptation=fresh_only, support=heldout_value, role=sender_only | 0.020 [-0.140, 0.180] | 0.000 [0.000, 0.000] | consistency: -0.475 [-0.518, -0.433] |
| conflict_minus_aligned | visibility=visible, adaptation=coadapt, support=full, role=alternating | -0.006 [-0.086, 0.074] | 0.026 [-0.047, 0.099] | message_gap: 0.004 [-0.061, 0.068] |
| visible_minus_hidden | population=conflict, adaptation=coadapt, support=full, role=alternating | 0.011 [-0.097, 0.119] | -0.024 [-0.118, 0.071] | consistency: -0.535 [-0.618, -0.452] |
| coadapt_minus_fresh_only | population=conflict, visibility=visible, support=full, role=alternating | 0.073 [-0.011, 0.157] | 0.050 [0.005, 0.096] | community_hamming: -0.062 [-0.114, -0.009] |
| conflict_minus_aligned | visibility=visible, adaptation=coadapt, support=full, role=sender_only | -0.139 [-0.357, 0.079] | -0.048 [-0.154, 0.057] | message_gap: -0.064 [-0.110, -0.017] |
| visible_minus_hidden | population=conflict, adaptation=coadapt, support=full, role=sender_only | -0.093 [-0.243, 0.056] | -0.033 [-0.213, 0.148] | consistency: -0.675 [-0.731, -0.619] |
| coadapt_minus_fresh_only | population=conflict, visibility=visible, support=full, role=sender_only | 0.066 [-0.000, 0.133] | 0.072 [0.018, 0.126] | community_hamming: 0.000 [0.000, 0.000] |
| conflict_minus_aligned | visibility=visible, adaptation=coadapt, support=heldout_combo, role=alternating | 0.000 [0.000, 0.000] | 0.005 [-0.074, 0.084] | message_gap: -0.019 [-0.083, 0.045] |
| visible_minus_hidden | population=conflict, adaptation=coadapt, support=heldout_combo, role=alternating | 0.000 [0.000, 0.000] | -0.031 [-0.120, 0.059] | consistency: -0.484 [-0.558, -0.409] |
| coadapt_minus_fresh_only | population=conflict, visibility=visible, support=heldout_combo, role=alternating | -0.150 [-0.276, -0.023] | 0.046 [-0.018, 0.110] | community_hamming: -0.049 [-0.088, -0.011] |
| conflict_minus_aligned | visibility=visible, adaptation=coadapt, support=heldout_combo, role=sender_only | 0.000 [0.000, 0.000] | -0.027 [-0.110, 0.056] | message_gap: -0.053 [-0.128, 0.022] |
| visible_minus_hidden | population=conflict, adaptation=coadapt, support=heldout_combo, role=sender_only | 0.000 [0.000, 0.000] | 0.031 [-0.077, 0.138] | consistency: -0.613 [-0.678, -0.548] |
| coadapt_minus_fresh_only | population=conflict, visibility=visible, support=heldout_combo, role=sender_only | -0.301 [-0.553, -0.049] | 0.161 [0.049, 0.274] | community_hamming: 0.000 [0.000, 0.000] |
| conflict_minus_aligned | visibility=visible, adaptation=coadapt, support=heldout_value, role=alternating | -0.048 [-0.122, 0.026] | 0.000 [0.000, 0.000] | message_gap: -0.025 [-0.089, 0.039] |
| visible_minus_hidden | population=conflict, adaptation=coadapt, support=heldout_value, role=alternating | 0.058 [-0.027, 0.144] | 0.000 [0.000, 0.000] | consistency: -0.372 [-0.429, -0.316] |
| coadapt_minus_fresh_only | population=conflict, visibility=visible, support=heldout_value, role=alternating | 0.097 [-0.028, 0.222] | -0.116 [-0.150, -0.082] | community_hamming: -0.086 [-0.147, -0.026] |
| conflict_minus_aligned | visibility=visible, adaptation=coadapt, support=heldout_value, role=sender_only | -0.114 [-0.278, 0.050] | 0.000 [0.000, 0.000] | message_gap: -0.046 [-0.144, 0.051] |
| visible_minus_hidden | population=conflict, adaptation=coadapt, support=heldout_value, role=sender_only | -0.005 [-0.238, 0.227] | 0.000 [0.000, 0.000] | consistency: -0.469 [-0.495, -0.443] |
| coadapt_minus_fresh_only | population=conflict, visibility=visible, support=heldout_value, role=sender_only | 0.185 [-0.014, 0.383] | -0.231 [-0.299, -0.162] | community_hamming: 0.000 [0.000, 0.000] |
