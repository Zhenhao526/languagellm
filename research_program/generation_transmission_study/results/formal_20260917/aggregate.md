# Generation transmission compact aggregation

- parent runs: 8
- child runs: 48
- split: heldout

| condition | role | natural | silent | scrambled | live−silent | live−scrambled | learning AUC | child semantic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `worker_live` | worker | 0.800 | 0.202 | 0.284 | 0.598 | 0.516 | 0.750 | 1.000 |
| `worker_silent` | worker | 0.202 | 0.202 | 0.284 | 0.598 | 0.516 | 0.202 | 0.312 |
| `worker_scrambled` | worker | 0.284 | 0.202 | 0.284 | 0.598 | 0.516 | 0.287 | 0.500 |
| `sender_live` | sender | 0.800 | 0.316 | 0.347 | 0.484 | 0.453 | 0.760 | 1.000 |
| `sender_silent` | sender | 0.316 | 0.316 | 0.347 | 0.484 | 0.453 | 0.316 | 0.531 |
| `sender_scrambled` | sender | 0.347 | 0.316 | 0.347 | 0.484 | 0.453 | 0.371 | 0.555 |

The paired effects compare the same seed's child run after replacing the same parent component. AUC is the trapezoidal heldout natural-return curve divided by the 3000-update horizon.

## Paired effects

| role | effect | mean | 95% t CI |
|---|---|---:|---:|
| worker | live-minus-silent | 0.5982 | [0.5401, 0.6563] |
| worker | live-minus-scrambled | 0.5159 | [0.4795, 0.5523] |
| worker | curve-auc-live-minus-silent | 0.5484 | [0.4951, 0.6016] |
| worker | child-minus-parent | 0.0000 | [-0.0000, 0.0000] |
| sender | live-minus-silent | 0.4837 | [0.3285, 0.6388] |
| sender | live-minus-scrambled | 0.4529 | [0.1673, 0.7384] |
| sender | curve-auc-live-minus-silent | 0.4434 | [0.3012, 0.5856] |
| sender | child-minus-parent | -0.0000 | [-0.0000, 0.0000] |
