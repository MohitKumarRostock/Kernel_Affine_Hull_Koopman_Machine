# Table 9: Centered multi-horizon Van der Pol unseen-trajectory generalization

| Training trajectories | Robust mean R2 | Clean mean R2 | Reference mean R2 | Delta E clean-robust | Delta E reference-robust |
|---:|---:|---:|---:|---:|---:|
| 1 | $0.8745\pm0.0055$ | $0.8768\pm0.0134$ | $0.8795\pm0.0081$ | $-0.0022\pm0.0055$ | $-0.0049\pm0.0077$ |
| 2 | $0.8876\pm0.0050$ | $0.8840\pm0.0047$ | $0.8889\pm0.0125$ | $+0.0036\pm0.0030$ | $-0.0013\pm0.0064$ |
| 3 | $0.8930\pm0.0062$ | $0.8936\pm0.0065$ | $0.8982\pm0.0044$ | $-0.0005\pm0.0033$ | $-0.0052\pm0.0051$ |
| 5 | $0.8985\pm0.0029$ | $0.8861\pm0.0027$ | $0.8958\pm0.0070$ | $+0.0124\pm0.0024$ | $+0.0028\pm0.0036$ |
| 8 | $0.9021\pm0.0100$ | $0.8891\pm0.0045$ | $0.9077\pm0.0030$ | $+0.0130\pm0.0032$ | $-0.0056\pm0.0070$ |

Mean R2 is the mean association R2 over horizons {1, 10, 50, 100, 200}; configuration entries are mean +/- SD over three replicate training-seed blocks.
Paired Delta E columns are mean +/- SE over matched replicates, with Delta E = centered error(comparator) - centered error(robust). Positive values therefore indicate lower centered error for robust_tuned.
