# Historical pilot integration

This directory preserves the completed nine-task exploratory gate-temperature
pilot from the research continuation workbench, including its original freeze,
source snapshot, predictions, limitations and numerical checks. Its original
README records the source-code access situation at the time of execution.
The upstream repository was subsequently recovered for the separate
`extensions/direct_reconstruction` experiment. No historical output or frozen
pilot file was revised to make the pilot appear to use that recovered code.

The pilot uses independently implemented known-law generators and three
seeds per law, a 160-evaluation budget, and gate temperatures only. The next
study uses the inherited generators, twelve fresh seeds per law, a public
pre-execution checkpoint, a 256-evaluation ceiling for both trainable modes,
and a reconstruction/shape intervention. Their errors and costs must not be
pooled. The pilot is exploratory, not part of the new simultaneous family.

The preserved bounded-window ridge and cached-reconstruction modules are
numerical building blocks. They have not been evaluated as an online KAHM
forecasting system; no streaming accuracy, update-latency, or memory claim
follows from their algebraic checks.
