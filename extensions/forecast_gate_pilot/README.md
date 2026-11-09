# Exploratory training of local forecast gates

This pilot compares forecast-trained KAHM gates with fixed KAHM and equally trained geometric controls on nine simulated tasks. Each of three data-generating laws has three independent realizations, so the results are exploratory.

The [scientific guide](SCIENTIFIC_GUIDE.md) describes the feature maps, data partitions, optimization budget, comparisons, and reproduction commands. [pilot_results](pilot_results/) contains inputs, selected forecasts, candidate losses, optimizer records, the execution source snapshot, and the numerical audit.

The implementation fits gate temperatures on partition B, selects predictor settings on C, and evaluates fixed forecasts on D. Training and selection use no D targets. The comparison asks whether the complete KAHM procedure improves prediction over equally trained alternatives under this allocation.

The [online module guide](INTEGRATION.md) documents a separate fixed-feature bounded-window ridge component. It does not implement online updates to the pilot's geometry. The [direct reconstruction study](../direct_reconstruction/README.md) evaluates a separate intervention on reconstruction bandwidths and gates.
