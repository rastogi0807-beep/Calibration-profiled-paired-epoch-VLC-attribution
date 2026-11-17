# Calibration-profiled-paired-epoch-VLC-attribution
This study investigates emitter–channel ambiguity in VLC links using calibrated reference-photodiode measurements and paired-epoch estimation. Results demonstrate improved attribution and mismatch detection, while multistart and independent forward-model audits reveal limitations in identifiability &amp; reliability under structural mismatch
## Model and units

The source is one effective low-pass pole, and propagation is a normalized LOS impulse plus an exponential diffuse tail. Both spectra use positive real gains. A shielded transmitter-local reference measures the source through a known receiver pole. Bandwidths and frequencies are stored in MHz; their ratios are dimensionless. Log bandwidths mean log(b / 1 MHz) and log(p / 1 MHz). Delays are supplied in ns and converted when forming phase.

Twelve complex probe values are used per receiver, including an idealized limiting zero-frequency normalization point. Independent circular Gaussian noise is a benchmark assumption, including at that point. A real instrument needs measured near-DC and complex-noise covariance. Exact baseline commissioning, calibrated phase and supplied noise variances are also assumed.

Four model-conditional labels are no change, emitter change, propagation change and mixed change. A reference-pole calibration error and an omitted diffuse delay are **outside** that label family. Raw result files retain diagnostic label columns for reproducibility; **do not interpret their four-class accuracy as causal detection accuracy**. The manuscript correctly reports residual/acceptance and physical-parameter error instead.
