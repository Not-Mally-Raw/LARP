# Stage 2 Calibration Report — Planet 1010375142

## 1. Physical Calibration Sequence Verified
- Cast raw uint16 to float64 before any differencing (prevented modular arithmetic underflow).
- Restored physical electrons via ADC gain (0.4369 e-/ADU) and offset (-1000.0 ADU).
- Subtracted master dark frame in physical electrons.
- Applied paired Correlated Double Sampling (CDS: S[1::2] - S[0::2]), achieving dramatic reset noise rejection.
- Applied normalized flat field with throughput guarding (>0.1 median).
- Cleaned dead/hot pixel channels using spatial median replacement.

## 2. Key Telemetry & Noise Floor
- FGS1 Median Flux: 2,648.3 e-/read
- Out-of-transit Scatter: 166.67 electrons
- Theoretical Poisson Limit: 51.47 electrons
- Diagnostic Ratio: 3.238
- Lag-1 Autocorrelation: 0.0029
