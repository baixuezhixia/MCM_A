# MCM 2026 Problem A: Modeling Smartphone Battery Drain

## Solution Overview

This repository contains a complete solution for the 2026 Mathematical Contest in Modeling (MCM) Problem A: Modeling Smartphone Battery Drain.

## Problem Statement

Develop a continuous-time mathematical model of a smartphone's battery that returns the state of charge (SOC) as a function of time under realistic usage conditions, to predict the remaining time-to-empty under different conditions.

## Solution Components

### Files

| File | Description |
|------|-------------|
| `battery_model.py` | Python implementation of the continuous-time battery model |
| `MCM_2026_Problem_A_Solution.md` | Complete written solution (25-page MCM paper format) |
| `2026_MCM_Problem_A.pdf` | Original problem statement |

### Generated Visualizations

After running the model, the following visualizations are generated:

- `scenario_comparison.png` - Battery life comparison across usage scenarios
- `discharge_curves.png` - SOC vs time discharge curves
- `sensitivity_analysis.png` - Parameter sensitivity analysis
- `temperature_effects.png` - Temperature impact on battery life
- `aging_effects.png` - Battery degradation over charge cycles
- `power_breakdown.png` - Component power consumption breakdown
- `optimization_impact.png` - Power-saving strategy effectiveness

## Mathematical Model

### Governing Equation

```
dSOC/dt = -P_total(t) / (V_nominal × Q_effective(T, n)) - k_self × SOC
```

Where:
- `SOC` = State of charge (0 to 1)
- `P_total(t)` = Total power consumption at time t
- `V_nominal` = 3.7V (nominal Li-ion voltage)
- `Q_effective` = Effective battery capacity (temperature and age dependent)
- `k_self` = Self-discharge rate (~0.01%/hour)

### Key Features

1. **Continuous-time formulation** using differential equations
2. **Multi-component power model** (screen, processor, network, GPS, background apps)
3. **Temperature effects** on battery capacity
4. **Battery aging** and capacity fade modeling
5. **Multiple usage scenarios** for realistic predictions

## Usage

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Running the Analysis

```bash
python battery_model.py
```

This will:
1. Run comprehensive battery drain analysis for 8 usage scenarios
2. Perform sensitivity analysis on key parameters
3. Analyze temperature and aging effects
4. Generate recommendations for power saving
5. Create all visualization files

### Sample Output

```
======================================================================
MCM 2026 Problem A: Smartphone Battery Drain Model Analysis
======================================================================

PART 1: Time-to-Empty Predictions for Different Scenarios
----------------------------------------------------------
IDLE           : Power =   535.0 mW, Time-to-empty = 27.35 hours
LIGHT          : Power =  1035.0 mW, Time-to-empty = 14.15 hours
MODERATE       : Power =  1725.0 mW, Time-to-empty =  8.49 hours
GAMING         : Power =  3315.0 mW, Time-to-empty =  4.42 hours
...
```

### Using the Model Programmatically

```python
from battery_model import SmartphoneBatteryModel, UsageParameters

# Create model with default 4000 mAh battery
model = SmartphoneBatteryModel()

# Configure usage scenario
model.usage = UsageParameters(
    screen_on=True,
    brightness_factor=0.5,
    processor_load=0.3,
    wifi_active=True,
    gps_active=False,
    n_background_apps=5
)

# Simulate discharge
result = model.simulate(t_span=(0, 24), SOC_initial=1.0)

# Get time-to-empty prediction
tte = model.predict_time_to_empty()
print(f"Time to empty: {tte:.2f} hours")
```

## Key Findings

1. **Processor is the dominant power consumer** (64.6% of typical power)
2. **Reducing processor load** provides the largest battery improvement (+36.7%)
3. **GPS and cellular** are major battery drains when active
4. **Cold weather** (-10°C) reduces capacity by 35%
5. **Combined optimizations** can extend battery life by 138%

## Model Validation

Parameters are validated against published specifications:

| Component | Model Value | Literature Range | Source |
|-----------|-------------|------------------|--------|
| Screen Power | 200-400 mW | 150-500 mW | Carroll & Heiser (2010) |
| CPU Power | 100-3000 mW | 80-3500 mW | Pathak et al. (2012) |
| GPS Power | 400 mW | 350-450 mW | Device specifications |
| Capacity fade | 0.02%/cycle | 0.01-0.03%/cycle | Battery University |

## License

This solution is provided for educational purposes as part of the MCM 2026 competition.

## Authors

Team Control Number: XXXXXX
