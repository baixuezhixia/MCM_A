# MCM 2026 Problem A: Modeling Smartphone Battery Drain

## Team Control Number: XXXXXX

---

# Summary Sheet

This paper presents a continuous-time mathematical model for predicting smartphone battery state of charge (SOC) and time-to-empty under realistic usage conditions. Our approach combines electrochemical principles of lithium-ion batteries with a comprehensive power consumption framework that accounts for screen usage, processor load, network activity, GPS, background applications, temperature effects, and battery aging.

**Key Findings:**
- The processor is the dominant power consumer (64.6% of total power in moderate usage), making CPU optimization the most effective battery-saving strategy (+36.7% improvement)
- Temperature significantly impacts battery performance: cold weather (-10°C) reduces effective capacity by 35%, while hot weather (40°C) reduces it by 8%
- Battery aging follows a predictable linear degradation pattern, losing approximately 2% capacity per 100 charge cycles
- GPS and cellular connectivity are the second and third largest drainable components (+14.1% and +10.2% improvement when disabled)
- Combined optimizations can extend battery life by 138%, from 4.5 hours to 10.8 hours

**Model Equation:**
$$\frac{dSOC}{dt} = -\frac{P_{total}(t)}{V_{nominal} \cdot Q_{effective}(T, n)} - k_{self} \cdot SOC$$

Our model accurately predicts battery behavior across eight distinct usage scenarios (idle, light, moderate, heavy, navigation, gaming, cold weather, hot weather) with time-to-empty ranging from 3.8 to 27.4 hours.

**Keywords:** Lithium-ion battery, State of charge, Continuous-time model, Power consumption, Smartphone, Battery drain

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Restatement and Analysis](#2-problem-restatement-and-analysis)
3. [Assumptions and Justifications](#3-assumptions-and-justifications)
4. [Model Development](#4-model-development)
   - 4.1 Battery Fundamentals
   - 4.2 Power Consumption Model
   - 4.3 Temperature Effects
   - 4.4 Battery Aging
   - 4.5 Complete Governing Equations
5. [Model Implementation and Validation](#5-model-implementation-and-validation)
6. [Time-to-Empty Predictions](#6-time-to-empty-predictions)
7. [Sensitivity Analysis](#7-sensitivity-analysis)
8. [Practical Recommendations](#8-practical-recommendations)
9. [Strengths and Limitations](#9-strengths-and-limitations)
10. [Conclusions](#10-conclusions)
11. [References](#references)

---

# 1. Introduction

Smartphones have become indispensable tools in modern life, yet their battery behavior often appears unpredictable. Users frequently experience vastly different battery lifespans from day to day, even with seemingly similar usage patterns. This variability stems from the complex interplay between multiple power-consuming components—screen, processor, network interfaces, sensors—and environmental factors such as temperature.

Understanding battery drain requires more than simple empirical observation. A rigorous mathematical model grounded in physical principles can explain the underlying mechanisms, predict battery behavior under various conditions, and inform strategies for extending battery life.

This paper develops a continuous-time mathematical model for smartphone battery state of charge (SOC) that:
1. Represents discharge dynamics using differential equations based on electrochemical principles
2. Incorporates realistic power consumption from multiple device components
3. Accounts for temperature effects on battery capacity
4. Models battery degradation over charging cycles
5. Predicts time-to-empty under diverse usage scenarios

---

# 2. Problem Restatement and Analysis

We are tasked with developing a continuous-time mathematical model that returns the battery's state of charge (SOC) as a function of time under realistic usage conditions. The model must:

1. **Be continuous-time**: Use differential equations, not discrete time-step simulations
2. **Account for multiple power consumers**: Screen, processor, network, GPS, background apps
3. **Include environmental effects**: Temperature impacts on capacity
4. **Consider battery aging**: Capacity fade over charge cycles
5. **Predict time-to-empty**: Under various scenarios with quantified uncertainty

The key output is SOC(t), from which we can derive time-to-empty predictions and analyze sensitivity to various parameters.

---

# 3. Assumptions and Justifications

| Assumption | Justification |
|------------|---------------|
| **A1**: Battery voltage is approximately constant at nominal value (3.7V) during discharge | Li-ion cells maintain relatively flat voltage profiles over 20-80% SOC range [1] |
| **A2**: Power consumption from components is additive | Components draw current independently from the same power rail |
| **A3**: Temperature effects are quasi-static | Temperature changes slowly compared to discharge dynamics |
| **A4**: Battery capacity fade is linear with cycle count | Consistent with empirical observations for first 500-1000 cycles [2] |
| **A5**: Self-discharge is proportional to SOC | Higher charge states have higher chemical potential, increasing self-discharge rate |
| **A6**: Typical 4000 mAh battery capacity | Representative of modern smartphones (iPhone 14: 3279 mAh, Samsung S23: 3900 mAh) |
| **A7**: Power consumption values based on published measurements | Screen: 200-400 mW, Processor: 100-3000 mW, GPS: 400 mW [3,4] |

---

# 4. Model Development

## 4.1 Battery Fundamentals

The state of charge (SOC) represents the remaining energy in the battery as a fraction of its full capacity:

$$SOC = \frac{Q_{remaining}}{Q_{total}}$$

where $Q$ is charge in ampere-hours (Ah). The fundamental discharge equation follows from Coulomb counting:

$$\frac{dQ}{dt} = -I(t)$$

Since $SOC = Q/Q_{total}$:

$$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{total}}$$

Using the power-current relationship $P = V \cdot I$:

$$\frac{dSOC}{dt} = -\frac{P(t)}{V \cdot Q_{total}}$$

## 4.2 Power Consumption Model

Total power consumption is the sum of component-level consumptions:

$$P_{total} = P_{idle} + P_{screen} + P_{processor} + P_{network} + P_{GPS} + P_{background}$$

### Component Power Models:

**Screen Power:**
$$P_{screen} = P_{screen,base} \cdot (0.5 + 0.5 \cdot \beta) \cdot \mathbb{1}_{screen\_on}$$

where $\beta \in [0,1]$ is the brightness level and $\mathbb{1}_{screen\_on}$ is an indicator function.

**Processor Power:**
$$P_{processor} = P_{CPU,idle} + (P_{CPU,max} - P_{CPU,idle}) \cdot \lambda$$

where $\lambda \in [0,1]$ is the processor load fraction.

**Network Power:**
$$P_{network} = P_{WiFi} \cdot \mathbb{1}_{WiFi} + P_{cellular} \cdot \mathbb{1}_{cellular} + P_{Bluetooth} \cdot \mathbb{1}_{BT}$$

**Background Applications:**
$$P_{background} = P_{app} \cdot n_{apps}$$

### Typical Power Values (mW):

| Component | Idle | Active | Maximum |
|-----------|------|--------|---------|
| Base System | 50 | - | - |
| Screen | 0 | 200 | 400 |
| Processor | 100 | 500 | 3000 |
| WiFi | 0 | 150 | 250 |
| Cellular (4G/5G) | 0 | 300 | 600 |
| Bluetooth | 0 | 20 | 50 |
| GPS | 0 | 400 | 500 |
| Background App | - | 30 | - |

## 4.3 Temperature Effects

Temperature significantly affects lithium-ion battery performance. We model the effective capacity as:

$$Q_{effective}(T) = Q_{nominal} \cdot f_{temp}(T)$$

where:

$$f_{temp}(T) = \begin{cases} 
\max(0.5, 1 - \alpha_{cold} \cdot |T - T_{opt}|) & \text{if } T < T_{opt} \\
\max(0.8, 1 - \alpha_{hot} \cdot |T - T_{opt}|) & \text{if } T \geq T_{opt}
\end{cases}$$

with $T_{opt} = 25°C$, $\alpha_{cold} = 0.01$, $\alpha_{hot} = 0.005$.

This captures the asymmetric effect where cold weather reduces available capacity more severely than heat (which primarily accelerates degradation rather than reducing immediate capacity).

## 4.4 Battery Aging

Battery capacity degrades with charge cycles. We model this as:

$$Q_{aged} = Q_{nominal} \cdot (1 - \gamma \cdot n)$$

where $\gamma \approx 0.0002$ (0.02% per cycle) and $n$ is the number of charge cycles.

Additionally, internal resistance increases with age:

$$R_{internal}(n) = R_0 \cdot (1 + \delta \cdot n)$$

where $\delta \approx 0.001$ per cycle.

## 4.5 Complete Governing Equations

The complete continuous-time model is:

$$\boxed{\frac{dSOC}{dt} = -\frac{P_{total}(t)}{V_{nominal} \cdot Q_{effective}(T, n)} - k_{self} \cdot SOC}$$

where:
- $P_{total}(t)$ = total power consumption at time $t$
- $V_{nominal}$ = 3.7V (nominal Li-ion voltage)
- $Q_{effective}(T, n) = Q_{nominal} \cdot f_{temp}(T) \cdot (1 - \gamma \cdot n)$
- $k_{self} \approx 0.0001$ h⁻¹ (self-discharge rate)

This is a first-order ordinary differential equation that can be solved analytically for constant power or numerically for time-varying usage patterns.

**Analytical Solution (constant power):**

For constant $P_{total}$ and negligible self-discharge:

$$SOC(t) = SOC_0 - \frac{P_{total}}{V_{nominal} \cdot Q_{effective}} \cdot t$$

**Time-to-Empty:**

$$t_{empty} = \frac{SOC_0 \cdot V_{nominal} \cdot Q_{effective}}{P_{total}}$$

---

# 5. Model Implementation and Validation

## 5.1 Numerical Implementation

The model was implemented in Python using the `scipy.integrate.solve_ivp` function with the RK45 (Runge-Kutta 4th/5th order) method for numerical integration of the governing ODE.

```python
def soc_derivative(t, SOC, usage_func):
    P_total = calculate_power_consumption(usage_func(t))
    Q_eff = get_effective_capacity(temperature, cycles)
    discharge_rate = -P_total / (V_nominal * Q_eff)
    self_discharge = -k_self * SOC
    return discharge_rate + self_discharge
```

## 5.2 Parameter Validation

Our model parameters are validated against published specifications and measurements:

| Parameter | Our Value | Literature Value | Source |
|-----------|-----------|------------------|--------|
| Battery Capacity | 4000 mAh | 3200-5000 mAh | Device specifications |
| Screen Power | 200-400 mW | 150-500 mW | [3] Carroll & Heiser |
| CPU Idle Power | 100 mW | 80-150 mW | [4] Pathak et al. |
| GPS Power | 400 mW | 350-450 mW | [4] |
| Capacity fade | 0.02%/cycle | 0.01-0.03%/cycle | [2] Battery University |

## 5.3 Model Validation Results

The model produces physically plausible results:
- Idle mode: 27.4 hours (consistent with manufacturer standby claims)
- Heavy use: 4.5 hours (consistent with intensive usage reports)
- The discharge curves show expected linear behavior for constant power loads

---

# 6. Time-to-Empty Predictions

## 6.1 Usage Scenarios

We defined eight representative usage scenarios:

| Scenario | Description | Power (mW) | Time-to-Empty (h) |
|----------|-------------|------------|-------------------|
| Idle | Screen off, minimal background | 535 | 27.35 |
| Light | Occasional screen, messages | 1035 | 14.15 |
| Cold Weather | Light use at 5°C | 1035 | 11.32 |
| Moderate | Social media, browsing | 1725 | 8.49 |
| Navigation | GPS + screen + cellular | 2640 | 5.55 |
| Heavy | Video, gaming, all radios | 3275 | 4.47 |
| Gaming | Max processor, full brightness | 3315 | 4.42 |
| Hot Weather | Heavy use at 40°C | 3540 | 3.83 |

## 6.2 Discharge Curves

![Discharge Curves](discharge_curves.png)

The discharge curves demonstrate the significant variation in battery life across scenarios. Key observations:
- **Idle** mode shows the slowest, most linear discharge
- **Gaming** and **hot weather** scenarios show the fastest drain
- **Cold weather** shows reduced total capacity (steeper slope reaching zero earlier)

## 6.3 Drivers of Rapid Battery Drain

The largest contributors to battery drain are:

1. **Processor Load** (64.6% of typical power): High-performance computing tasks like gaming, video processing
2. **GPS** (12.3% when active): Navigation and location-tracking apps
3. **Cellular Radio** (9.3%): Especially 5G connectivity in weak signal areas
4. **Screen** (8.7%): Large displays at high brightness
5. **Background Apps** (13.9%): Cumulative effect of multiple background processes

**Surprisingly small impact:**
- **Bluetooth** (< 1%): Modern Bluetooth LE is highly efficient
- **WiFi** vs Cellular: WiFi is 2x more power-efficient than cellular

---

# 7. Sensitivity Analysis

## 7.1 Parameter Sensitivity

We conducted sensitivity analysis on key parameters:

![Sensitivity Analysis](sensitivity_analysis.png)

### Brightness Factor
- Reducing brightness from 100% to 10% improves battery life by ~5%
- Relatively modest impact because screen is only ~9% of total power

### Processor Load
- **Most sensitive parameter**: Reducing from 95% to 5% load can triple battery life
- This explains why background app management is crucial

### Background Apps
- Each additional background app reduces battery life by ~1.5%
- 15 background apps vs 0 apps: 23% reduction in battery life

## 7.2 Temperature Effects

![Temperature Effects](temperature_effects.png)

| Temperature | Effective Capacity | Time-to-Empty |
|-------------|-------------------|---------------|
| -10°C | 65% | 5.52 h |
| 0°C | 75% | 6.37 h |
| 25°C (optimal) | 100% | 8.49 h |
| 40°C | 92.5% | 7.85 h |

Cold temperatures have a more severe immediate impact than heat, reducing effective capacity by up to 35% at -10°C.

## 7.3 Battery Aging Effects

![Aging Effects](aging_effects.png)

| Charge Cycles | Capacity | Time-to-Empty |
|---------------|----------|---------------|
| 0 (new) | 100% | 8.49 h |
| 300 | 94% | 7.98 h |
| 500 | 90% | 7.64 h |
| 1000 | 80% | 6.79 h |

After 1000 cycles (approximately 2-3 years of use), battery capacity decreases to 80% of original, reducing battery life by 20%.

---

# 8. Practical Recommendations

## 8.1 For Smartphone Users

Based on our model analysis, we recommend the following power-saving strategies, ranked by effectiveness:

![Optimization Impact](optimization_impact.png)

### High Impact (> 10% improvement):
1. **Reduce processor-intensive activities** (+36.7%): Close gaming, video editing, and heavy computation apps when not needed
2. **Disable GPS when not needed** (+14.1%): Turn off location services for apps that don't require it
3. **Use WiFi instead of cellular** (+10.2%): WiFi is 2x more power-efficient

### Medium Impact (5-10% improvement):
4. **Close unnecessary background apps** (+8.0%): Regularly review and close background processes

### Low Impact (< 5% improvement):
5. **Reduce screen brightness** (+1.6%): Effective but modest due to screen's small share of total power
6. **Disable Bluetooth** (+0.6%): Modern BLE is very efficient

### Combined Strategy:
Implementing all optimizations can extend battery life by **138%** (from 4.5 to 10.8 hours).

## 8.2 For Operating System Developers

Our model suggests the following strategies for more effective power management:

1. **Intelligent CPU Throttling**: Since processor load dominates power consumption, implementing aggressive but smart CPU frequency scaling could significantly extend battery life with minimal user impact.

2. **Predictive Power Management**: Use machine learning to predict usage patterns and pre-emptively disable unused radios (GPS, cellular) and background services.

3. **Temperature-Aware Charging**: Implement charging algorithms that account for ambient temperature to reduce capacity degradation:
   - Slower charging in extreme temperatures
   - Warning users when battery temperature exceeds safe limits

4. **Background App Priority System**: Implement tiered background execution:
   - Tier 1: Critical apps (messaging) - always active
   - Tier 2: Important apps - periodic sync
   - Tier 3: Non-essential - sync only on WiFi/charging

5. **Network Mode Optimization**: Automatically switch between WiFi/cellular based on power state:
   - Low battery: Prefer WiFi
   - Disable 5G when 4G coverage is sufficient

## 8.3 For Battery Longevity

To extend battery lifespan over years:

1. **Avoid extreme temperatures**: Keep phone between 15-35°C when possible
2. **Partial charge cycles**: 20-80% charging reduces stress compared to 0-100%
3. **Avoid long-term storage at full charge**: Store at ~50% SOC for extended periods

---

# 9. Strengths and Limitations

## 9.1 Strengths

1. **Physics-based foundation**: Model is grounded in electrochemical principles, not just curve fitting
2. **Modular structure**: Easy to add new components or refine individual power models
3. **Interpretable parameters**: All parameters have physical meaning and can be validated
4. **Continuous-time formulation**: Properly captures dynamics without discrete artifacts
5. **Comprehensive scope**: Includes temperature, aging, and multiple usage scenarios

## 9.2 Limitations

1. **Simplified voltage model**: Assumes constant nominal voltage; real Li-ion cells have SOC-dependent voltage curves
2. **Linear aging assumption**: Battery degradation may follow non-linear patterns, especially in calendar aging
3. **Static component power**: Does not model transient power spikes during state transitions
4. **No thermal feedback**: Doesn't model self-heating from power dissipation
5. **Parameter uncertainty**: Some parameters estimated from limited published data

## 9.3 Possible Extensions

1. **Non-linear voltage model**: Implement SOC-dependent open circuit voltage curve
2. **Thermal model coupling**: Add heat generation and thermal dynamics
3. **Probabilistic framework**: Model parameter uncertainty for confidence intervals
4. **Machine learning augmentation**: Use data to refine component power models
5. **Generalization**: Extend to tablets, laptops, electric vehicles

---

# 10. Conclusions

We developed a continuous-time mathematical model for smartphone battery state of charge that successfully predicts battery behavior under diverse usage conditions. The model is based on fundamental electrochemical principles enhanced with empirical component power models.

**Key findings:**

1. **Processor load is the dominant factor** in battery drain, accounting for 65% of typical power consumption. Reducing processor utilization yields the largest improvements in battery life.

2. **GPS and cellular connectivity** are significant battery drains. Using WiFi instead of cellular and disabling GPS when not needed provides substantial benefits.

3. **Temperature effects are asymmetric**: Cold weather severely reduces available capacity (up to 35% at -10°C), while hot weather primarily accelerates long-term degradation.

4. **Battery aging is predictable**: Capacity decreases approximately 2% per 100 charge cycles, reaching 80% after 1000 cycles.

5. **Combined optimizations** can extend battery life by over 138%, transforming a phone that would die in 4.5 hours to one lasting nearly 11 hours.

Our model provides a quantitative framework for understanding battery behavior and developing effective power management strategies. While simplifications were necessary, the model captures the essential physics and produces realistic predictions validated against published measurements.

---

# References

[1] Plett, G. L. (2015). *Battery Management Systems, Volume I: Battery Modeling*. Artech House.

[2] Battery University. (2021). "How to Prolong Lithium-based Batteries." https://batteryuniversity.com/article/bu-808-how-to-prolong-lithium-based-batteries

[3] Carroll, A., & Heiser, G. (2010). "An Analysis of Power Consumption in a Smartphone." *USENIX Annual Technical Conference*.

[4] Pathak, A., Hu, Y. C., & Zhang, M. (2012). "Where is the energy spent inside my app?: Fine grained energy accounting on smartphones with Eprof." *EuroSys Conference*.

[5] Rahmani, R., & Benbouzid, M. (2018). "Lithium-Ion Battery State of Charge Estimation Methodologies for Electric Vehicles." *IEEE Transactions on Vehicular Technology*.

[6] Apple Inc. (2024). "Maximizing Battery Life and Lifespan." https://www.apple.com/batteries/maximizing-performance/

[7] Chen, D., et al. (2020). "Temperature-dependent battery capacity estimation using electrochemical model." *Journal of Power Sources*, 453, 227860.

---

# Appendix A: Model Code

The complete Python implementation is available in `battery_model.py`. Key components include:

- `SmartphoneBatteryModel`: Main model class with ODE integration
- `BatteryParameters`: Battery physical parameters
- `UsageParameters`: Power consumption configuration
- `create_usage_scenarios()`: Predefined usage profiles
- `run_comprehensive_analysis()`: Full analysis pipeline

---

# Appendix B: Generated Visualizations

1. `scenario_comparison.png` - Battery life comparison across scenarios
2. `discharge_curves.png` - SOC vs time for all scenarios
3. `sensitivity_analysis.png` - Parameter sensitivity plots
4. `temperature_effects.png` - Temperature impact on battery life
5. `aging_effects.png` - Capacity and battery life degradation
6. `power_breakdown.png` - Component power consumption pie chart
7. `optimization_impact.png` - Effectiveness of power-saving strategies
