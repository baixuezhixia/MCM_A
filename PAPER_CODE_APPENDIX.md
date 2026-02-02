# 论文附录：核心代码片段
# Paper Appendix: Core Code Snippets

本文档包含MCM 2026问题A（智能手机电池SOC连续时间建模）的核心代码片段，适合放入论文附录中。

---

## 1. 核心微分方程模型 (Core ODE Model)

### 1.1 SOC微分方程
```python
def dSOC_dt(self, t: float, SOC: float, usage_func=None) -> float:
    """
    Core differential equation (energy-based):
    dSOC(t)/dt = -P_total(t) / E_total
    
    SOC Definition: SOC = E_remaining / E_total (能量比值)
    
    Parameters:
        t : Time in hours
        SOC : State of charge (0 to 1)
        usage_func : Returns UsageState for given time t
    
    Returns:
        Rate of change of SOC (per hour)
    """
    usage = usage_func(t) if usage_func else UsageState()
    
    # Total power in Watts
    P_total_W = self.calculate_P_total(usage) / 1000.0
    
    # Effective energy capacity (Wh): E = V_nominal × Q
    C_eff_Wh = self.get_effective_capacity_Wh(usage.temperature_C)
    
    # Core equation: dSOC/dt = -P_total / E_total
    return -P_total_W / C_eff_Wh
```

### 1.2 ODE数值求解
```python
def simulate(self, t_span, SOC_initial=1.0, usage_func=None):
    """
    Solve the ODE to simulate battery discharge.
    
    Parameters:
        t_span : (t_start, t_end) in hours
        SOC_initial : Initial SOC (0 to 1)
        usage_func : Time-varying usage function
    """
    def ode(t, y):
        return [self.dSOC_dt(t, y[0], usage_func)]
    
    # Terminal event: shutdown at 5% SOC
    def shutdown(t, y):
        return y[0] - 0.05
    shutdown.terminal = True
    shutdown.direction = -1
    
    sol = solve_ivp(ode, t_span, [SOC_initial], 
                    events=shutdown, dense_output=True)
    return sol.t, sol.y[0]
```

---

## 2. 功耗模型 (Power Consumption Model)

### 2.1 总功耗计算
```python
def calculate_P_total(self, usage: UsageState) -> float:
    """
    Total power: P_total = P_base + P_screen + P_cpu + P_network + P_GPS + P_other
    Returns power in mW.
    """
    P_total = self.power.P_base_mW  # Base power: ~80 mW
    P_total += self.calculate_P_screen(usage)
    P_total += self.calculate_P_cpu(usage)
    P_total += self.calculate_P_network(usage)
    
    if usage.gps_active:
        P_total += self.power.P_gps_mW  # ~350 mW
    if usage.bluetooth_active:
        P_total += self.power.P_bluetooth_mW  # ~15 mW
        
    return P_total
```

### 2.2 屏幕功耗模型
```python
def calculate_P_screen(self, usage: UsageState) -> float:
    """
    Screen power model: P_screen = k_screen × A × B(t)^n
    Based on Carroll & Heiser (2010): 74-412 mW range
    """
    if not usage.screen_on:
        return 0.0
    
    P_min, P_max = 74, 412  # mW
    B_norm = max(0.01, usage.brightness)
    brightness_factor = B_norm ** 2.0  # Brightness exponent n=2
    
    return P_min + (P_max - P_min) * brightness_factor
```

### 2.3 CPU功耗模型
```python
def calculate_P_cpu(self, usage: UsageState) -> float:
    """
    CPU power model: P_cpu = P_static + k_cpu × U^m × f^p
    Based on Zhang et al. (2010): CMOS dynamic power scaling
    """
    U = usage.cpu_utilization        # 0 to 1
    f_norm = usage.cpu_frequency_normalized  # 0 to 1
    
    # P_dynamic ∝ U^m × f^p (m=1.5, p=2.8 for CMOS)
    P_dynamic = 0.8 * (U ** 1.5) * (f_norm ** 2.8) * 4000
    
    return 50 + P_dynamic  # P_static = 50 mW
```

### 2.4 网络功耗模型
```python
def calculate_P_network(self, usage: UsageState) -> float:
    """
    Network power: P_network = P_idle(Mode) + k_tx×R_tx + k_rx×R_rx
    Based on Pathak et al. (2012)
    """
    P = 0.0
    if usage.wifi_active:
        P += 25 + 0.1*usage.data_rate_tx_kbps + 0.05*usage.data_rate_rx_kbps
    if usage.cellular_active:
        P += 80 + 0.15*usage.data_rate_tx_kbps + 0.08*usage.data_rate_rx_kbps
    return P
```

---

## 3. 电池老化模型 (Battery Aging Model)

### 3.1 有效容量计算
```python
def get_effective_capacity_Wh(self, temperature: float, cycles: int) -> float:
    """
    Effective energy capacity considering temperature and aging:
    E_eff = E_nominal × f_temp(T) × f_age(n)
    """
    # Temperature factor (interpolated from industry data)
    temp_factor = self._get_temp_factor(temperature)
    
    # Aging factor (exponential model from NASA data)
    alpha = 0.0012  # Capacity fade coefficient
    age_factor = max(0.7, np.exp(-alpha * cycles))
    
    # E_nominal = V_nominal × Q_nominal
    E_nominal = 3.85 * 4.5  # V × Ah = Wh
    return E_nominal * temp_factor * age_factor
```

### 3.2 容量衰减模型（基于NASA数据拟合）
```python
def estimate_capacity_fade_rate(self):
    """
    Fit linear capacity fade model to NASA battery data:
    C(n) = C0 * (1 - γ * n)
    
    Returns mean fade rate γ ≈ 0.0008 (0.08% per cycle)
    """
    fade_rates = []
    for battery_id, data in self.battery_data.items():
        if data.n_cycles >= 50:
            def capacity_model(n, C0, gamma):
                return C0 * (1 - gamma * n)
            
            popt, _ = curve_fit(capacity_model, 
                               data.cycle_numbers, 
                               data.capacities,
                               p0=[data.initial_capacity, 0.002])
            fade_rates.append(popt[1])  # γ
    
    return np.mean(fade_rates)
```

---

## 4. 开路电压模型 (Open Circuit Voltage Model)

### 4.1 OCV多项式模型（基于Zenodo数据）
```python
def get_ocv(self, soc: float) -> float:
    """
    Open Circuit Voltage using Zenodo polynomial coefficients:
    V(SOC) = c0 + c1*SOC + c2*SOC^2 + c3*SOC^3 + c4*SOC^4 + c5*SOC^5
    """
    soc = np.clip(soc, 0.001, 1)
    
    # Zenodo-fitted coefficients
    c = [3.349, 2.441, -9.555, 20.922, -20.325, 7.381]
    
    V = sum(c[i] * soc**i for i in range(6))
    return max(3.0, V)  # V_min = 3.0V
```

---

## 5. 使用场景预测 (Usage Scenario Predictions)

### 5.1 电池续航预测
```python
def predict_time_to_empty(self, power_mw, initial_soc=1.0, 
                          soh=1.0, temperature=25.0):
    """Predict time to reach shutdown SOC (5%)"""
    def ode(t, y):
        if y[0] <= 0.05:
            return [0.0]
        E_wh = 3.7 * 4.5 * soh  # Energy capacity
        return [-power_mw/1000 / E_wh]
    
    def shutdown(t, y):
        return y[0] - 0.05
    shutdown.terminal = True
    shutdown.direction = -1
    
    sol = solve_ivp(ode, [0, 100], [initial_soc], events=shutdown)
    return sol.t[-1]

# Example usage scenarios
scenarios = {
    'idle':     {'brightness': 0.0, 'cpu_load': 0.05},  # ~50h
    'light':    {'brightness': 0.3, 'cpu_load': 0.15},  # ~25h
    'moderate': {'brightness': 0.5, 'cpu_load': 0.35},  # ~15h
    'heavy':    {'brightness': 0.7, 'cpu_load': 0.55},  # ~8h
    'gaming':   {'brightness': 0.9, 'cpu_load': 0.85},  # ~4h
}
```

---

## 6. 敏感性分析 (Sensitivity Analysis)

```python
def sensitivity_analysis(model, base_params):
    """
    Analyze sensitivity of battery life to each parameter
    """
    results = {}
    base_tte = model.predict_time_to_empty(base_params)
    
    # Brightness sensitivity
    for b in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tte = model.predict(brightness=b)
        results[f'brightness_{b}'] = (tte - base_tte) / base_tte
    
    # CPU load sensitivity
    for cpu in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tte = model.predict(cpu_load=cpu)
        results[f'cpu_{cpu}'] = (tte - base_tte) / base_tte
    
    # SOH (battery aging) sensitivity
    for soh in [1.0, 0.9, 0.8, 0.7]:
        tte = model.predict(soh=soh)
        results[f'soh_{soh}'] = (tte - base_tte) / base_tte
    
    return results
```

---

## 数据来源说明 (Data Sources)

| 数据集 | 来源 | 用途 |
|--------|------|------|
| NASA Battery | NASA Ames Prognostics Repository | 容量衰减参数估计 |
| Zenodo AndroWatts | Zenodo.org/records/14314943 | 功耗分解与亮度-功耗关系 |
| Carroll & Heiser (2010) | USENIX ATC | 屏幕/CPU基准功耗 |
| Pathak et al. (2012) | EuroSys | 网络功耗模型 |
| Zhang et al. (2010) | CODES+ISSS | CPU功耗公式 |

---

## 模型参数汇总 (Model Parameters Summary)

| 参数 | 符号 | 值 | 来源 |
|------|------|-----|------|
| 标称容量 | C_nominal | 4500 mAh | Modern smartphone |
| 标称电压 | V_nominal | 3.85 V | Li-ion standard |
| 容量衰减率 | γ | 0.0008/cycle | NASA data fitting |
| 基础功耗 | P_base | 80 mW | Literature |
| 屏幕功耗 | P_screen | 74-412 mW | Carroll & Heiser |
| CPU静态功耗 | P_cpu_static | 50 mW | Zhang et al. |
| 关机阈值 | SOC_shutdown | 5% | BMS standard |

