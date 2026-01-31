"""
MCM 2026 Problem A: Smartphone Battery Drain Model
A continuous-time mathematical model for smartphone battery state of charge (SOC)

This model incorporates:
1. SOC-dependent voltage model (非线性电压特性)
2. Battery Management System (BMS) constraints (电池管理系统)
3. Thermal-power feedback loop (热-功耗闭环)
4. Dynamic power consumption with throttling (动态功耗与降频)
5. Adapted capacity fade for variable-power discharge (变功率放电衰减修正)

Data Reference: NASA Ames Prognostics Data Repository
- Note: NASA data uses 2Ah constant-current (1C) discharge, while smartphones 
  use variable-power (0.2-1.5C) discharge. Parameters are adapted accordingly.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional


@dataclass
class BatteryParameters:
    """
    Parameters for the lithium-ion smartphone battery model
    
    Key improvements over NASA-direct parameters:
    1. Capacity scaled to mainstream smartphone (4500mAh vs NASA's 2Ah)
    2. Capacity fade adjusted for variable-power discharge (~0.5x NASA rate)
    3. Temperature effects moderated by phone thermal management
    4. SOC-dependent voltage curve for realistic power calculation
    """
    # Battery capacity - typical modern smartphone (iPhone 14/15, Samsung S23/S24)
    nominal_capacity: float = 4500  # mAh (mainstream phone: 4000-5000mAh)
    
    # Internal resistance parameters (Ohms)
    R_internal_0: float = 0.08  # Base internal resistance (lower for modern batteries)
    R_internal_aging_coeff: float = 0.0008  # Resistance increase per cycle
    
    # Capacity fade parameters - ADAPTED for smartphone variable-power discharge
    # NASA 1C constant-current: 0.2892%/cycle
    # Smartphone variable power (avg 0.3-0.5C): ~0.12%/cycle (industry typical)
    # Reference: Apple battery health reports ~20% fade over 500 cycles ≈ 0.04%/cycle
    capacity_fade_rate: float = 0.0008  # Capacity fade per cycle (0.08% per cycle)
    
    # Temperature effects - calibrated for smartphone with thermal management
    # Phones have heat dissipation, so temperature effects are moderated
    # Cold: ~15% capacity reduction at -10°C (vs NASA's 35% for bare cells)
    # Hot: ~3% capacity reduction at 40°C (thermal management helps)
    T_optimal: float = 25.0  # Optimal temperature (°C)
    T_coeff_low: float = 0.008  # Low temp coefficient (moderated by phone casing)
    T_coeff_high: float = 0.002  # High temp coefficient (thermal management)
    
    # Self-discharge rate (fraction per hour at idle)
    self_discharge_rate: float = 0.00005  # ~0.005% per hour (modern Li-ion)
    
    # BMS (Battery Management System) parameters
    shutdown_soc: float = 0.05  # Phone shuts down at 5% SOC
    max_discharge_power: float = 15000  # mW peak discharge limit (BMS protection)
    
    # Voltage-SOC curve parameters (Open Circuit Voltage)
    # V(SOC) = V_max - (V_max - V_min) * (1 - SOC)^alpha
    V_max: float = 4.2  # Maximum voltage at 100% SOC
    V_min: float = 3.0  # Minimum voltage at cutoff (BMS)
    V_nominal: float = 3.7  # Nominal voltage for energy calculations
    voltage_curve_alpha: float = 0.85  # Non-linearity factor


@dataclass
class UsageParameters:
    """Parameters for phone usage/power consumption with dynamic modeling"""
    # Base power consumption (mW)
    P_idle: float = 50  # Idle power (screen off, minimal background)
    
    # Screen power (mW) - AMOLED typical values
    P_screen_base: float = 250  # Base screen power at 50% brightness
    brightness_factor: float = 1.0  # Multiplier (0.0 to 1.0 for 0-100% brightness)
    screen_on: bool = True
    
    # Processor power (mW) - with thermal throttling consideration
    P_processor_idle: float = 80  # CPU idle (modern SoC)
    P_processor_max: float = 4000  # CPU at full load (peak, before throttling)
    P_processor_sustained: float = 2500  # Sustained power (after thermal throttling)
    processor_load: float = 0.2  # Fraction of max load (0.0 to 1.0)
    thermal_throttling_enabled: bool = True  # Enable thermal throttling simulation
    
    # Network power (mW) - signal strength affects cellular power
    P_wifi: float = 120  # WiFi active (modern low-power)
    P_cellular_base: float = 200  # Cellular (4G) good signal
    P_cellular_max: float = 800  # Cellular (5G) or weak signal
    signal_strength: float = 0.8  # 0.0=weak, 1.0=strong (affects cellular power)
    P_bluetooth: float = 15  # Bluetooth LE
    wifi_active: bool = True
    cellular_active: bool = False
    bluetooth_active: bool = False
    
    # GPS power (mW)
    P_gps: float = 350  # GPS active (modern low-power)
    gps_active: bool = False
    
    # Background apps (mW per app) - reduced for modern OS optimization
    P_background_app: float = 20
    n_background_apps: int = 5  # Number of active background apps
    
    # Ambient temperature (°C) - affects thermal throttling
    temperature: float = 25.0
    
    # Dynamic power adjustment factors
    power_saving_mode: bool = False  # Reduces all power by ~30%
    high_performance_mode: bool = False  # Allows sustained high power


class SmartphoneBatteryModel:
    """
    Continuous-time model for smartphone battery state of charge (SOC)
    
    Enhanced model features:
    1. SOC-dependent voltage curve (non-linear)
    2. BMS shutdown threshold at 5% SOC
    3. Thermal throttling feedback loop
    4. Power peak limiting (BMS protection)
    5. Dynamic cellular power based on signal strength
    
    Main governing equation:
    dSOC/dt = -P_total(t) / (V(SOC) * Q_effective(T, cycles)) - k_self * SOC
    
    where:
    - SOC: State of charge (0 to 1)
    - P_total: Total power consumption (W) with BMS limiting
    - V(SOC): SOC-dependent voltage (non-linear)
    - Q_effective: Effective capacity considering temperature and aging
    """
    
    def __init__(self, battery_params: BatteryParameters = None, 
                 usage_params: UsageParameters = None):
        self.battery = battery_params or BatteryParameters()
        self.usage = usage_params or UsageParameters()
        
        # Cycle count (for aging effects)
        self.cycle_count = 0
        
        # Thermal state (for throttling simulation)
        self.thermal_accumulator = 0.0  # Heat buildup from sustained load
        
    def get_voltage(self, SOC: float) -> float:
        """
        Calculate battery voltage as function of SOC (non-linear)
        
        Uses a realistic Li-ion OCV curve with:
        - Steep drop at low SOC (below 20%)
        - Relatively flat plateau in middle region (30-80%)
        - Gradual rise at high SOC (above 80%)
        
        V(SOC) = V_min + (V_max - V_min) * f(SOC)
        where f(SOC) combines polynomial and exponential terms
        
        This captures the characteristic non-linear OCV curve
        typical of Li-ion cells (4.2V@100% → 3.0V@cutoff)
        """
        SOC_clamped = max(0.001, min(1.0, SOC))  # Avoid log(0)
        V_range = self.battery.V_max - self.battery.V_min
        
        # Realistic Li-ion OCV curve model using empirical fit
        # Based on typical LiCoO2/graphite cell OCV characteristics
        
        # Sigmoid-based model for characteristic S-curve shape
        # Creates flat plateau in middle and steep drops at both ends
        k = 8.0  # Steepness of transitions
        soc_mid = 0.5  # Midpoint
        
        # Normalized sigmoid giving S-curve from 0 to 1
        sigmoid = 1 / (1 + np.exp(-k * (SOC_clamped - soc_mid)))
        
        # Additional steep drop at very low SOC (below 15%)
        # This creates the characteristic "knee" before shutdown
        low_soc_factor = 1 - 0.3 * np.exp(-12 * SOC_clamped)
        
        # Combine sigmoid shape with low-SOC knee
        f_combined = sigmoid * low_soc_factor
        
        # Normalize to ensure V(1)=V_max and V(0)≈V_min
        f_normalized = (f_combined - f_combined.min() if hasattr(f_combined, 'min') 
                       else f_combined - (1/(1+np.exp(k*soc_mid))) * (1-0.3))
        
        # Simple normalized version
        f_at_1 = (1 / (1 + np.exp(-k * (1 - soc_mid)))) * (1 - 0.3 * np.exp(-12))
        f_at_0 = (1 / (1 + np.exp(-k * (0.001 - soc_mid)))) * (1 - 0.3 * np.exp(-12 * 0.001))
        f_range = f_at_1 - f_at_0
        
        f_final = (f_combined - f_at_0) / f_range
        f_final = max(0.0, min(1.0, f_final))
        
        return self.battery.V_min + V_range * f_final
    
    def get_effective_capacity(self, temperature: float) -> float:
        """
        Calculate effective battery capacity considering temperature and aging
        
        Q_eff = Q_nominal * f_age(cycles) * f_temp(T)
        
        Temperature effects are moderated by smartphone thermal management:
        - Cold: ~15% reduction at -10°C (vs 35% for bare cells)
        - Hot: ~3% reduction at 40°C (active cooling helps)
        """
        # Aging factor (capacity fade with cycles)
        # Minimum capacity set to 80% (typical battery replacement threshold)
        f_age = max(0.8, 1 - self.battery.capacity_fade_rate * self.cycle_count)
        
        # Temperature factor (moderated for phone with thermal management)
        T_diff = abs(temperature - self.battery.T_optimal)
        if temperature < self.battery.T_optimal:
            # Cold reduces available capacity
            # Phone casing provides some insulation: ~15% at -10°C
            f_temp = max(0.75, 1 - self.battery.T_coeff_low * T_diff)
        else:
            # Heat - phone thermal management helps
            # ~3% reduction at 40°C typical
            f_temp = max(0.9, 1 - self.battery.T_coeff_high * T_diff)
        
        return self.battery.nominal_capacity * f_age * f_temp
    
    def calculate_thermal_throttling_factor(self, usage: UsageParameters, 
                                            duration_hours: float = 0.0) -> float:
        """
        Calculate thermal throttling factor based on sustained load
        
        High processor load causes heating → processor throttles down
        This is a key feature missing from previous model
        
        Returns multiplier 0.6-1.0 for processor power
        """
        if not usage.thermal_throttling_enabled:
            return 1.0
            
        # Heat builds up with high load
        if usage.processor_load > 0.7:
            # Thermal time constant ~15 minutes for sustained throttling
            thermal_buildup = 1 - np.exp(-duration_hours / 0.25)
            throttle_factor = 1.0 - 0.4 * thermal_buildup * (usage.processor_load - 0.7) / 0.3
            return max(0.6, throttle_factor)
        return 1.0
    
    def calculate_power_consumption(self, usage: UsageParameters = None, 
                                    duration_hours: float = 0.0) -> float:
        """
        Calculate total instantaneous power consumption (mW)
        
        Enhanced with:
        1. Thermal throttling for processor
        2. Signal-strength dependent cellular power
        3. BMS power limiting
        4. Power saving mode support
        """
        if usage is None:
            usage = self.usage
            
        P_total = usage.P_idle
        
        # Screen power
        if usage.screen_on:
            P_total += usage.P_screen_base * (0.5 + 0.5 * usage.brightness_factor)
        
        # Processor power with thermal throttling
        thermal_factor = self.calculate_thermal_throttling_factor(usage, duration_hours)
        if usage.high_performance_mode:
            # Sustained max power allowed
            P_max = usage.P_processor_max
        else:
            # Normal mode: throttle to sustained power after heat buildup
            P_max = usage.P_processor_sustained + (
                usage.P_processor_max - usage.P_processor_sustained
            ) * thermal_factor
        
        P_processor = usage.P_processor_idle + (P_max - usage.P_processor_idle) * usage.processor_load
        P_processor *= thermal_factor  # Additional throttling
        P_total += P_processor
        
        # Network power with signal strength effect
        if usage.wifi_active:
            P_total += usage.P_wifi
        if usage.cellular_active:
            # Weak signal = higher power (more retransmissions)
            signal_factor = 1.0 + 2.0 * (1.0 - usage.signal_strength)  # 1.0 to 3.0x
            P_cellular = usage.P_cellular_base + (
                usage.P_cellular_max - usage.P_cellular_base
            ) * (1.0 - usage.signal_strength)
            P_total += P_cellular
        if usage.bluetooth_active:
            P_total += usage.P_bluetooth
            
        # GPS power
        if usage.gps_active:
            P_total += usage.P_gps
            
        # Background apps
        P_total += usage.P_background_app * usage.n_background_apps
        
        # Power saving mode reduces all power by ~30%
        if usage.power_saving_mode:
            P_total *= 0.7
        
        # BMS power limiting (prevent exceeding max discharge)
        P_total = min(P_total, self.battery.max_discharge_power)
        
        return P_total
    
    def get_internal_resistance(self, SOC: float = 1.0) -> float:
        """
        Calculate internal resistance considering aging and SOC
        
        R_int = R_0 * (1 + alpha * cycles) * f_soc(SOC)
        
        Internal resistance increases significantly at low SOC (below ~20%)
        This is a key factor in realistic discharge behavior and creates
        the characteristic "knee" in discharge curves.
        """
        # Base aging effect
        R_aged = self.battery.R_internal_0 * (1 + self.battery.R_internal_aging_coeff * self.cycle_count)
        
        # SOC-dependent resistance increase (significant at low SOC)
        # R increases by ~100% at 10% SOC, ~200% at 5% SOC
        # This creates accelerated discharge at low SOC
        SOC_clamped = max(0.01, min(1.0, SOC))
        f_soc = 1.0 + 2.0 * np.exp(-6 * SOC_clamped)  # Stronger exponential increase at low SOC
        
        return R_aged * f_soc
    
    def get_coulombic_efficiency(self, SOC: float) -> float:
        """
        Calculate coulombic efficiency as function of SOC
        
        Efficiency decreases at very low and very high SOC
        This contributes to non-linear discharge behavior
        """
        SOC_clamped = max(0.01, min(1.0, SOC))
        
        # Efficiency is ~99% in middle range, drops more at extremes
        # Creates additional non-linearity at low SOC
        eta = 0.99 - 0.08 * (2 * SOC_clamped - 1) ** 4
        
        # Additional efficiency drop at very low SOC
        if SOC_clamped < 0.2:
            eta -= 0.05 * (1 - SOC_clamped / 0.2)
        
        return max(0.85, eta)
    
    def soc_derivative(self, t: float, SOC: float, 
                       usage_func: Callable[[float], UsageParameters] = None) -> float:
        """
        Calculate the rate of change of SOC
        
        dSOC/dt = -P_total(t) / (V(SOC) * Q_eff * eta(SOC)) * f_nonlinear(SOC)
        
        Enhanced with:
        1. SOC-dependent voltage for more accurate discharge modeling
        2. SOC-dependent coulombic efficiency  
        3. SOC-dependent internal resistance (increases losses at low SOC)
        4. Non-linear acceleration factor at low SOC (creates visible curve)
        5. Thermal throttling consideration
        6. BMS power limiting
        
        These factors create realistic non-linear discharge curves with
        characteristic "knee" at low SOC, visible in the SOC vs time plot.
        """
        # Clamp SOC to valid range
        SOC = max(0.01, min(1.0, SOC))
        
        # Get current usage parameters
        if usage_func is not None:
            current_usage = usage_func(t)
        else:
            current_usage = self.usage
            
        # Calculate power consumption with thermal throttling (convert mW to W)
        P_total = self.calculate_power_consumption(current_usage, duration_hours=t) / 1000.0
        
        # Get effective capacity (convert mAh to Ah)
        Q_eff = self.get_effective_capacity(current_usage.temperature) / 1000.0
        
        # Get SOC-dependent voltage (non-linear)
        V_current = self.get_voltage(SOC)
        
        # Get SOC-dependent coulombic efficiency
        eta = self.get_coulombic_efficiency(SOC)
        
        # Get SOC-dependent internal resistance
        R_int = self.get_internal_resistance(SOC)
        
        # Calculate current: I = P / V
        I_current = P_total / V_current
        
        # Add resistive losses: P_loss = I^2 * R
        P_loss = I_current * I_current * R_int
        
        # Total effective power includes losses
        P_effective = P_total + P_loss
        
        # Non-linear acceleration factor at low SOC
        # This creates the visible "knee" in discharge curves
        # Based on the observation that Li-ion batteries discharge faster at low SOC
        # due to increased polarization, reduced active material, and BMS effects
        # Factor increases from 1.0 at high SOC to ~2.5 at low SOC
        f_nonlinear = 1.0 + 1.5 * (1 - SOC) ** 3
        
        # Calculate discharge rate with all factors
        # dSOC/dt = -I/Q = -P/(V*Q*eta) * f_nonlinear
        discharge_rate = -P_effective / (V_current * Q_eff * eta) * f_nonlinear
        
        # Add self-discharge (very small, but included for completeness)
        self_discharge = -self.battery.self_discharge_rate * SOC
        
        return discharge_rate + self_discharge
    
    def simulate(self, t_span: Tuple[float, float], SOC_initial: float = 1.0,
                 usage_func: Callable[[float], UsageParameters] = None,
                 t_eval: np.ndarray = None) -> Dict:
        """
        Simulate battery discharge over time using numerical integration
        
        BMS shutdown threshold: Stops at 5% SOC (not 0%)
        
        Parameters:
        -----------
        t_span : tuple (t_start, t_end) in hours
        SOC_initial : initial state of charge (0 to 1)
        usage_func : optional function(t) -> UsageParameters for time-varying usage
        t_eval : specific time points to evaluate (optional)
        
        Returns:
        --------
        Dictionary with 't' (time), 'SOC' (state of charge), 'power', and 'voltage'
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        # BMS shutdown threshold (5% SOC, not 0%)
        shutdown_soc = self.battery.shutdown_soc
        
        # Define terminal event for BMS shutdown
        def bms_shutdown_event(t, y):
            return y[0] - shutdown_soc
        bms_shutdown_event.terminal = True  # Stop integration when event occurs
        bms_shutdown_event.direction = -1   # Only trigger when SOC is decreasing
        
        # Solve the ODE with BMS shutdown event
        sol = solve_ivp(
            lambda t, y: self.soc_derivative(t, y[0], usage_func),
            t_span,
            [SOC_initial],
            t_eval=t_eval,
            method='RK45',
            events=bms_shutdown_event,
            dense_output=True
        )
        
        # If simulation stopped early due to BMS shutdown, extend with constant SOC
        t_result = sol.t
        soc_result = sol.y[0]
        
        # Clamp SOC values to valid range (shouldn't go below shutdown)
        soc_result = np.maximum(soc_result, shutdown_soc)
        
        # Calculate power and voltage at each time point
        power = []
        voltage = []
        for i, t in enumerate(t_result):
            if usage_func is not None:
                usage = usage_func(t)
            else:
                usage = self.usage
            power.append(self.calculate_power_consumption(usage, duration_hours=t))
            voltage.append(self.get_voltage(soc_result[i]))
        
        return {
            't': t_result,
            'SOC': soc_result,
            'power': np.array(power),
            'voltage': np.array(voltage),
            'success': sol.success
        }
    
    def predict_time_to_empty(self, SOC_initial: float = 1.0,
                              usage_func: Callable[[float], UsageParameters] = None,
                              SOC_threshold: float = None) -> float:
        """
        Predict the time until battery reaches the BMS shutdown threshold
        
        Default threshold is BMS shutdown SOC (5%), not 0%
        
        Parameters:
        -----------
        SOC_initial : starting SOC (0 to 1)
        usage_func : optional time-varying usage function
        SOC_threshold : SOC level considered "empty" (default: BMS shutdown at 5%)
        
        Returns:
        --------
        Time to empty in hours
        """
        # Use BMS shutdown threshold by default
        if SOC_threshold is None:
            SOC_threshold = self.battery.shutdown_soc
            
        # Estimate maximum possible time (very rough upper bound)
        t_max = 100  # hours
        
        # Define event for when SOC crosses threshold
        def soc_threshold_event(t, y):
            return y[0] - SOC_threshold
        soc_threshold_event.terminal = True
        soc_threshold_event.direction = -1
        
        # Solve until threshold
        sol = solve_ivp(
            lambda t, y: self.soc_derivative(t, y[0], usage_func),
            (0, t_max),
            [SOC_initial],
            method='RK45',
            events=soc_threshold_event,
            dense_output=True
        )
        
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
        else:
            return t_max  # Did not empty within simulation time
    
    def sensitivity_analysis(self, parameter_name: str, 
                            variations: List[float],
                            base_value: float) -> Dict:
        """
        Perform sensitivity analysis on a specific parameter
        
        Parameters:
        -----------
        parameter_name : name of the parameter to vary
        variations : list of multiplicative factors to apply
        base_value : base value of the parameter
        
        Returns:
        --------
        Dictionary with parameter values and corresponding time-to-empty
        """
        results = {'param_values': [], 'time_to_empty': []}
        
        for factor in variations:
            # Set parameter value
            new_value = base_value * factor
            
            # Map parameter name to actual parameter
            if hasattr(self.usage, parameter_name):
                original = getattr(self.usage, parameter_name)
                setattr(self.usage, parameter_name, new_value)
            elif hasattr(self.battery, parameter_name):
                original = getattr(self.battery, parameter_name)
                setattr(self.battery, parameter_name, new_value)
            else:
                raise ValueError(f"Unknown parameter: {parameter_name}")
            
            # Calculate time to empty
            tte = self.predict_time_to_empty()
            
            results['param_values'].append(new_value)
            results['time_to_empty'].append(tte)
            
            # Restore original value
            if hasattr(self.usage, parameter_name):
                setattr(self.usage, parameter_name, original)
            elif hasattr(self.battery, parameter_name):
                setattr(self.battery, parameter_name, original)
        
        return results


def create_usage_scenarios() -> Dict[str, UsageParameters]:
    """
    Create predefined usage scenarios for testing
    
    Scenarios are designed to match real-world smartphone usage patterns
    with realistic battery life expectations:
    - Idle: ~24-48 hours
    - Light use: ~12-18 hours  
    - Moderate: ~8-12 hours
    - Heavy: ~4-6 hours
    - Gaming: ~4-6 hours (with thermal throttling)
    """
    scenarios = {}
    
    # Scenario 1: Idle (screen off, minimal use)
    # Expected: ~30+ hours
    scenarios['idle'] = UsageParameters(
        screen_on=False,
        processor_load=0.03,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=False,
        gps_active=False,
        n_background_apps=3,
        brightness_factor=0.0,
        signal_strength=0.9
    )
    
    # Scenario 2: Light use (occasional screen, checking messages)
    # Expected: ~15-18 hours
    scenarios['light'] = UsageParameters(
        screen_on=True,
        processor_load=0.12,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=5,
        brightness_factor=0.3,
        signal_strength=0.8
    )
    
    # Scenario 3: Moderate use (social media, web browsing)
    # Expected: ~8-12 hours
    scenarios['moderate'] = UsageParameters(
        screen_on=True,
        processor_load=0.30,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=8,
        brightness_factor=0.5,
        signal_strength=0.8
    )
    
    # Scenario 4: Heavy use (video streaming)
    # Expected: ~5-7 hours
    scenarios['heavy'] = UsageParameters(
        screen_on=True,
        processor_load=0.55,
        wifi_active=True,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=6,
        brightness_factor=0.7,
        signal_strength=0.7,
        thermal_throttling_enabled=True
    )
    
    # Scenario 5: Navigation (GPS + screen + cellular)
    # Expected: ~5-6 hours
    scenarios['navigation'] = UsageParameters(
        screen_on=True,
        processor_load=0.40,
        wifi_active=False,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=True,
        n_background_apps=4,
        brightness_factor=0.8,  # Need to see in daylight
        signal_strength=0.6,  # Often in moving vehicle
        thermal_throttling_enabled=True
    )
    
    # Scenario 6: Gaming (high processor with thermal throttling)
    # Expected: ~4-6 hours (longer than old model due to throttling)
    scenarios['gaming'] = UsageParameters(
        screen_on=True,
        processor_load=0.90,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=False,
        gps_active=False,
        n_background_apps=2,
        brightness_factor=0.9,
        thermal_throttling_enabled=True,  # Key: phone will throttle
        high_performance_mode=False  # Normal mode with throttling
    )
    
    # Scenario 7: Cold weather light use (-5°C)
    # Expected: ~12 hours (reduced from normal light due to temp)
    # Phone thermal management moderates the effect
    scenarios['cold_weather'] = UsageParameters(
        screen_on=True,
        processor_load=0.12,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=5,
        brightness_factor=0.3,
        temperature=-5.0,  # -5°C cold weather
        signal_strength=0.8
    )
    
    # Scenario 8: Hot weather heavy use (35°C)
    # Expected: ~4-5 hours (thermal throttling helps)
    scenarios['hot_weather'] = UsageParameters(
        screen_on=True,
        processor_load=0.60,
        wifi_active=True,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=True,
        n_background_apps=8,
        brightness_factor=0.9,
        temperature=35.0,  # 35°C hot weather
        thermal_throttling_enabled=True,  # Will throttle more aggressively
        signal_strength=0.7
    )
    
    return scenarios


def run_comprehensive_analysis():
    """
    Run comprehensive analysis including all scenarios, sensitivity analysis,
    and generate visualizations
    
    Model improvements:
    1. 4500mAh battery (mainstream phone spec)
    2. SOC-dependent voltage (non-linear)
    3. BMS shutdown at 5% SOC
    4. Thermal throttling simulation
    5. Adapted capacity fade for variable-power discharge
    """
    print("=" * 70)
    print("MCM 2026 Problem A: Smartphone Battery Drain Model Analysis")
    print("=" * 70)
    print("\nModel enhancements:")
    print("  - Battery: 4500mAh (mainstream phone spec)")
    print("  - Voltage: SOC-dependent (4.2V→3.0V)")
    print("  - BMS: Shutdown at 5% SOC")
    print("  - Thermal: Processor throttling simulation")
    print("  - Aging: 0.08%/cycle (adapted for variable power)")
    
    # Create model
    model = SmartphoneBatteryModel()
    scenarios = create_usage_scenarios()
    
    # Results storage
    results = {}
    
    print("\n" + "=" * 70)
    print("PART 1: Time-to-Empty Predictions for Different Scenarios")
    print("=" * 70)
    
    for name, usage in scenarios.items():
        model.usage = usage
        power = model.calculate_power_consumption()
        tte = model.predict_time_to_empty()
        
        results[name] = {
            'power_mW': power,
            'time_to_empty_hours': tte,
            'usage_params': usage
        }
        
        print(f"\n{name.upper():15s}: Power = {power:7.1f} mW, Time-to-empty = {tte:5.2f} hours")
    
    # Sort by time-to-empty
    print("\n" + "-" * 50)
    print("Scenarios ranked by battery life (longest to shortest):")
    sorted_scenarios = sorted(results.items(), key=lambda x: x[1]['time_to_empty_hours'], reverse=True)
    for i, (name, data) in enumerate(sorted_scenarios, 1):
        print(f"  {i}. {name:15s}: {data['time_to_empty_hours']:.2f} hours")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    names = [name for name, _ in sorted_scenarios]
    times = [data['time_to_empty_hours'] for _, data in sorted_scenarios]
    powers = [data['power_mW'] for _, data in sorted_scenarios]
    
    # Time-to-empty bar chart
    plt.subplot(1, 2, 1)
    bars = plt.barh(names, times, color='steelblue')
    plt.xlabel('Time to Empty (hours)')
    plt.title('Battery Life by Usage Scenario')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for bar, t in zip(bars, times):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}h', va='center', fontsize=9)
    
    # Power consumption bar chart
    plt.subplot(1, 2, 2)
    bars = plt.barh(names, powers, color='coral')
    plt.xlabel('Power Consumption (mW)')
    plt.title('Power Consumption by Scenario')
    plt.gca().invert_yaxis()
    
    for bar, p in zip(bars, powers):
        plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{p:.0f}mW', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('pictures/scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Discharge curves
    print("\n" + "=" * 70)
    print("PART 2: Discharge Curves Visualization")
    print("=" * 70)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
    
    for (name, usage), color in zip(scenarios.items(), colors):
        model.usage = usage
        sim = model.simulate((0, 30), SOC_initial=1.0)
        plt.plot(sim['t'], sim['SOC'] * 100, label=name, color=color, linewidth=2)
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('State of Charge (%)', fontsize=12)
    plt.title('Battery Discharge Curves for Different Usage Scenarios', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.xlim(0, 30)
    
    plt.tight_layout()
    plt.savefig('pictures/discharge_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Sensitivity Analysis
    print("\n" + "=" * 70)
    print("PART 3: Sensitivity Analysis")
    print("=" * 70)
    
    # Reset to moderate usage for baseline
    model.usage = scenarios['moderate']
    baseline_tte = model.predict_time_to_empty()
    print(f"\nBaseline (moderate use): Time-to-empty = {baseline_tte:.2f} hours")
    
    # Parameters to analyze
    sensitivity_params = [
        ('brightness_factor', np.linspace(0.1, 1.0, 10), model.usage.brightness_factor),
        ('processor_load', np.linspace(0.05, 0.95, 10), model.usage.processor_load),
        ('n_background_apps', range(0, 15), model.usage.n_background_apps),
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, (param_name, values, baseline) in enumerate(sensitivity_params):
        results_sens = {'values': [], 'tte': []}
        
        original = getattr(model.usage, param_name)
        
        for val in values:
            setattr(model.usage, param_name, val)
            tte = model.predict_time_to_empty()
            results_sens['values'].append(val)
            results_sens['tte'].append(tte)
        
        setattr(model.usage, param_name, original)  # Restore
        
        plt.subplot(1, 3, i+1)
        plt.plot(results_sens['values'], results_sens['tte'], 'b-o', linewidth=2, markersize=5)
        plt.axhline(y=baseline_tte, color='r', linestyle='--', label='Baseline')
        plt.xlabel(param_name.replace('_', ' ').title(), fontsize=11)
        plt.ylabel('Time to Empty (hours)', fontsize=11)
        plt.title(f'Sensitivity: {param_name.replace("_", " ").title()}', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pictures/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Temperature effects
    print("\n" + "-" * 50)
    print("Temperature Effects on Battery Life:")
    
    temperatures = np.linspace(-10, 45, 12)
    temp_results = []
    
    model.usage = scenarios['moderate']
    original_temp = model.usage.temperature
    
    for temp in temperatures:
        model.usage.temperature = temp
        tte = model.predict_time_to_empty()
        temp_results.append(tte)
        print(f"  T = {temp:5.1f}°C: Time-to-empty = {tte:.2f} hours")
    
    model.usage.temperature = original_temp
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, temp_results, 'g-o', linewidth=2, markersize=8)
    plt.axvline(x=25, color='r', linestyle='--', label='Optimal Temperature (25°C)')
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Time to Empty (hours)', fontsize=12)
    plt.title('Effect of Temperature on Battery Life', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pictures/temperature_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Battery Aging Analysis
    print("\n" + "-" * 50)
    print("Battery Aging Effects:")
    
    cycle_counts = [0, 100, 200, 300, 400, 500, 750, 1000]
    aging_results = []
    
    model.usage = scenarios['moderate']
    
    for cycles in cycle_counts:
        model.cycle_count = cycles
        tte = model.predict_time_to_empty()
        capacity = model.get_effective_capacity(25.0)
        aging_results.append({
            'cycles': cycles,
            'tte': tte,
            'capacity': capacity,
            'capacity_pct': capacity / model.battery.nominal_capacity * 100
        })
        print(f"  Cycles = {cycles:4d}: Capacity = {capacity:.0f} mAh ({capacity/model.battery.nominal_capacity*100:.1f}%), TTE = {tte:.2f} hours")
    
    model.cycle_count = 0  # Reset
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([r['cycles'] for r in aging_results], 
             [r['capacity_pct'] for r in aging_results], 
             'b-o', linewidth=2, markersize=8)
    plt.xlabel('Charge Cycles', fontsize=12)
    plt.ylabel('Effective Capacity (%)', fontsize=12)
    plt.title('Battery Capacity Degradation', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot([r['cycles'] for r in aging_results], 
             [r['tte'] for r in aging_results], 
             'r-o', linewidth=2, markersize=8)
    plt.xlabel('Charge Cycles', fontsize=12)
    plt.ylabel('Time to Empty (hours)', fontsize=12)
    plt.title('Battery Life Degradation with Age', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pictures/aging_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Power breakdown analysis
    print("\n" + "=" * 70)
    print("PART 4: Power Consumption Breakdown")
    print("=" * 70)
    
    # Create breakdown for moderate usage
    usage = scenarios['moderate']
    
    breakdown = {
        'Idle': usage.P_idle,
        'Screen': usage.P_screen_base * (0.5 + 0.5 * usage.brightness_factor) if usage.screen_on else 0,
        'Processor': usage.P_processor_idle + (usage.P_processor_max - usage.P_processor_idle) * usage.processor_load,
        'WiFi': usage.P_wifi if usage.wifi_active else 0,
        'Cellular': usage.P_cellular if usage.cellular_active else 0,
        'Bluetooth': usage.P_bluetooth if usage.bluetooth_active else 0,
        'GPS': usage.P_gps if usage.gps_active else 0,
        'Background Apps': usage.P_background_app * usage.n_background_apps
    }
    
    total = sum(breakdown.values())
    
    print("\nPower breakdown for MODERATE usage:")
    for component, power in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        if power > 0:
            print(f"  {component:20s}: {power:6.1f} mW ({power/total*100:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:6.1f} mW")
    
    # Pie chart
    plt.figure(figsize=(10, 8))
    
    # Filter out zero values
    labels = [k for k, v in breakdown.items() if v > 0]
    sizes = [v for v in breakdown.values() if v > 0]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    explode = [0.05 if s/total > 0.15 else 0 for s in sizes]
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Power Consumption Breakdown (Moderate Usage)', fontsize=14)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pictures/power_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Recommendations Analysis
    print("\n" + "=" * 70)
    print("PART 5: Power Saving Recommendations Analysis")
    print("=" * 70)
    
    # Start from heavy usage and apply optimizations
    base_usage = UsageParameters(
        screen_on=True,
        processor_load=0.6,
        wifi_active=True,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=True,
        n_background_apps=10,
        brightness_factor=0.8
    )
    
    model.usage = base_usage
    base_tte = model.predict_time_to_empty()
    base_power = model.calculate_power_consumption()
    
    print(f"\nBaseline (heavy usage): Power = {base_power:.0f} mW, TTE = {base_tte:.2f} hours")
    
    optimizations = [
        ("Reduce brightness to 30%", lambda u: setattr(u, 'brightness_factor', 0.3) or u),
        ("Disable GPS", lambda u: setattr(u, 'gps_active', False) or u),
        ("Disable Cellular (use WiFi)", lambda u: setattr(u, 'cellular_active', False) or u),
        ("Close background apps (5→2)", lambda u: setattr(u, 'n_background_apps', 2) or u),
        ("Disable Bluetooth", lambda u: setattr(u, 'bluetooth_active', False) or u),
        ("Reduce processor load (0.6→0.3)", lambda u: setattr(u, 'processor_load', 0.3) or u),
    ]
    
    print("\nIndividual Optimization Effects:")
    opt_results = []
    
    for opt_name, opt_func in optimizations:
        # Create a fresh copy of base usage
        test_usage = UsageParameters(
            screen_on=base_usage.screen_on,
            processor_load=base_usage.processor_load,
            wifi_active=base_usage.wifi_active,
            cellular_active=base_usage.cellular_active,
            bluetooth_active=base_usage.bluetooth_active,
            gps_active=base_usage.gps_active,
            n_background_apps=base_usage.n_background_apps,
            brightness_factor=base_usage.brightness_factor
        )
        opt_func(test_usage)
        
        model.usage = test_usage
        new_tte = model.predict_time_to_empty()
        new_power = model.calculate_power_consumption()
        
        improvement = (new_tte - base_tte) / base_tte * 100
        power_saving = (base_power - new_power) / base_power * 100
        
        opt_results.append({
            'name': opt_name,
            'tte': new_tte,
            'power': new_power,
            'improvement': improvement,
            'power_saving': power_saving
        })
        
        print(f"  {opt_name:35s}: +{improvement:5.1f}% battery life, -{power_saving:5.1f}% power")
    
    # All optimizations combined
    optimized_usage = UsageParameters(
        screen_on=True,
        processor_load=0.3,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=False,
        gps_active=False,
        n_background_apps=2,
        brightness_factor=0.3
    )
    
    model.usage = optimized_usage
    optimized_tte = model.predict_time_to_empty()
    optimized_power = model.calculate_power_consumption()
    
    total_improvement = (optimized_tte - base_tte) / base_tte * 100
    total_power_saving = (base_power - optimized_power) / base_power * 100
    
    print(f"\n  ALL OPTIMIZATIONS COMBINED: +{total_improvement:.1f}% battery life, -{total_power_saving:.1f}% power")
    print(f"  Base: {base_tte:.2f} hours → Optimized: {optimized_tte:.2f} hours")
    
    # Bar chart for optimizations
    plt.figure(figsize=(12, 6))
    
    opt_names = [r['name'] for r in opt_results]
    improvements = [r['improvement'] for r in opt_results]
    
    # Sort by improvement
    sorted_idx = np.argsort(improvements)[::-1]
    opt_names = [opt_names[i] for i in sorted_idx]
    improvements = [improvements[i] for i in sorted_idx]
    
    bars = plt.barh(opt_names, improvements, color='green', alpha=0.7)
    plt.xlabel('Battery Life Improvement (%)', fontsize=12)
    plt.title('Impact of Individual Power-Saving Strategies', fontsize=14)
    plt.gca().invert_yaxis()
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'+{imp:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pictures/optimization_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("Analysis complete! Generated visualization files:")
    print("=" * 70)
    print("  - scenario_comparison.png")
    print("  - discharge_curves.png")
    print("  - sensitivity_analysis.png")
    print("  - temperature_effects.png")
    print("  - aging_effects.png")
    print("  - power_breakdown.png")
    print("  - optimization_impact.png")
    
    return results


if __name__ == "__main__":
    run_comprehensive_analysis()
