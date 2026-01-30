"""
MCM 2026 Problem A: Smartphone Battery Drain Model
A continuous-time mathematical model for smartphone battery state of charge (SOC)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional


@dataclass
class BatteryParameters:
    """Parameters for the lithium-ion battery model"""
    # Battery capacity
    nominal_capacity: float = 4000  # mAh (typical smartphone battery)
    
    # Internal resistance parameters (Ohms)
    R_internal_0: float = 0.1  # Base internal resistance
    R_internal_aging_coeff: float = 0.001  # Resistance increase per cycle
    
    # Capacity fade parameters
    capacity_fade_rate: float = 0.0002  # Capacity fade per cycle (0.02% per cycle)
    
    # Temperature effects
    T_optimal: float = 25.0  # Optimal temperature (°C)
    T_coeff_low: float = 0.01  # Low temperature capacity reduction coefficient
    T_coeff_high: float = 0.005  # High temperature degradation coefficient
    
    # Self-discharge rate (fraction per hour at idle)
    self_discharge_rate: float = 0.0001  # ~0.01% per hour


@dataclass
class UsageParameters:
    """Parameters for phone usage/power consumption"""
    # Base power consumption (mW)
    P_idle: float = 50  # Idle power (screen off, minimal background)
    
    # Screen power (mW)
    P_screen_base: float = 200  # Base screen power at 50% brightness
    brightness_factor: float = 1.0  # Multiplier (0.0 to 1.0 for 0-100% brightness)
    screen_on: bool = True
    
    # Processor power (mW)
    P_processor_idle: float = 100  # CPU idle
    P_processor_max: float = 3000  # CPU at full load
    processor_load: float = 0.2  # Fraction of max load (0.0 to 1.0)
    
    # Network power (mW)
    P_wifi: float = 150  # WiFi active
    P_cellular: float = 300  # Cellular (4G/5G)
    P_bluetooth: float = 20  # Bluetooth
    wifi_active: bool = True
    cellular_active: bool = False
    bluetooth_active: bool = False
    
    # GPS power (mW)
    P_gps: float = 400  # GPS active
    gps_active: bool = False
    
    # Background apps (mW per app)
    P_background_app: float = 30
    n_background_apps: int = 5  # Number of active background apps
    
    # Ambient temperature (°C)
    temperature: float = 25.0


class SmartphoneBatteryModel:
    """
    Continuous-time model for smartphone battery state of charge (SOC)
    
    The model is based on a modified Coulomb counting approach with
    temperature-dependent capacity and usage-dependent discharge rate.
    
    Main governing equation:
    dSOC/dt = -P_total(t) / (V_nominal * Q_effective(T, cycles))
    
    where:
    - SOC: State of charge (0 to 1)
    - P_total: Total power consumption (W)
    - V_nominal: Nominal battery voltage (V)
    - Q_effective: Effective battery capacity considering temperature and aging
    """
    
    def __init__(self, battery_params: BatteryParameters = None, 
                 usage_params: UsageParameters = None):
        self.battery = battery_params or BatteryParameters()
        self.usage = usage_params or UsageParameters()
        
        # Battery nominal voltage
        self.V_nominal = 3.7  # Typical Li-ion nominal voltage
        
        # Cycle count (for aging effects)
        self.cycle_count = 0
        
    def get_effective_capacity(self, temperature: float) -> float:
        """
        Calculate effective battery capacity considering temperature and aging
        
        Q_eff = Q_nominal * f_age(cycles) * f_temp(T)
        
        f_age = 1 - alpha * cycles  (linear capacity fade)
        f_temp = 1 - beta * |T - T_opt|  (temperature deviation effect)
        """
        # Aging factor (capacity fade with cycles)
        f_age = max(0.7, 1 - self.battery.capacity_fade_rate * self.cycle_count)
        
        # Temperature factor
        T_diff = abs(temperature - self.battery.T_optimal)
        if temperature < self.battery.T_optimal:
            # Cold reduces available capacity more significantly
            f_temp = max(0.5, 1 - self.battery.T_coeff_low * T_diff)
        else:
            # Heat degrades battery but capacity reduction is less immediate
            f_temp = max(0.8, 1 - self.battery.T_coeff_high * T_diff)
        
        return self.battery.nominal_capacity * f_age * f_temp
    
    def calculate_power_consumption(self, usage: UsageParameters = None) -> float:
        """
        Calculate total instantaneous power consumption (mW)
        
        P_total = P_idle + P_screen + P_processor + P_network + P_gps + P_background
        """
        if usage is None:
            usage = self.usage
            
        P_total = usage.P_idle
        
        # Screen power
        if usage.screen_on:
            P_total += usage.P_screen_base * (0.5 + 0.5 * usage.brightness_factor)
        
        # Processor power (linear interpolation between idle and max)
        P_processor = (usage.P_processor_idle + 
                      (usage.P_processor_max - usage.P_processor_idle) * usage.processor_load)
        P_total += P_processor
        
        # Network power
        if usage.wifi_active:
            P_total += usage.P_wifi
        if usage.cellular_active:
            P_total += usage.P_cellular
        if usage.bluetooth_active:
            P_total += usage.P_bluetooth
            
        # GPS power
        if usage.gps_active:
            P_total += usage.P_gps
            
        # Background apps
        P_total += usage.P_background_app * usage.n_background_apps
        
        return P_total
    
    def get_internal_resistance(self) -> float:
        """
        Calculate internal resistance considering aging
        
        R_int = R_0 * (1 + alpha * cycles)
        """
        return self.battery.R_internal_0 * (1 + self.battery.R_internal_aging_coeff * self.cycle_count)
    
    def soc_derivative(self, t: float, SOC: float, 
                       usage_func: Callable[[float], UsageParameters] = None) -> float:
        """
        Calculate the rate of change of SOC
        
        dSOC/dt = -P_total(t) / (V * Q_eff) - k_self * SOC
        
        This is the core continuous-time equation governing battery drain.
        """
        # Get current usage parameters
        if usage_func is not None:
            current_usage = usage_func(t)
        else:
            current_usage = self.usage
            
        # Calculate power consumption (convert mW to W)
        P_total = self.calculate_power_consumption(current_usage) / 1000.0
        
        # Get effective capacity (convert mAh to Ah)
        Q_eff = self.get_effective_capacity(current_usage.temperature) / 1000.0
        
        # Calculate discharge rate
        # dSOC/dt = -I/Q = -P/(V*Q)
        discharge_rate = -P_total / (self.V_nominal * Q_eff)
        
        # Add self-discharge (very small, but included for completeness)
        self_discharge = -self.battery.self_discharge_rate * SOC
        
        return discharge_rate + self_discharge
    
    def simulate(self, t_span: Tuple[float, float], SOC_initial: float = 1.0,
                 usage_func: Callable[[float], UsageParameters] = None,
                 t_eval: np.ndarray = None) -> Dict:
        """
        Simulate battery discharge over time using numerical integration
        
        Parameters:
        -----------
        t_span : tuple (t_start, t_end) in hours
        SOC_initial : initial state of charge (0 to 1)
        usage_func : optional function(t) -> UsageParameters for time-varying usage
        t_eval : specific time points to evaluate (optional)
        
        Returns:
        --------
        Dictionary with 't' (time), 'SOC' (state of charge), and 'power' (power consumption)
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: self.soc_derivative(t, y[0], usage_func),
            t_span,
            [SOC_initial],
            t_eval=t_eval,
            method='RK45',
            events=lambda t, y: y[0] - 0.01  # Stop when SOC reaches 1%
        )
        
        # Calculate power at each time point
        power = []
        for t in sol.t:
            if usage_func is not None:
                usage = usage_func(t)
            else:
                usage = self.usage
            power.append(self.calculate_power_consumption(usage))
        
        return {
            't': sol.t,
            'SOC': sol.y[0],
            'power': np.array(power),
            'success': sol.success
        }
    
    def predict_time_to_empty(self, SOC_initial: float = 1.0,
                              usage_func: Callable[[float], UsageParameters] = None,
                              SOC_threshold: float = 0.01) -> float:
        """
        Predict the time until battery reaches the threshold SOC
        
        Uses numerical simulation to find when SOC crosses the threshold.
        
        Parameters:
        -----------
        SOC_initial : starting SOC (0 to 1)
        usage_func : optional time-varying usage function
        SOC_threshold : SOC level considered "empty" (default 1%)
        
        Returns:
        --------
        Time to empty in hours
        """
        # Estimate maximum possible time (very rough upper bound)
        min_power = self.battery.nominal_capacity * self.V_nominal / 100  # 100 hours minimum estimate
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
    """
    scenarios = {}
    
    # Scenario 1: Idle (screen off, minimal use)
    scenarios['idle'] = UsageParameters(
        screen_on=False,
        processor_load=0.05,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=False,
        gps_active=False,
        n_background_apps=3,
        brightness_factor=0.0
    )
    
    # Scenario 2: Light use (occasional screen, checking messages)
    scenarios['light'] = UsageParameters(
        screen_on=True,
        processor_load=0.15,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=5,
        brightness_factor=0.3
    )
    
    # Scenario 3: Moderate use (social media, web browsing)
    scenarios['moderate'] = UsageParameters(
        screen_on=True,
        processor_load=0.35,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=8,
        brightness_factor=0.5
    )
    
    # Scenario 4: Heavy use (video streaming, gaming)
    scenarios['heavy'] = UsageParameters(
        screen_on=True,
        processor_load=0.75,
        wifi_active=True,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=10,
        brightness_factor=0.8
    )
    
    # Scenario 5: Navigation (GPS + screen + cellular)
    scenarios['navigation'] = UsageParameters(
        screen_on=True,
        processor_load=0.5,
        wifi_active=False,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=True,
        n_background_apps=5,
        brightness_factor=0.7
    )
    
    # Scenario 6: Gaming (max processor, full screen)
    scenarios['gaming'] = UsageParameters(
        screen_on=True,
        processor_load=0.95,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=False,
        gps_active=False,
        n_background_apps=2,
        brightness_factor=1.0
    )
    
    # Scenario 7: Cold weather light use
    scenarios['cold_weather'] = UsageParameters(
        screen_on=True,
        processor_load=0.15,
        wifi_active=True,
        cellular_active=False,
        bluetooth_active=True,
        gps_active=False,
        n_background_apps=5,
        brightness_factor=0.3,
        temperature=5.0  # 5°C cold weather
    )
    
    # Scenario 8: Hot weather heavy use
    scenarios['hot_weather'] = UsageParameters(
        screen_on=True,
        processor_load=0.7,
        wifi_active=True,
        cellular_active=True,
        bluetooth_active=True,
        gps_active=True,
        n_background_apps=10,
        brightness_factor=0.9,
        temperature=40.0  # 40°C hot weather
    )
    
    return scenarios


def run_comprehensive_analysis():
    """
    Run comprehensive analysis including all scenarios, sensitivity analysis,
    and generate visualizations
    """
    print("=" * 70)
    print("MCM 2026 Problem A: Smartphone Battery Drain Model Analysis")
    print("=" * 70)
    
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
    plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('discharge_curves.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('temperature_effects.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('aging_effects.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('power_breakdown.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('optimization_impact.png', dpi=150, bbox_inches='tight')
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
