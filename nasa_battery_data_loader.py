"""
NASA Battery Data Loader and Parameter Estimator
Loads and analyzes battery aging data from NASA Ames Prognostics Data Repository
for use in battery model parameter estimation and validation.

Data Source: NASA Prognostics Center of Excellence Data Set Repository
Reference: B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
import glob


@dataclass
class BatteryTestData:
    """Container for a single battery's test data"""
    battery_id: str
    cycle_numbers: np.ndarray
    capacities: np.ndarray  # Ah
    temperatures: np.ndarray  # °C
    discharge_times: np.ndarray  # hours
    voltages_mean: np.ndarray  # V
    voltages_min: np.ndarray  # V
    voltages_max: np.ndarray  # V
    
    @property
    def initial_capacity(self) -> float:
        return self.capacities[0]
    
    @property
    def final_capacity(self) -> float:
        return self.capacities[-1]
    
    @property
    def n_cycles(self) -> int:
        return len(self.capacities)
    
    @property
    def capacity_fade_rate(self) -> float:
        """Linear capacity fade rate per cycle"""
        return (1 - self.final_capacity / self.initial_capacity) / self.n_cycles
    
    @property
    def capacity_retention(self) -> float:
        """Final capacity as percentage of initial"""
        return self.final_capacity / self.initial_capacity * 100


@dataclass
class DischargeCurve:
    """Container for a single discharge cycle data"""
    battery_id: str
    cycle_number: int
    time: np.ndarray  # seconds
    voltage: np.ndarray  # V
    current: np.ndarray  # A
    temperature: np.ndarray  # °C
    capacity: float  # Ah
    
    @property
    def duration_hours(self) -> float:
        return self.time[-1] / 3600
    
    def get_soc_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate SOC from discharge curve using coulomb counting"""
        # Cumulative charge (integrate current over time)
        dt = np.diff(self.time, prepend=0)
        charge = np.cumsum(np.abs(self.current) * dt)  # Ah*s
        charge_ah = charge / 3600  # Convert to Ah
        
        # SOC = 1 - (charge_removed / total_capacity)
        soc = 1 - charge_ah / self.capacity
        time_hours = self.time / 3600
        
        return time_hours, np.clip(soc, 0, 1)


class NASABatteryDataLoader:
    """
    Loader for NASA Battery Aging Data Set
    Extracts cycling data, capacity fade, and discharge curves for model validation
    """
    
    def __init__(self, data_dir: str = "requests/5. Battery Data Set"):
        self.data_dir = data_dir
        self.battery_data: Dict[str, BatteryTestData] = {}
        self.discharge_curves: Dict[str, List[DischargeCurve]] = {}
        
    def find_all_mat_files(self) -> List[str]:
        """Find all .mat battery data files"""
        mat_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mat') and file.startswith('B'):
                    mat_files.append(os.path.join(root, file))
        return mat_files
    
    def load_battery_file(self, filepath: str) -> Optional[BatteryTestData]:
        """Load and extract data from a single battery .mat file"""
        try:
            data = loadmat(filepath)
            
            # Find battery key
            battery_keys = [k for k in data.keys() if k.startswith('B') and not k.startswith('__')]
            if not battery_keys:
                return None
            
            battery_key = battery_keys[0]
            battery_data = data[battery_key]
            cycles = battery_data['cycle'][0, 0]
            
            # Extract discharge cycle data
            cycle_nums = []
            capacities = []
            temperatures = []
            discharge_times = []
            voltages_mean = []
            voltages_min = []
            voltages_max = []
            
            cycle_count = 0
            for i in range(cycles.shape[1]):
                cycle = cycles[0, i]
                cycle_type = cycle['type'][0]
                
                if cycle_type == 'discharge':
                    cycle_data = cycle['data'][0, 0]
                    
                    if 'Capacity' in cycle_data.dtype.names:
                        cap = cycle_data['Capacity'][0, 0]
                        if cap > 0:  # Valid capacity
                            cycle_count += 1
                            cycle_nums.append(cycle_count)
                            capacities.append(cap)
                            
                            # Temperature
                            if 'Temperature_measured' in cycle_data.dtype.names:
                                temp = cycle_data['Temperature_measured'].flatten()
                                temperatures.append(np.mean(temp))
                            else:
                                temperatures.append(25.0)
                            
                            # Discharge time
                            if 'Time' in cycle_data.dtype.names:
                                time = cycle_data['Time'].flatten()
                                discharge_times.append(time[-1] / 3600)
                            else:
                                discharge_times.append(1.0)
                            
                            # Voltage statistics
                            if 'Voltage_measured' in cycle_data.dtype.names:
                                v = cycle_data['Voltage_measured'].flatten()
                                voltages_mean.append(np.mean(v))
                                voltages_min.append(np.min(v))
                                voltages_max.append(np.max(v))
                            else:
                                voltages_mean.append(3.7)
                                voltages_min.append(2.7)
                                voltages_max.append(4.2)
            
            if len(capacities) > 0:
                return BatteryTestData(
                    battery_id=battery_key,
                    cycle_numbers=np.array(cycle_nums),
                    capacities=np.array(capacities),
                    temperatures=np.array(temperatures),
                    discharge_times=np.array(discharge_times),
                    voltages_mean=np.array(voltages_mean),
                    voltages_min=np.array(voltages_min),
                    voltages_max=np.array(voltages_max)
                )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
        
        return None
    
    def load_discharge_curves(self, filepath: str, max_curves: int = 10) -> List[DischargeCurve]:
        """Load detailed discharge curves from a battery file"""
        curves = []
        try:
            data = loadmat(filepath)
            battery_keys = [k for k in data.keys() if k.startswith('B') and not k.startswith('__')]
            if not battery_keys:
                return curves
            
            battery_key = battery_keys[0]
            battery_data = data[battery_key]
            cycles = battery_data['cycle'][0, 0]
            
            cycle_count = 0
            for i in range(cycles.shape[1]):
                if len(curves) >= max_curves:
                    break
                    
                cycle = cycles[0, i]
                cycle_type = cycle['type'][0]
                
                if cycle_type == 'discharge':
                    cycle_data = cycle['data'][0, 0]
                    
                    if ('Capacity' in cycle_data.dtype.names and 
                        'Time' in cycle_data.dtype.names and
                        'Voltage_measured' in cycle_data.dtype.names and
                        'Current_measured' in cycle_data.dtype.names):
                        
                        cap = cycle_data['Capacity'][0, 0]
                        if cap > 0:
                            cycle_count += 1
                            
                            # Only sample some curves (beginning, middle, end)
                            if cycle_count in [1, 2, 3] or cycle_count % 50 == 0:
                                curves.append(DischargeCurve(
                                    battery_id=battery_key,
                                    cycle_number=cycle_count,
                                    time=cycle_data['Time'].flatten(),
                                    voltage=cycle_data['Voltage_measured'].flatten(),
                                    current=cycle_data['Current_measured'].flatten(),
                                    temperature=cycle_data['Temperature_measured'].flatten() if 'Temperature_measured' in cycle_data.dtype.names else np.full_like(cycle_data['Time'].flatten(), 25.0),
                                    capacity=cap
                                ))
        except Exception as e:
            print(f"Error loading curves from {filepath}: {e}")
        
        return curves
    
    def load_all_data(self, min_cycles: int = 20, require_positive_fade: bool = True):
        """Load all battery data from the dataset"""
        mat_files = self.find_all_mat_files()
        
        for filepath in mat_files:
            battery_data = self.load_battery_file(filepath)
            
            if battery_data is not None:
                # Filter based on criteria
                if battery_data.n_cycles >= min_cycles:
                    if not require_positive_fade or battery_data.capacity_fade_rate > 0:
                        # Avoid duplicates
                        if battery_data.battery_id not in self.battery_data:
                            self.battery_data[battery_data.battery_id] = battery_data
                            
                            # Load discharge curves for validation
                            curves = self.load_discharge_curves(filepath)
                            if curves:
                                self.discharge_curves[battery_data.battery_id] = curves
        
        print(f"Loaded {len(self.battery_data)} batteries from NASA dataset")
        return self
    
    def estimate_capacity_fade_rate(self) -> Dict:
        """
        Estimate capacity fade rate from all loaded batteries
        Uses linear regression on capacity vs cycle number
        """
        all_fade_rates = []
        fit_results = []
        
        for battery_id, data in self.battery_data.items():
            if data.n_cycles >= 50 and data.initial_capacity > 0.5:
                # Fit linear model: C(n) = C0 * (1 - gamma * n)
                def capacity_model(n, C0, gamma):
                    return C0 * (1 - gamma * n)
                
                try:
                    popt, pcov = curve_fit(
                        capacity_model,
                        data.cycle_numbers,
                        data.capacities,
                        p0=[data.initial_capacity, 0.002],
                        bounds=([0, 0], [10, 1])
                    )
                    
                    C0_fit, gamma_fit = popt
                    
                    # Calculate R² for goodness of fit
                    pred = capacity_model(data.cycle_numbers, C0_fit, gamma_fit)
                    ss_res = np.sum((data.capacities - pred) ** 2)
                    ss_tot = np.sum((data.capacities - np.mean(data.capacities)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    fit_results.append({
                        'battery_id': battery_id,
                        'C0': C0_fit,
                        'gamma': gamma_fit,
                        'r_squared': r_squared,
                        'n_cycles': data.n_cycles
                    })
                    
                    if r_squared > 0.5:  # Only use good fits
                        all_fade_rates.append(gamma_fit)
                        
                except Exception as e:
                    pass
        
        return {
            'mean_gamma': np.mean(all_fade_rates) if all_fade_rates else 0.0,
            'std_gamma': np.std(all_fade_rates) if all_fade_rates else 0.0,
            'min_gamma': np.min(all_fade_rates) if all_fade_rates else 0.0,
            'max_gamma': np.max(all_fade_rates) if all_fade_rates else 0.0,
            'n_batteries': len(all_fade_rates),
            'fit_results': fit_results
        }
    
    def estimate_voltage_parameters(self) -> Dict:
        """Estimate voltage-related parameters"""
        all_mean_v = []
        all_min_v = []
        all_max_v = []
        
        for battery_id, data in self.battery_data.items():
            all_mean_v.extend(data.voltages_mean.tolist())
            all_min_v.extend(data.voltages_min.tolist())
            all_max_v.extend(data.voltages_max.tolist())
        
        return {
            'V_nominal': np.mean(all_mean_v) if all_mean_v else 3.7,
            'V_min': np.mean(all_min_v) if all_min_v else 2.7,
            'V_max': np.mean(all_max_v) if all_max_v else 4.2,
            'V_range': [np.min(all_min_v), np.max(all_max_v)] if all_min_v else [2.7, 4.2]
        }
    
    def estimate_temperature_effects(self) -> Dict:
        """Analyze temperature effects on capacity"""
        temp_capacity_data = []
        
        for battery_id, data in self.battery_data.items():
            for i in range(len(data.capacities)):
                temp_capacity_data.append({
                    'temperature': data.temperatures[i],
                    'capacity': data.capacities[i],
                    'relative_capacity': data.capacities[i] / data.initial_capacity
                })
        
        # Group by temperature ranges
        temp_groups = {
            'cold': [d for d in temp_capacity_data if d['temperature'] < 20],
            'room': [d for d in temp_capacity_data if 20 <= d['temperature'] <= 30],
            'warm': [d for d in temp_capacity_data if d['temperature'] > 30]
        }
        
        results = {}
        for group_name, group_data in temp_groups.items():
            if group_data:
                temps = [d['temperature'] for d in group_data]
                caps = [d['relative_capacity'] for d in group_data]
                results[group_name] = {
                    'mean_temp': np.mean(temps),
                    'mean_relative_capacity': np.mean(caps),
                    'n_samples': len(group_data)
                }
        
        return results
    
    def get_model_parameters(self) -> Dict:
        """
        Get all estimated parameters for use in the battery model
        """
        fade_params = self.estimate_capacity_fade_rate()
        voltage_params = self.estimate_voltage_parameters()
        temp_params = self.estimate_temperature_effects()
        
        # Calculate average capacity
        all_capacities = []
        for data in self.battery_data.values():
            all_capacities.append(data.initial_capacity)
        
        # The test batteries are ~2Ah cells; smartphones typically use 3-5Ah
        # Scale factor: smartphone battery is ~2x larger
        avg_test_capacity = np.mean(all_capacities) if all_capacities else 2.0
        
        return {
            'capacity_fade_rate': fade_params['mean_gamma'],
            'capacity_fade_rate_std': fade_params['std_gamma'],
            'V_nominal': voltage_params['V_nominal'],
            'V_min': voltage_params['V_min'],
            'V_max': voltage_params['V_max'],
            'test_battery_capacity_ah': avg_test_capacity,
            'recommended_smartphone_capacity_mah': avg_test_capacity * 2 * 1000,  # Scale to smartphone
            'temperature_effects': temp_params,
            'n_batteries_analyzed': len(self.battery_data),
            'fade_rate_fit_details': fade_params['fit_results']
        }
    
    def get_validation_discharge_curves(self) -> List[DischargeCurve]:
        """Get discharge curves for model validation"""
        all_curves = []
        for curves in self.discharge_curves.values():
            all_curves.extend(curves)
        return all_curves
    
    def plot_capacity_fade(self, save_path: str = 'pictures/nasa_capacity_fade.png'):
        """Plot capacity fade curves from all batteries"""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.battery_data)))
        
        for (battery_id, data), color in zip(self.battery_data.items(), colors):
            # Normalize to percentage of initial capacity
            relative_capacity = data.capacities / data.initial_capacity * 100
            plt.plot(data.cycle_numbers, relative_capacity, 
                    label=battery_id, color=color, alpha=0.7, linewidth=1.5)
        
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Capacity Retention (%)', fontsize=12)
        plt.title('Battery Capacity Fade from NASA Aging Dataset', fontsize=14)
        plt.legend(loc='lower left', fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.ylim(50, 105)
        
        # Add 80% threshold line
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% EOL threshold')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved capacity fade plot to {save_path}")
    
    def plot_discharge_curves(self, battery_id: str = None, 
                              save_path: str = 'pictures/nasa_discharge_curves.png'):
        """Plot discharge voltage curves showing aging effect"""
        if battery_id is None:
            # Use first battery with curves
            for bid, curves in self.discharge_curves.items():
                if len(curves) > 2:
                    battery_id = bid
                    break
        
        if battery_id not in self.discharge_curves:
            print(f"No discharge curves for battery {battery_id}")
            return
        
        curves = self.discharge_curves[battery_id]
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(curves)))
        
        for curve, color in zip(curves, colors):
            time_hours, soc = curve.get_soc_curve()
            plt.plot(time_hours, soc * 100, 
                    label=f'Cycle {curve.cycle_number} (Cap: {curve.capacity:.2f} Ah)',
                    color=color, linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('State of Charge (%)', fontsize=12)
        plt.title(f'Discharge Curves for Battery {battery_id} - NASA Dataset\n(Showing capacity fade with cycling)', 
                 fontsize=14)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved discharge curves to {save_path}")


def load_nasa_data():
    """Convenience function to load NASA battery data"""
    loader = NASABatteryDataLoader()
    loader.load_all_data(min_cycles=20)
    return loader


if __name__ == "__main__":
    print("="*70)
    print("NASA Battery Data Analysis")
    print("="*70)
    
    # Load data
    loader = NASABatteryDataLoader()
    loader.load_all_data(min_cycles=20)
    
    # Print battery summary
    print("\nLoaded Batteries:")
    print("-"*70)
    for battery_id, data in sorted(loader.battery_data.items()):
        print(f"{battery_id}: {data.n_cycles:3d} cycles, "
              f"Capacity: {data.initial_capacity:.3f} -> {data.final_capacity:.3f} Ah "
              f"({data.capacity_retention:.1f}% retention), "
              f"Fade: {data.capacity_fade_rate*100:.4f}%/cycle")
    
    # Estimate parameters
    print("\n" + "="*70)
    print("ESTIMATED MODEL PARAMETERS")
    print("="*70)
    
    params = loader.get_model_parameters()
    
    print(f"\n1. CAPACITY FADE RATE:")
    print(f"   γ (mean) = {params['capacity_fade_rate']:.6f} ({params['capacity_fade_rate']*100:.4f}% per cycle)")
    print(f"   γ (std)  = {params['capacity_fade_rate_std']:.6f}")
    
    print(f"\n2. VOLTAGE PARAMETERS:")
    print(f"   V_nominal = {params['V_nominal']:.2f} V")
    print(f"   V_min = {params['V_min']:.2f} V")
    print(f"   V_max = {params['V_max']:.2f} V")
    
    print(f"\n3. CAPACITY (scaled for smartphone):")
    print(f"   Recommended = {params['recommended_smartphone_capacity_mah']:.0f} mAh")
    
    print(f"\n4. TEMPERATURE EFFECTS:")
    for group, effect in params['temperature_effects'].items():
        if effect:
            print(f"   {group}: T_avg={effect['mean_temp']:.1f}°C, "
                  f"Relative capacity={effect['mean_relative_capacity']*100:.1f}%")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    loader.plot_capacity_fade()
    loader.plot_discharge_curves()
    
    print("\nDone!")
