# -*- coding: utf-8 -*-
"""
MCM 2026 Problem A: Zenodo Dataset Analysis
Analysis of AndroWatts + Mendeley Battery Data for Data-Driven Battery Model

This module analyzes the combined dataset to extract:
1. Brightness-Display Power relationship
2. CPU Frequency-Power relationship  
3. Component power breakdown
4. Battery aging parameters (SOH, OCV curves)
5. Temperature effects

Data Sources:
- AndroWatts (Zenodo): https://zenodo.org/records/14314943
- Mendeley Battery Degradation: https://data.mendeley.com/datasets/v8k6bsr6tf/1
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import json

# Output directory for figures
FIGURES_DIR = "pictures"


@dataclass
class DatasetAnalysisResults:
    """Container for analysis results"""
    # Brightness-Power model
    brightness_slope: float
    brightness_intercept: float
    brightness_r2: float
    brightness_power_by_range: Dict[str, float]
    
    # CPU Frequency-Power model
    cpu_freq_exponent: float
    cpu_freq_coefficient: float
    cpu_freq_r2: float
    
    # Component breakdown
    component_percentages: Dict[str, float]
    component_mean_power_mw: Dict[str, float]
    
    # Battery aging states
    aging_states: Dict[str, Dict[str, float]]
    
    # Battery life vs aging (from ALL 36,000 rows)
    battery_life_vs_aging: Dict[str, Dict[str, float]]
    
    # Overall statistics
    total_power_mean_w: float
    total_power_std_w: float
    total_power_min_w: float
    total_power_max_w: float
    temperature_mean_c: float
    n_samples: int
    n_unique_tests: int


class ZenodoDataAnalyzer:
    """
    Analyzer for the combined AndroWatts + Mendeley dataset
    
    This class processes the master modeling table containing:
    - 36,000 rows (1,000 phone tests × 36 battery states)
    - 93 columns including power measurements, device state, and battery parameters
    """
    
    def __init__(self, data_path: str):
        """
        Initialize with path to the master modeling table CSV
        
        Args:
            data_path: Path to the master modeling table CSV file
                      (e.g., 'MCM2026A Battery Data Table: master_modeling_table.csv')
        """
        self.data_path = data_path
        self.df = None
        self.unique_tests = None
        self.results = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Get unique phone tests (avoid duplicates from battery state combinations)
        self.unique_tests = self.df.drop_duplicates(subset='phone_test_id')
        
        print(f"Total rows: {len(self.df)}")
        print(f"Unique phone tests: {len(self.unique_tests)}")
        print(f"Columns: {len(self.df.columns)}")
        
        return self.df
    
    def analyze_brightness_power(self) -> Tuple[float, float, float, Dict]:
        """
        Analyze brightness vs display power relationship
        
        Returns:
            slope, intercept, r_squared, power_by_brightness_range
        """
        print("\n" + "="*60)
        print("Analyzing Brightness-Display Power Relationship")
        print("="*60)
        
        df = self.unique_tests
        
        # Get brightness and display power
        brightness = df['Brightness'].values
        # Convert from μW to mW
        display_power_mw = df['Display_ENERGY_UW'].values / 1000
        
        # Filter out zero brightness for fitting
        mask = brightness > 0
        B = brightness[mask]
        P = display_power_mw[mask]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(B, P)
        r_squared = r_value ** 2
        
        print(f"Linear model: P_display(mW) = {slope:.2f} * B + {intercept:.2f}")
        print(f"R² = {r_squared:.4f}")
        
        # Group by brightness ranges
        df_copy = df.copy()
        df_copy['brightness_bin'] = pd.cut(
            df_copy['Brightness'], 
            bins=[0, 20, 40, 60, 80, 100],
            labels=['0-20', '21-40', '41-60', '61-80', '81-100']
        )
        
        power_by_range = {}
        grouped = df_copy.groupby('brightness_bin')['Display_ENERGY_UW'].mean()
        for label, power_uw in grouped.items():
            power_by_range[str(label)] = power_uw / 1000  # Convert to mW
            print(f"  Brightness {label}%: {power_uw/1000:.1f} mW ({power_uw/1e6:.2f} W)")
        
        return slope, intercept, r_squared, power_by_range
    
    def analyze_cpu_frequency_power(self) -> Tuple[float, float, float]:
        """
        Analyze CPU frequency vs power relationship
        
        Fits: P_cpu = k * f^n
        
        Returns:
            exponent, coefficient, r_squared
        """
        print("\n" + "="*60)
        print("Analyzing CPU Frequency-Power Relationship")
        print("="*60)
        
        df = self.unique_tests
        
        # Get CPU frequency (Big core) and total CPU power
        cpu_freq_ghz = df['CPU_BIG_FREQ_KHz'].values / 1e6  # Convert to GHz
        cpu_power_mw = (
            df['CPU_BIG_ENERGY_UW'].values + 
            df['CPU_MID_ENERGY_UW'].values + 
            df['CPU_LITTLE_ENERGY_UW'].values
        ) / 1000  # Convert to mW
        
        # Filter valid data
        mask = (cpu_freq_ghz > 0) & (cpu_power_mw > 0)
        freq = cpu_freq_ghz[mask]
        power = cpu_power_mw[mask]
        
        # Log-log linear regression to find power law exponent
        log_freq = np.log(freq)
        log_power = np.log(power)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_freq, log_power)
        coefficient = np.exp(intercept)
        exponent = slope
        r_squared = r_value ** 2
        
        print(f"Power law model: P_cpu(mW) = {coefficient:.2f} * f^{exponent:.2f}")
        print(f"R² (log-log) = {r_squared:.4f}")
        
        return exponent, coefficient, r_squared
    
    def analyze_component_breakdown(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Analyze power consumption breakdown by component
        
        Returns:
            percentage_dict, mean_power_dict (in mW)
        """
        print("\n" + "="*60)
        print("Analyzing Component Power Breakdown")
        print("="*60)
        
        df = self.unique_tests
        
        # Define components and their corresponding columns
        components = {
            'CPU (Big+Mid+Little)': lambda d: (
                d['CPU_BIG_ENERGY_UW'] + 
                d['CPU_MID_ENERGY_UW'] + 
                d['CPU_LITTLE_ENERGY_UW']
            ) / 1000,
            'Display': lambda d: d['Display_ENERGY_UW'] / 1000,
            'GPU': lambda d: d['GPU_ENERGY_UW'] / 1000,
            'GPU3D': lambda d: d['GPU3D_ENERGY_UW'] / 1000,
            'WLAN/BT': lambda d: d['WLANBT_ENERGY_UW'] / 1000,
            'Cellular': lambda d: d['CELLULAR_ENERGY_UW'] / 1000,
            'GPS': lambda d: d['GPS_ENERGY_UW'] / 1000,
            'Memory': lambda d: d['Memory_ENERGY_UW'] / 1000,
            'Camera': lambda d: d['Camera_ENERGY_UW'] / 1000,
            'Sensor': lambda d: d['Sensor_ENERGY_UW'] / 1000,
            'Infrastructure': lambda d: d['INFRASTRUCTURE_ENERGY_UW'] / 1000,
            'UFS (Disk)': lambda d: d['UFS(Disk)_ENERGY_UW'] / 1000,
        }
        
        total_power_mw = df['P_total_uW'].values / 1000
        
        percentages = {}
        mean_powers = {}
        
        print(f"\n{'Component':<25} {'Mean (mW)':>12} {'% of Total':>12}")
        print("-" * 50)
        
        for name, calc_func in components.items():
            power = calc_func(df).values
            mean_power = power.mean()
            pct = (power / total_power_mw * 100).mean()
            
            percentages[name] = pct
            mean_powers[name] = mean_power
            print(f"{name:<25} {mean_power:>12.1f} {pct:>11.1f}%")
        
        return percentages, mean_powers
    
    def analyze_battery_aging(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze battery aging states from Mendeley data
        
        Uses ALL 36,000 rows to extract battery aging parameters.
        
        Returns:
            Dict mapping aging state to {SOH, Q_full_Ah, ocv_coefficients}
        """
        print("\n" + "="*60)
        print("Analyzing Battery Aging States (using ALL 36,000 rows)")
        print("="*60)
        
        df = self.df  # Use full dataset, not unique_tests
        
        # Group by battery state
        aging_cols = ['battery_state_label', 'SOH', 'Q_full_Ah', 
                      'ocv_c0', 'ocv_c1', 'ocv_c2', 'ocv_c3', 'ocv_c4', 'ocv_c5']
        
        aging_data = df[aging_cols].drop_duplicates()
        
        aging_states = {}
        
        print(f"\n{'State':<12} {'SOH':>8} {'Q_full (Ah)':>12}")
        print("-" * 35)
        
        for _, row in aging_data.groupby('battery_state_label').first().iterrows():
            state = row.name
            aging_states[state] = {
                'SOH': row['SOH'],
                'Q_full_Ah': row['Q_full_Ah'],
                'ocv_c0': row['ocv_c0'],
                'ocv_c1': row['ocv_c1'],
                'ocv_c2': row['ocv_c2'],
                'ocv_c3': row['ocv_c3'],
                'ocv_c4': row['ocv_c4'],
                'ocv_c5': row['ocv_c5'],
            }
            print(f"{state:<12} {row['SOH']:>8.3f} {row['Q_full_Ah']:>12.2f}")
        
        return aging_states
    
    def analyze_battery_life_vs_aging(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze battery life (t_empty) vs aging state using ALL 36,000 rows
        
        This analysis uses the full Cartesian product of phone tests × battery states
        to understand how aging affects battery life across different usage patterns.
        
        Returns:
            Dict with t_empty statistics by aging state
        """
        print("\n" + "="*60)
        print("Analyzing Battery Life vs Aging (using ALL 36,000 rows)")
        print("="*60)
        
        df = self.df  # Use full dataset
        
        t_empty_stats = {}
        
        print(f"\n{'State':<12} {'Samples':>8} {'Mean t_empty':>14} {'Min':>10} {'Max':>10}")
        print("-" * 60)
        
        for state in ['new', 'slight', 'moderate', 'aged', 'old', 'eol']:
            state_data = df[df['battery_state_label'] == state]
            t_empty = state_data['t_empty_h_est']
            
            t_empty_stats[state] = {
                'n_samples': len(state_data),
                'mean_hours': t_empty.mean(),
                'std_hours': t_empty.std(),
                'min_hours': t_empty.min(),
                'max_hours': t_empty.max(),
                'soh_mean': state_data['SOH'].mean(),
            }
            
            print(f"{state:<12} {len(state_data):>8} {t_empty.mean():>14.2f}h {t_empty.min():>10.2f}h {t_empty.max():>10.2f}h")
        
        # Calculate overall statistics for all 36000 rows
        print(f"\n{'TOTAL':<12} {len(df):>8} {df['t_empty_h_est'].mean():>14.2f}h {df['t_empty_h_est'].min():>10.2f}h {df['t_empty_h_est'].max():>10.2f}h")
        
        return t_empty_stats
    
    def analyze_overall_statistics(self) -> Dict[str, float]:
        """
        Compute overall dataset statistics
        """
        print("\n" + "="*60)
        print("Overall Dataset Statistics")
        print("="*60)
        
        df = self.unique_tests
        
        # Total power in Watts
        total_power_w = df['P_total_uW'].values / 1e6
        
        stats_dict = {
            'total_power_mean_w': total_power_w.mean(),
            'total_power_std_w': total_power_w.std(),
            'total_power_min_w': total_power_w.min(),
            'total_power_max_w': total_power_w.max(),
            'temperature_mean_c': df['temp_c'].mean(),
            'n_samples': len(self.df),
            'n_unique_tests': len(df),
        }
        
        print(f"Total Power (W): Mean={stats_dict['total_power_mean_w']:.2f}, "
              f"Std={stats_dict['total_power_std_w']:.2f}")
        print(f"Total Power Range (W): {stats_dict['total_power_min_w']:.2f} - "
              f"{stats_dict['total_power_max_w']:.2f}")
        print(f"Average Temperature: {stats_dict['temperature_mean_c']:.1f}°C")
        print(f"Number of samples: {stats_dict['n_samples']}")
        print(f"Number of unique tests: {stats_dict['n_unique_tests']}")
        
        return stats_dict
    
    def run_full_analysis(self) -> DatasetAnalysisResults:
        """
        Run complete analysis pipeline
        """
        if self.df is None:
            self.load_data()
        
        # Run all analyses
        b_slope, b_intercept, b_r2, power_by_range = self.analyze_brightness_power()
        cpu_exp, cpu_coef, cpu_r2 = self.analyze_cpu_frequency_power()
        pct_dict, mean_dict = self.analyze_component_breakdown()
        aging_states = self.analyze_battery_aging()
        battery_life_stats = self.analyze_battery_life_vs_aging()  # Uses ALL 36,000 rows
        overall = self.analyze_overall_statistics()
        
        self.results = DatasetAnalysisResults(
            brightness_slope=b_slope,
            brightness_intercept=b_intercept,
            brightness_r2=b_r2,
            brightness_power_by_range=power_by_range,
            cpu_freq_exponent=cpu_exp,
            cpu_freq_coefficient=cpu_coef,
            cpu_freq_r2=cpu_r2,
            component_percentages=pct_dict,
            component_mean_power_mw=mean_dict,
            aging_states=aging_states,
            battery_life_vs_aging=battery_life_stats,  # New field
            **overall
        )
        
        return self.results
    
    def generate_figures(self, output_dir: str = FIGURES_DIR):
        """
        Generate analysis figures for the paper
        """
        if self.df is None:
            self.load_data()
        if self.results is None:
            self.run_full_analysis()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
        self._plot_brightness_power(output_dir)
        self._plot_cpu_frequency_power(output_dir)
        self._plot_component_breakdown(output_dir)
        self._plot_aging_effects(output_dir)
        self._plot_power_distribution(output_dir)
        self._plot_battery_life_vs_aging(output_dir)  # New: uses ALL 36,000 rows
        
        print(f"\nFigures saved to {output_dir}/")
    
    def _plot_brightness_power(self, output_dir: str):
        """Plot brightness vs display power relationship"""
        df = self.unique_tests
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot with regression line
        ax1 = axes[0]
        brightness = df['Brightness'].values
        display_power = df['Display_ENERGY_UW'].values / 1000  # mW
        
        ax1.scatter(brightness, display_power, alpha=0.5, s=20, label='Data points')
        
        # Regression line
        x_line = np.linspace(0, 100, 100)
        y_line = self.results.brightness_slope * x_line + self.results.brightness_intercept
        ax1.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'Linear fit: P = {self.results.brightness_slope:.1f}B + {self.results.brightness_intercept:.0f}')
        
        ax1.set_xlabel('Brightness Level (0-100)')
        ax1.set_ylabel('Display Power (mW)')
        ax1.set_title(f'Brightness vs Display Power (R² = {self.results.brightness_r2:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar chart by brightness range
        ax2 = axes[1]
        ranges = list(self.results.brightness_power_by_range.keys())
        powers = [self.results.brightness_power_by_range[r] for r in ranges]
        
        bars = ax2.bar(ranges, powers, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Brightness Range (%)')
        ax2.set_ylabel('Average Display Power (mW)')
        ax2.set_title('Display Power by Brightness Range')
        
        # Add value labels
        for bar, val in zip(bars, powers):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zenodo_brightness_power.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/zenodo_brightness_power.png")
    
    def _plot_cpu_frequency_power(self, output_dir: str):
        """Plot CPU frequency vs power relationship"""
        df = self.unique_tests
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get data
        cpu_freq_ghz = df['CPU_BIG_FREQ_KHz'].values / 1e6
        cpu_power_mw = (
            df['CPU_BIG_ENERGY_UW'].values + 
            df['CPU_MID_ENERGY_UW'].values + 
            df['CPU_LITTLE_ENERGY_UW'].values
        ) / 1000
        
        mask = (cpu_freq_ghz > 0) & (cpu_power_mw > 0)
        freq = cpu_freq_ghz[mask]
        power = cpu_power_mw[mask]
        
        # Linear scale scatter
        ax1 = axes[0]
        ax1.scatter(freq, power, alpha=0.5, s=20, label='Data points')
        
        # Power law fit
        x_line = np.linspace(freq.min(), freq.max(), 100)
        y_line = self.results.cpu_freq_coefficient * (x_line ** self.results.cpu_freq_exponent)
        ax1.plot(x_line, y_line, 'r-', linewidth=2,
                label=f'Fit: P = {self.results.cpu_freq_coefficient:.0f} × f^{self.results.cpu_freq_exponent:.2f}')
        
        ax1.set_xlabel('CPU Frequency (GHz)')
        ax1.set_ylabel('CPU Power (mW)')
        ax1.set_title(f'CPU Frequency vs Power (R² = {self.results.cpu_freq_r2:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log-log scale
        ax2 = axes[1]
        ax2.scatter(freq, power, alpha=0.5, s=20, label='Data points')
        ax2.plot(x_line, y_line, 'r-', linewidth=2, label='Power law fit')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('CPU Frequency (GHz) [log scale]')
        ax2.set_ylabel('CPU Power (mW) [log scale]')
        ax2.set_title('CPU Frequency vs Power (Log-Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zenodo_cpu_frequency_power.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/zenodo_cpu_frequency_power.png")
    
    def _plot_component_breakdown(self, output_dir: str):
        """Plot component power breakdown"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        ax1 = axes[0]
        labels = list(self.results.component_percentages.keys())
        sizes = [self.results.component_percentages[l] for l in labels]
        
        # Sort by size
        sorted_data = sorted(zip(labels, sizes), key=lambda x: -x[1])
        labels, sizes = zip(*sorted_data)
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # Explode the largest slice
        explode = [0.05 if i == 0 else 0 for i in range(len(labels))]
        
        # Custom autopct function to hide labels for small slices (< 3%) to prevent overlap
        def make_autopct(values):
            def autopct(pct):
                return f'{pct:.1f}%' if pct >= 3 else ''
            return autopct
        
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=None, autopct=make_autopct(sizes), startangle=90,
            colors=colors, explode=explode, pctdistance=0.75
        )
        # Use legend instead of labels to avoid overlap
        ax1.legend(wedges, labels, title="Components", loc="center left",
                   bbox_to_anchor=(-0.3, 0.5), fontsize=8)
        ax1.set_title('Component Power Breakdown (% of Total)')
        
        # Bar chart
        ax2 = axes[1]
        y_pos = range(len(labels))
        bars = ax2.barh(y_pos, sizes, color='steelblue', edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Percentage of Total Power (%)')
        ax2.set_title('Component Power Contribution')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for bar, val in zip(bars, sizes):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zenodo_component_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/zenodo_component_breakdown.png")
    
    def _plot_aging_effects(self, output_dir: str):
        """Plot battery aging effects"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # SOH vs aging state
        ax1 = axes[0]
        states = ['new', 'slight', 'moderate', 'aged', 'old', 'eol']
        soh_values = [self.results.aging_states.get(s, {}).get('SOH', 0) for s in states]
        
        bars = ax1.bar(states, soh_values, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Aging State')
        ax1.set_ylabel('State of Health (SOH)')
        ax1.set_title('Battery SOH by Aging State')
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.8, color='r', linestyle='--', label='Replacement Threshold (80%)')
        ax1.legend()
        
        # Add value labels
        for bar, val in zip(bars, soh_values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # OCV curves for different aging states
        ax2 = axes[1]
        soc_range = np.linspace(0, 1, 100)
        
        for state in ['new', 'moderate', 'eol']:
            if state in self.results.aging_states:
                coeffs = self.results.aging_states[state]
                ocv = (coeffs['ocv_c0'] + 
                       coeffs['ocv_c1'] * soc_range +
                       coeffs['ocv_c2'] * soc_range**2 +
                       coeffs['ocv_c3'] * soc_range**3 +
                       coeffs['ocv_c4'] * soc_range**4 +
                       coeffs['ocv_c5'] * soc_range**5)
                ax2.plot(soc_range * 100, ocv, linewidth=2, label=f'{state.capitalize()} (SOH={coeffs["SOH"]:.2f})')
        
        ax2.set_xlabel('State of Charge (%)')
        ax2.set_ylabel('Open Circuit Voltage (V)')
        ax2.set_title('OCV(SOC) Curves at Different Aging States')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zenodo_aging_effects.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/zenodo_aging_effects.png")
    
    def _plot_power_distribution(self, output_dir: str):
        """Plot power distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        df = self.unique_tests
        total_power_w = df['P_total_uW'].values / 1e6
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(total_power_w, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.results.total_power_mean_w, color='r', linestyle='--', 
                   linewidth=2, label=f'Mean = {self.results.total_power_mean_w:.1f} W')
        ax1.set_xlabel('Total Power (W)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Total Power Consumption')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by brightness level
        ax2 = axes[1]
        df_copy = df.copy()
        df_copy['brightness_bin'] = pd.cut(
            df_copy['Brightness'], 
            bins=[0, 25, 50, 75, 100],
            labels=['0-25%', '26-50%', '51-75%', '76-100%']
        )
        df_copy['power_w'] = df_copy['P_total_uW'] / 1e6
        
        df_copy.boxplot(column='power_w', by='brightness_bin', ax=ax2)
        ax2.set_xlabel('Brightness Range')
        ax2.set_ylabel('Total Power (W)')
        ax2.set_title('Power Distribution by Brightness Level')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zenodo_power_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/zenodo_power_distribution.png")
    
    def _plot_battery_life_vs_aging(self, output_dir: str):
        """
        Plot battery life vs aging state using ALL 36,000 rows
        
        This visualization shows how estimated battery life varies with:
        1. Battery aging state (SOH)
        2. Usage patterns (1000 different power consumption profiles)
        """
        df = self.df  # Use ALL 36,000 rows
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot: t_empty by aging state
        ax1 = axes[0]
        states_order = ['new', 'slight', 'moderate', 'aged', 'old', 'eol']
        
        # Prepare data for box plot
        data_by_state = [df[df['battery_state_label'] == s]['t_empty_h_est'].values 
                         for s in states_order]
        
        bp = ax1.boxplot(data_by_state, labels=states_order, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(states_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Battery Aging State')
        ax1.set_ylabel('Estimated Battery Life (hours)')
        ax1.set_title(f'Battery Life Distribution by Aging State\n(N = {len(df):,} samples)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Scatter plot: t_empty vs SOH
        ax2 = axes[1]
        # Sample to avoid overplotting
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        scatter = ax2.scatter(sample_df['SOH'], sample_df['t_empty_h_est'], 
                              c=sample_df['P_total_uW']/1e6, cmap='viridis',
                              alpha=0.5, s=10)
        
        ax2.set_xlabel('State of Health (SOH)')
        ax2.set_ylabel('Estimated Battery Life (hours)')
        ax2.set_title(f'Battery Life vs SOH\n(sampled {sample_size:,} of {len(df):,} points)')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Total Power (W)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zenodo_battery_life_vs_aging.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/zenodo_battery_life_vs_aging.png (uses ALL {len(df):,} rows)")
    
    def export_results_json(self, output_path: str = "analysis_results.json"):
        """Export analysis results to JSON"""
        if self.results is None:
            self.run_full_analysis()
        
        results_dict = {
            'brightness_model': {
                'slope': self.results.brightness_slope,
                'intercept': self.results.brightness_intercept,
                'r_squared': self.results.brightness_r2,
                'power_by_range': self.results.brightness_power_by_range,
            },
            'cpu_frequency_model': {
                'exponent': self.results.cpu_freq_exponent,
                'coefficient': self.results.cpu_freq_coefficient,
                'r_squared': self.results.cpu_freq_r2,
            },
            'component_breakdown': {
                'percentages': self.results.component_percentages,
                'mean_power_mw': self.results.component_mean_power_mw,
            },
            'aging_states': self.results.aging_states,
            'battery_life_vs_aging': self.results.battery_life_vs_aging,  # From ALL 36,000 rows
            'overall_statistics': {
                'total_power_mean_w': self.results.total_power_mean_w,
                'total_power_std_w': self.results.total_power_std_w,
                'total_power_min_w': self.results.total_power_min_w,
                'total_power_max_w': self.results.total_power_max_w,
                'temperature_mean_c': self.results.temperature_mean_c,
                'n_samples': self.results.n_samples,
                'n_unique_tests': self.results.n_unique_tests,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults exported to: {output_path}")
        return results_dict


# Default data path - can be overridden via command line or function argument
DEFAULT_DATA_PATH = "requests/Zenodo Data Set/MCM2026A题锂电池数据表：master_modeling_table.csv"
# English description: MCM2026A Battery Data Table: master_modeling_table.csv


def main(data_path: str = None):
    """
    Main analysis function
    
    Args:
        data_path: Optional path to the CSV data file. 
                   Defaults to DEFAULT_DATA_PATH if not provided.
    """
    # Use default path if not specified
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Expected file: MCM2026A Battery Data Table (master_modeling_table.csv)")
        return None
    
    # Create analyzer and run analysis
    analyzer = ZenodoDataAnalyzer(data_path)
    results = analyzer.run_full_analysis()
    
    # Generate figures
    analyzer.generate_figures()
    
    # Export results
    analyzer.export_results_json()
    
    # Print summary for paper
    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)
    
    print("\n## Data-Driven Model Parameters")
    print(f"\n### Brightness-Display Power Model:")
    print(f"P_display(mW) = {results.brightness_slope:.2f} × Brightness + {results.brightness_intercept:.2f}")
    print(f"R² = {results.brightness_r2:.4f}")
    
    print(f"\n### CPU Frequency-Power Model:")
    print(f"P_cpu ∝ f^{results.cpu_freq_exponent:.2f}")
    print(f"R² = {results.cpu_freq_r2:.4f}")
    
    print(f"\n### Component Power Breakdown:")
    for comp, pct in sorted(results.component_percentages.items(), key=lambda x: -x[1]):
        print(f"  {comp}: {pct:.1f}%")
    
    print(f"\n### Overall Statistics:")
    print(f"  Total Power: Mean={results.total_power_mean_w:.1f}W, "
          f"Range={results.total_power_min_w:.1f}-{results.total_power_max_w:.1f}W")
    print(f"  Average Temperature: {results.temperature_mean_c:.1f}°C")
    print(f"  Sample Size: {results.n_unique_tests} unique tests")
    
    return results


if __name__ == "__main__":
    results = main()
