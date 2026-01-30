# Battery Model Validation Report

## Overview

This report validates our continuous-time battery model against synthetic datasets 
based on published smartphone power consumption studies.

## Validation Metrics

| Dataset | RMSE (%) | MAE (%) | Correlation | TTE Error (h) |
|---------|----------|---------|-------------|---------------|
| Idle Standby Profile | 15.59 | 13.44 | 0.9995 | 11.97 |
| Light Usage Profile | 14.75 | 12.81 | 0.9993 | 4.03 |
| Heavy Gaming Profile | 4.49 | 3.03 | 0.9965 | 0.33 |
| Navigation Profile | 8.39 | 7.16 | 0.9982 | 0.50 |
| Video Streaming Profile | 4.99 | 4.14 | 0.9992 | 0.53 |
| Cold Weather Profile | 6.91 | 5.74 | 0.9983 | 1.15 |

## Summary

- **Average RMSE**: 9.19%
- **Average MAE**: 7.72%  
- **Average Correlation**: 0.9985

## Interpretation

✅ **Good Performance**: Model RMSE below 10% is acceptable for practical use.
✅ **Excellent Fit**: Correlation above 0.95 shows model captures discharge dynamics well.

## Figures

- `model_validation.png`: Comparison of model predictions vs dataset for each scenario
- `validation_metrics.png`: RMSE and correlation summary across all scenarios

## Data Sources

For further validation with real-world data, the following public datasets are available:

1. **NASA Battery Dataset** - Li-ion aging data (https://data.nasa.gov/)
2. **CALCE Battery Data** - Discharge curves and cycle life (https://calce.umd.edu/battery-data)
3. **Mobile Phone Usage Dataset** - App usage and battery drain (Kaggle)
4. **CRAWDAD Wireless Traces** - Real-world usage patterns

## Conclusion

The model demonstrates acceptable performance across diverse usage scenarios, with particularly 
strong performance in Heavy Gaming Profile and room for improvement in Idle Standby Profile.
