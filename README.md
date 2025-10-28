# 003 Advanced Trading Strategies: Deep Learning

### Ejecutar:
1. Instala dependencias:
   ```bash
   pip install -r requirements.txt

Objectives
Engineer time series features from technical indicators at multiple timeframes
Design and train multiple deep learning architectures
Implement proper ML workflow tracking and experiment management with MLFlow
Monitor model data drift in production-like conditions
Develop backtesting systems integrating ML predictions with realistic trading costs
Create production-ready code and professional documentation
 
Key Concepts
Feature Engineering: Converting raw price data into predictive signals
Time Series Architectures: MLP, CNN, for sequential prediction
Class Imbalance: Handling skewed distribution of trading signals
Data Drift: Detecting when model input distributions shift over time
Model Versioning: Tracking and comparing multiple model experiments
Technical Requirements
 

Data Requirements
Historical Data: 15 years of daily price data
Asset Selection: Student's choice
Data Quality: Handle missing data appropriately
Data Splits: 60% Training, 20% Testing, 20% Validation (chronological, no look-ahead)
Frequency Note: Daily data will result in limited samples; address with class weighting
 

Feature Engineering
Minimum 20 features
Momentum, Volatility and Volume Indicators
Data normalization
 

Target Variable
Define appropriate labels for supervised learning
Remember that market tends toward "hold" (imbalanced distribution)
Apply class weights during training to penalize minority class errors, or adjust thresholds to create more balanced splits
 

Deep Learning Models
MLP: Baseline model, learns feature combinations without temporal structure
CNN: Capture local temporal patterns in feature sequences
: Learn long term dependencies and sequential patterns
 

MLFlow Implementation
Experiment Tracking: Compare all three model types and select the best model for final backtesting
 

Data Drift Modeling (notebook is fine)
Timeline View: For each feature, plot distribution over train/test/validation periods
Drift Statistics Table: Feature name, KS-test p-value, Drift detected
Highlighting: Clearly mark periods/features with significant drift
Summary: Top 5 most-drifted features with explanations
Interpretation may be market regime changes, volatility spikes, etc.

 

Backtesting Implementation
Signal Generation
Strategy Parameters: SL, TP, N shares
Strategy Constants:
Commission = 0.125%
Borrow Rate = 0.25% (annualized)
