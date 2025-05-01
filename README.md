# Loan Pricing Calculator

An interactive calculator for determining minimum loan prices based on various factors including risk tier, cash availability, and loan type.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run loan_pricing_calculator.py
```

## Features

- Interactive sliders and inputs for all pricing parameters
- Real-time calculation of minimum loan price
- Detailed breakdown of formula components
- Full pricing matrix for all combinations
- Support for:
  - Different risk tiers (r1, r2, r3)
  - Cash buckets (Tight, Normal, Flush, Excess)
  - Micro vs standard loans
  - Customizable base parameters

## Formula

The minimum loan price is calculated as:

```
P_raw = Hk + α·R - ψ·F + γ·I_micro
P_min = max(P_raw, floor)
```

Where:
- Hk: Hisham constant (base rate)
- α: Risk slope
- R: Risk multiplier (0.75 for r1, 1.0 for r2, 1.25 for r3)
- ψ: Cash-pressure coefficient
- F: Free cash percentage
- γ: Micro loan surcharge
- I_micro: 1 if micro loan, 0 otherwise
- floor: Minimum allowed price (default 1.80) 