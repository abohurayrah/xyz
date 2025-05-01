import streamlit as st
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_irr(principal, monthly_rate, tenure, admin_fee_rate=0.015, repayment_type='monthly', consider_vat=True):
    """Calculate IRR using the logic from irr_calculator_app.py"""
    base_amount = principal
    monthly_fee = base_amount * monthly_rate * tenure
    admin_fee = base_amount * admin_fee_rate
    
    if consider_vat:
        # Apply 15% VAT reduction to fees
        vat_rate = 0.15
        monthly_fee_after_vat = monthly_fee * (1 - vat_rate)
        admin_fee_after_vat = admin_fee * (1 - vat_rate)
    else:
        monthly_fee_after_vat = monthly_fee
        admin_fee_after_vat = admin_fee
    
    total_income = monthly_fee_after_vat + admin_fee_after_vat
    total_sales = base_amount + total_income
    
    cash_flows = []
    cash_flows.append(-principal)
    
    if repayment_type == 'monthly':
        monthly_principal = base_amount / tenure
        monthly_profit = monthly_fee_after_vat / tenure
        
        for i in range(tenure):
            if i == 0:
                cash_flows.append(monthly_principal + monthly_profit + admin_fee_after_vat)
            else:
                cash_flows.append(monthly_principal + monthly_profit)
    
    try:
        def npv(rate, cash_flows):
            if rate <= -1:
                return float('inf')
            return sum(cf / (1 + rate) ** (i) for i, cf in enumerate(cash_flows))
        
        monthly_irr = optimize.newton(lambda r: npv(r, cash_flows), x0=0.1, tol=1e-6, maxiter=100)
        annual_irr = (1 + monthly_irr) ** 12 - 1 if monthly_irr > -1 else float('nan')
    except:
        if principal > 0 and tenure > 0:
            annual_irr = ((total_sales / principal) ** (12 / tenure)) - 1
        else:
            annual_irr = 0
    
    return annual_irr * 100  # Return as percentage

st.set_page_config(
    page_title="Deal Pricing Calculator",
    layout="wide"
)

st.title("Deal Pricing Calculator")

# About section with LaTeX formulas
st.markdown("## About This Calculator")
st.markdown("""
This calculator determines the minimum deal price based on risk tier, cash availability, and deal type. 
The pricing model accounts for a base rate (Hk), risk adjustments, cash deployment pressure, and micro-deal premiums.
The final price is the maximum of the calculated raw price and a floor value to ensure minimum profitability.
""")

# Create tabs for formula explanation
formula_tabs = st.tabs(["Base Formula", "Risk Multiplier", "Free Cash Ratio"])

with formula_tabs[0]:
    st.markdown("""
    ### Raw Price Formula
    The raw price is calculated as:
    """)
    st.latex(r"P_{\text{raw}} = H_k + \alpha\,M_r - \psi\,F + \gamma\,I_{\text{micro}}")
    st.markdown("""
    Where:
    - Hk: Base rate (H's constant)
    - α: Risk-slope coefficient
    - Mr: Risk multiplier
    - ψ: Cash-pressure coefficient
    - F: Free-cash ratio (%)
    - γ: Micro-deal surcharge
    - I_micro: 1 if micro ticket, 0 otherwise
    """)

with formula_tabs[1]:
    st.markdown("### Risk Multiplier")
    st.latex(r"""
    M_r = \begin{cases}
    0.75 & \text{if } r_1\\
    1.00 & \text{if } r_2\\
    1.25 & \text{if } r_3
    \end{cases}
    """)
    st.markdown("Risk multipliers adjust the price based on the risk tier of the deal.")

with formula_tabs[2]:
    st.markdown("### Free Cash Ratio")
    st.latex(r"F = 100 \times \frac{\text{Cash on hand}}{\text{AUM}_{\text{deployed}}}")
    st.markdown("The free cash ratio determines price adjustments based on available capital.")
    
    st.markdown("### Cash Bucket Definitions")
    st.markdown("""
    | Cash Bucket | Free Cash Range | Business Impact |
    |------------|-----------------|-----------------|
    | Tight | 0 – 5% | About to run dry – squeeze pricing up |
    | Normal | 5 – 15% | Business as usual |
    | Flush | 15 – 30% | Deploy or die – discount aggressively |
    | Excess | > 30% | You're hoarding – sell at cost |
    """)

st.markdown("---")

# Create main column for base parameters
col1 = st.container()

with col1:
    st.subheader("Base Parameters")
    params_cols = st.columns(5)
    
    with params_cols[0]:
        hk = st.number_input("H's Constant (Hk)", min_value=1.3, max_value=2.5, value=2.0, step=0.1,
                            help="Base rate (default: 2.0)")
    
    with params_cols[1]:
        alpha = st.number_input("Risk Slope (α)", min_value=0.0, max_value=1.0, value=0.55, step=0.05,
                               help="Risk adjustment factor (default: 0.55)")
    
    with params_cols[2]:
        psi = st.number_input("Cash-Pressure (ψ)", min_value=0.0, max_value=0.1, value=0.04, step=0.01,
                             help="Cash deployment pressure (default: 0.04)")
    
    with params_cols[3]:
        gamma = st.number_input("Micro Surcharge (γ)", min_value=0.0, max_value=2.0, value=0.95, step=0.05,
                               help="Additional charge for micro loans (default: 0.95)")
    
    with params_cols[4]:
        floor = st.number_input("Price Floor", min_value=1.0, max_value=3.0, value=1.80, step=0.05,
                               help="Minimum allowed price (default: 1.80)")

# Create a small button for advanced settings
if st.button("⚙️ Advanced Settings", help="Configure scenario and IRR parameters"):
    with st.expander("Advanced Settings", expanded=True):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("##### Scenario Parameters")
            risk_tier = st.selectbox(
                "Risk Tier",
                options=["r1", "r2", "r3"],
                format_func=lambda x: f"{x} ({0.75 if x == 'r1' else 1.0 if x == 'r2' else 1.25}x multiplier)"
            )
            
            cash_bucket = st.selectbox(
                "Cash Bucket",
                options=["Tight", "Normal", "Flush", "Excess"],
                format_func=lambda x: f"{x} ({5 if x == 'Tight' else 10 if x == 'Normal' else 22 if x == 'Flush' else 35}% free cash)"
            )
            
            is_micro = st.checkbox("Is Micro deal?", value=False)
            consider_vat = st.checkbox("Consider 15% VAT Impact", value=True, 
                help="When checked, IRR calculations will account for 15% VAT deduction from fees")
            include_admin_fee = st.checkbox("Include Admin Fee", value=True,
                help="When checked, a one-time admin fee will be included in IRR calculations")

        with adv_col2:
            st.markdown("##### IRR Parameters")
            principal = st.number_input("Principal Amount", min_value=1000.0, max_value=10000000.0, value=100000.0, step=10000.0)
            tenure = st.number_input("Selected Tenure (months)", min_value=1, max_value=36, value=1, step=1)
            if include_admin_fee:
                admin_fee_rate = st.number_input("Admin Fee Rate (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.1) / 100
            else:
                admin_fee_rate = 0.0
else:
    # Default values when advanced settings are hidden
    risk_tier = "r2"  # Default to middle risk tier
    cash_bucket = "Normal"  # Default to normal cash bucket
    is_micro = False  # Default to non-micro deal
    principal = 100000.0  # Default principal
    tenure = 1  # Default tenure
    admin_fee_rate = 0.015  # Default admin fee rate
    consider_vat = True  # Default to considering VAT
    include_admin_fee = True  # Default to including admin fee

# Calculate the price
risk_multipliers = {"r1": 0.75, "r2": 1.0, "r3": 1.25}
cash_percentages = {"Tight": 5, "Normal": 10, "Flush": 22, "Excess": 35}

R = risk_multipliers[risk_tier]
F = cash_percentages[cash_bucket]
I_micro = 1 if is_micro else 0

P_raw = hk + alpha * R - psi * F + gamma * I_micro
P_min = max(P_raw, floor)

# Calculate IRR for current scenario
current_irr = calculate_irr(principal, P_min/100, tenure, admin_fee_rate if include_admin_fee else 0.0, consider_vat=consider_vat)

# Generate matrices
st.markdown("---")
st.subheader("Pricing and IRR Matrices")

def generate_matrices():
    records = []
    tenures = range(1, 7)  # 1-6 months
    
    for t in tenures:
        for r in ["r1", "r2", "r3"]:
            for m in [False, True]:  # First core (False), then micro (True)
                for c in ["Tight", "Normal", "Flush", "Excess"]:
                    R = risk_multipliers[r]
                    F = cash_percentages[c]
                    I = 1 if m else 0
                    P = max(hk + alpha * R - psi * F + gamma * I, floor)
                    irr = calculate_irr(principal, P/100, t, admin_fee_rate if include_admin_fee else 0.0, consider_vat=consider_vat)
                    records.append({
                        "Tenure": t,
                        "Risk": r,
                        "Type": "Micro" if m else "Core",
                        "Cash bucket": c,
                        "Min price %/mo": round(P, 3),
                        "IRR %": round(irr, 2)
                    })
    
    df = pd.DataFrame(records)
    # Sort to ensure r1 core, r1 micro, r2 core, r2 micro, r3 core, r3 micro order
    df['Risk_order'] = df['Risk'].map({'r1': 0, 'r2': 1, 'r3': 2})
    df['Type_order'] = df['Type'].map({'Core': 0, 'Micro': 1})
    df = df.sort_values(['Tenure', 'Risk_order', 'Type_order']).drop(['Risk_order', 'Type_order'], axis=1)
    
    # Create price matrix
    price_matrix = df.pivot_table(
        index=["Risk", "Type"],
        columns="Cash bucket",
        values="Min price %/mo"
    )
    
    # Create IRR matrices for each tenure
    irr_matrices = {}
    for t in tenures:
        irr_matrix = df[df['Tenure'] == t].pivot_table(
            index=["Risk", "Type"],
            columns="Cash bucket",
            values="IRR %"
        )
        irr_matrices[t] = irr_matrix
    
    return price_matrix, irr_matrices

price_matrix, irr_matrices = generate_matrices()

# Display price matrix
st.write("Minimum Price Matrix (%/month)")
st.dataframe(price_matrix.style.format("{:.3f}"))

# Display IRR matrices for each tenure
irr_title_col, vat_toggle_col, admin_toggle_col = st.columns([2, 1, 1])
with irr_title_col:
    st.write("IRR Matrices (%/year) by Tenure")
with vat_toggle_col:
    consider_vat = st.toggle("Include 15% VAT", value=consider_vat,
        help="Toggle to see IRR values before/after 15% VAT deduction from fees")
with admin_toggle_col:
    include_admin_fee = st.toggle("Include Admin Fee", value=include_admin_fee,
        help="Toggle to include/exclude one-time admin fee in IRR calculations")

# Recalculate matrices with current settings
price_matrix, irr_matrices = generate_matrices()

# Show current calculation parameters
status_text = []
if consider_vat:
    status_text.append("after 15% VAT")
if include_admin_fee:
    status_text.append(f"including {admin_fee_rate*100:.1f}% admin fee")
elif not include_admin_fee:
    status_text.append("excluding admin fee")
st.caption(f"Values shown {', '.join(status_text)}")

tabs = st.tabs([f"{t} Month{'s' if t > 1 else ''}" for t in range(1, 7)])

for i, tab in enumerate(tabs, 1):
    with tab:
        st.dataframe(irr_matrices[i].style.format("{:.2f}"))

# 
