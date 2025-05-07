import streamlit as st
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn is imported but not used, keep for potential future plots

# --- Initialization and Configuration ---

# Must be the first Streamlit command
st.set_page_config(
    page_title="Deal Pricing Calculator",
    layout="wide",
    initial_sidebar_state="expanded",  # Keep sidebar open initially
    menu_items={
        'About': "A calculator for determining minimum deal pricing based on risk, cash flow, and other factors."
    }
)

# Initialize session state variables only if they don't exist
defaults = {
    'consider_vat': True,
    'include_admin_fee': True,
    'admin_fee_rate': 0.015,  # Store as rate (0.015), display as %
    'repayment_type': "monthly",
    'hk': 2.0,
    'alpha': 0.55,
    'psi': 0.04,
    'gamma': 0.95,
    'floor': 1.80,
    'risk_tier': "r2",
    'cash_bucket': "Normal",
    'is_micro': False,
    'principal': 100000.0,
    'tenure': 3,
    'custom_rate': None,  # For IRR calculator
    'ae_markup': 0.5  # 0.5% markup for AE pricing
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Check for admin access via query parameter
# Use st.query_params for the modern way to access query parameters
query_params = st.query_params.to_dict()
is_admin = query_params.get('access', '') == 'admin'

# --- Calculation Functions ---

def calculate_irr(principal, monthly_rate, tenure, admin_fee_rate=0.015, consider_vat=True, repayment_type='monthly'):
    """Calculate IRR (Annualized Percentage Rate)."""
    if principal <= 0 or tenure <= 0:
        return 0.0  # Avoid division by zero or meaningless calculation

    base_amount = principal
    # Note: The formula calculates total fee based on *initial* monthly rate * tenure
    total_monthly_fee = base_amount * monthly_rate * tenure
    admin_fee = base_amount * admin_fee_rate if admin_fee_rate > 0 else 0

    vat_rate = 0.15 if consider_vat else 0.0

    # Apply VAT reduction to fees *before* distribution
    monthly_fee_after_vat = total_monthly_fee * (1 - vat_rate)
    admin_fee_after_vat = admin_fee * (1 - vat_rate)

    total_income_after_vat = monthly_fee_after_vat + admin_fee_after_vat
    total_repayment_target = base_amount + total_income_after_vat # Total expected back from borrower

    cash_flows = [-principal]  # Initial outflow (t=0)

    # Repayment Calculations (t=1 to t=tenure)
    if repayment_type == 'monthly':
        monthly_principal = base_amount / tenure
        monthly_profit_component = monthly_fee_after_vat / tenure # Distribute VAT-adjusted fee evenly
        
        # Distribute admin fee (if any) over the first payment
        first_payment = monthly_principal + monthly_profit_component + admin_fee_after_vat
        regular_payment = monthly_principal + monthly_profit_component

        cash_flows.append(first_payment)
        for _ in range(tenure - 1):
            cash_flows.append(regular_payment)

    elif repayment_type == 'bullet':
        # No payments until the end
        for _ in range(tenure - 1):
            cash_flows.append(0)
        # Final payment includes principal + all fees (VAT adjusted)
        cash_flows.append(total_repayment_target) # Principal + Total VAT-adjusted income

    elif repayment_type == 'bi_monthly':
        # Payments every 2 months, or at the end if tenure is odd
        num_full_periods = tenure // 2
        has_remainder = tenure % 2 != 0
        num_payments = num_full_periods + (1 if has_remainder else 0)

        if num_payments > 0:
            payment_principal = base_amount / num_payments
            payment_profit_component = monthly_fee_after_vat / num_payments
            payment_admin_component = admin_fee_after_vat / num_payments # Distribute admin fee across payments
            
            regular_bi_monthly_payment = payment_principal + payment_profit_component + payment_admin_component
            
            payment_idx = 0
            for i in range(1, tenure + 1):
                is_payment_month = (i % 2 == 0) or (i == tenure and has_remainder)
                if is_payment_month:
                    cash_flows.append(regular_bi_monthly_payment)
                    payment_idx += 1
                else:
                    cash_flows.append(0)
            # Ensure correct number of cashflows (t=0 + t=1..tenure)
            assert len(cash_flows) == tenure + 1, f"Expected {tenure+1} cashflows, got {len(cash_flows)}"
        else: # Should not happen if tenure > 0
             for _ in range(tenure): cash_flows.append(0)

    # --- IRR Calculation ---
    try:
        # Use numpy_financial for robustness if available, fallback to scipy/simple calc
        try:
            import numpy_financial as npf
            # npf.irr requires at least one positive and one negative value
            if any(cf > 0 for cf in cash_flows) and any(cf < 0 for cf in cash_flows):
                 monthly_irr = npf.irr(cash_flows)
                 # Check if IRR calculation was successful (doesn't return nan or inf)
                 if np.isfinite(monthly_irr):
                     annual_irr = (1 + monthly_irr) ** 12 - 1
                 else:
                     raise ValueError("IRR calculation failed") # Force fallback
            elif sum(cf for cf in cash_flows if cf > 0) > abs(principal):
                # Approximate if npf fails but profit exists
                annual_irr = ((total_repayment_target / principal) ** (12 / tenure)) - 1 if principal > 0 else 0
            else:
                annual_irr = -1 # Indicate loss if total repayments < principal
        except (ImportError, ValueError): # Fallback if numpy_financial not installed or fails
             # Define NPV function for scipy.optimize.newton
            def npv(rate, flows):
                if rate <= -1:  # Avoid domain error for (1+rate)
                    return float('inf')
                return sum(cf / (1 + rate)**i for i, cf in enumerate(flows))

            # Attempt to find IRR using Newton-Raphson method
            try:
                monthly_irr = optimize.newton(lambda r: npv(r, cash_flows), x0=0.1, tol=1e-6, maxiter=100)
                # Check if the result is valid
                if not np.isfinite(monthly_irr) or monthly_irr <= -1:
                     raise optimize.nonlin.NoConvergence('Calculation failed')
                annual_irr = (1 + monthly_irr) ** 12 - 1
            except (RuntimeError, optimize.nonlin.NoConvergence):
                # Further fallback: simple rate of return if optimization fails
                if principal > 0 and tenure > 0 and total_repayment_target > 0:
                    annual_irr = ((total_repayment_target / principal) ** (12 / tenure)) - 1
                else:
                    annual_irr = -1.0 if total_repayment_target < principal else 0.0 # Indicate loss or zero return

    except Exception: # Catch-all for any unexpected error during calculation
        # Provide a default value or signal an error (e.g., NaN)
        annual_irr = np.nan # Indicate error

    return annual_irr * 100 if np.isfinite(annual_irr) else np.nan # Return as percentage or NaN


def generate_matrices(hk, alpha, psi, gamma, floor, current_principal, current_admin_fee_rate, current_consider_vat, current_repayment_type):
    """Generate Price and IRR matrices based on current parameters."""
    records = []
    tenures = range(1, 7)  # 1-6 months
    risk_multipliers = {"r1": 0.75, "r2": 1.0, "r3": 1.25}
    cash_percentages = {"Tight": 5, "Normal": 10, "Flush": 22, "Excess": 35}

    # Use a fixed principal for matrix generation for consistency,
    # but use current settings for IRR calculation details
    matrix_principal = 100000.0 # Or use current_principal if preferred

    for t in tenures:
        for r in ["r1", "r2", "r3"]:
            for m in [False, True]:  # Core (False), then Micro (True)
                for c in ["Tight", "Normal", "Flush", "Excess"]:
                    R = risk_multipliers[r]
                    F = cash_percentages[c]
                    I = 1 if m else 0
                    P_raw = hk + alpha * R - psi * F + gamma * I
                    P = max(P_raw, floor) # The price % per month

                    # Calculate IRR for this specific matrix cell configuration
                    irr = calculate_irr(matrix_principal, P/100, t,
                                      admin_fee_rate=current_admin_fee_rate,
                                      consider_vat=current_consider_vat,
                                      repayment_type=current_repayment_type)
                    records.append({
                        "Tenure": t,
                        "Risk": r,
                        "Type": "Micro" if m else "Core",
                        "Cash bucket": c,
                        "Min price %/mo": round(P, 3),
                        "IRR %": irr # Keep precision for now, round later
                    })

    df = pd.DataFrame(records)
    # Sort for consistent matrix layout
    df['Risk_order'] = df['Risk'].map({'r1': 0, 'r2': 1, 'r3': 2})
    df['Type_order'] = df['Type'].map({'Core': 0, 'Micro': 1})
    df['Cash_order'] = df['Cash bucket'].map({'Tight': 0, 'Normal': 1, 'Flush': 2, 'Excess': 3})
    df = df.sort_values(['Tenure', 'Risk_order', 'Type_order', 'Cash_order']).drop(['Risk_order', 'Type_order', 'Cash_order'], axis=1)

    # --- Create Price Matrix (Tenure-Independent) ---
    # Pivot for the price matrix (doesn't depend on tenure)
    price_matrix_df = df[df['Tenure'] == 1].pivot_table( # Price is same for all tenures, pick one
        index=["Risk", "Type"],
        columns="Cash bucket",
        values="Min price %/mo",
        sort=False # Keep the sorted order
    )
    # Reorder columns manually if pivot doesn't respect sort
    price_matrix_df = price_matrix_df[["Tight", "Normal", "Flush", "Excess"]]

    # --- Create IRR Matrices (One per Tenure) ---
    irr_matrices_dict = {}
    for t in tenures:
        irr_matrix_df = df[df['Tenure'] == t].pivot_table(
            index=["Risk", "Type"],
            columns="Cash bucket",
            values="IRR %",
            sort=False # Keep the sorted order
        )
        # Reorder columns manually
        irr_matrix_df = irr_matrix_df[["Tight", "Normal", "Flush", "Excess"]]
        irr_matrices_dict[t] = irr_matrix_df

    return price_matrix_df, irr_matrices_dict


# --- Main Layout ---
st.image("https://datascienceplc.com/assets/img/logo-new.png", width=200)
st.title("📈 Deal Pricing Calculator")

# Create two main tabs
deal_tab, model_tab = st.tabs(["💼 Deal Calculator", "📊 Pricing Model"])

with deal_tab:
    # Create three main sections
    st.markdown("### 📋 Deal Setup")
    
    # Deal Information in a clean grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Principal & Type")
        principal = st.number_input(
            "Principal Amount",
            min_value=1000.0,
            max_value=10000000.0,
            value=st.session_state.principal,
            step=10000.0,
            key="principal",
            format="%.2f"
        )
        
        is_micro = st.checkbox(
            "Micro Deal",
            value=st.session_state.is_micro,
            key="is_micro",
            help="Check if this is a micro deal"
        )
    
    with col2:
        st.markdown("#### 🎯 Risk & Cash")
        risk_tier = st.selectbox(
            "Risk Tier",
            options=["r1", "r2", "r3"],
            index=["r1", "r2", "r3"].index(st.session_state.risk_tier),
            format_func=lambda x: f"{x} ({0.75 if x == 'r1' else 1.0 if x == 'r2' else 1.25}x)",
            key="risk_tier"
        )
        
        cash_bucket = st.selectbox(
            "Cash Bucket",
            options=["Tight", "Normal", "Flush", "Excess"],
            index=["Tight", "Normal", "Flush", "Excess"].index(st.session_state.cash_bucket),
            key="cash_bucket"
        )
    
    with col3:
        st.markdown("#### 🧾 Fee Settings")
        consider_vat = st.checkbox(
            "Include VAT (15%)",
            value=st.session_state.consider_vat,
            key="consider_vat",
            help="Apply 15% VAT reduction"
        )
        
        include_admin_fee = st.checkbox(
            "Include Admin Fee",
            value=st.session_state.include_admin_fee,
            key="include_admin_fee"
        )
        
        if include_admin_fee:
            admin_fee_perc = st.number_input(
                "Admin Fee %",
                min_value=0.0,
                max_value=5.0,
                value=st.session_state.admin_fee_rate * 100,
                step=0.1,
                format="%.1f"
            )
            st.session_state.admin_fee_rate = admin_fee_perc / 100
        else:
            st.session_state.admin_fee_rate = 0.0
    
    st.markdown("---")
    
    # Calculated Pricing and IRR Calculator side by side
    pricing_col, irr_col = st.columns([1, 1])
    
    with pricing_col:
        st.markdown("### 💰 Calculated Pricing")
        
        # Calculate prices
        hk_val = st.session_state.hk
        alpha_val = st.session_state.alpha
        psi_val = st.session_state.psi
        gamma_val = st.session_state.gamma
        floor_val = st.session_state.floor
        
        risk_multipliers = {"r1": 0.75, "r2": 1.0, "r3": 1.25}
        cash_percentages = {"Tight": 5, "Normal": 10, "Flush": 22, "Excess": 35}
        
        R_val = risk_multipliers[risk_tier]
        F_val = cash_percentages[cash_bucket]
        I_micro_val = 1 if is_micro else 0
        
        P_raw_calc = hk_val + alpha_val * R_val - psi_val * F_val + gamma_val * I_micro_val
        P_min_calc = max(P_raw_calc, floor_val)
        ae_price = P_min_calc + st.session_state.ae_markup
        
        # Display prices in a clean layout
        st.metric(
            "Base Price (%/month)",
            f"{P_min_calc:.3f}%",
            delta=f"{P_min_calc - P_raw_calc:+.3f}% from raw" if P_min_calc != P_raw_calc else None
        )
        
        p1, p2 = st.columns(2)
        with p1:
            st.metric(
                "AE Price (%/month)",
                f"{ae_price:.3f}%",
                delta=f"+{st.session_state.ae_markup:.1f}%"
            )
   
    with irr_col:
        st.markdown("### 📊 IRR Calculator")
        
        # Rate selection
        use_custom_rate = st.checkbox("Use Custom Rate", value=False)
        if use_custom_rate:
            rate_to_use = st.slider(
                "Monthly Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(P_min_calc),
                step=0.1,
                format="%.2f"
            )
        else:
            rate_to_use = P_min_calc
        
        # Tenure and repayment settings
        t1, t2 = st.columns(2)
        with t1:
            tenure = st.slider(
                "Tenure (months)",
                min_value=1,
                max_value=12,
                value=st.session_state.tenure,
                step=1,
                key="tenure"
            )
        
        with t2:
            repayment_type = st.selectbox(
                "Repayment Type",
                options=["monthly", "bi_monthly", "bullet"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="repayment_type"
            )
        
        # Calculate IRR and profit
        irr = calculate_irr(
            principal,
            rate_to_use / 100,
            tenure,
            st.session_state.admin_fee_rate,
            consider_vat,
            repayment_type
        )
        
        monthly_fee = principal * (rate_to_use / 100) * tenure
        admin_fee = principal * st.session_state.admin_fee_rate if include_admin_fee else 0
        
        if consider_vat:
            monthly_fee *= 0.85
            admin_fee *= 0.85
        
        total_profit = monthly_fee + admin_fee
        
        # Display results
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Annual IRR", f"{irr:.2f}%")
        with r2:
            st.metric("Total Profit", f"{total_profit:,.2f}")
    
    st.markdown("---")
    
    # Payment Schedule in an expander
    with st.expander("📅 View Payment Schedule", expanded=False):
        monthly_principal = principal / tenure
        monthly_fee_portion = monthly_fee / tenure
        
        schedule_data = []
        running_total = 0
        
        for month in range(tenure):
            if month == 0:
                payment = monthly_principal + monthly_fee_portion + admin_fee
            else:
                payment = monthly_principal + monthly_fee_portion
            
            running_total += payment
            
            schedule_data.append({
                "Month": month + 1,
                "Principal": monthly_principal,
                "Fee": monthly_fee_portion + (admin_fee if month == 0 else 0),
                "Total Payment": payment,
                "Running Total": running_total
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(
            schedule_df.style.format({
                "Principal": "{:,.2f}",
                "Fee": "{:,.2f}",
                "Total Payment": "{:,.2f}",
                "Running Total": "{:,.2f}"
            }),
            hide_index=True
        )

with model_tab:
    # Admin Settings if applicable
    if is_admin:
        with st.expander("🔒 Admin Settings", expanded=False):
            st.warning("⚠️ Modifying these values impacts all pricing calculations.")
            p_cols = st.columns(5)
            with p_cols[0]: st.number_input("Hk", min_value=1.0, max_value=3.0, value=st.session_state.hk, step=0.05, key="hk", help="Base rate constant.")
            with p_cols[1]: st.number_input("α (Risk Slope)", min_value=0.0, max_value=1.0, value=st.session_state.alpha, step=0.05, key="alpha", help="Sensitivity to risk multiplier.")
            with p_cols[2]: st.number_input("ψ (Cash Pressure)", min_value=0.0, max_value=0.1, value=st.session_state.psi, step=0.005, key="psi", help="Sensitivity to free cash ratio.")
            with p_cols[3]: st.number_input("γ (Micro Surcharge)", min_value=0.0, max_value=2.0, value=st.session_state.gamma, step=0.05, key="gamma", help="Additive factor for micro deals.")
            with p_cols[4]: st.number_input("Floor %", min_value=1.0, max_value=3.0, value=st.session_state.floor, step=0.05, key="floor", help="Minimum allowed price % per month.")
    
    # Model Explanation
    with st.expander("📖 Understanding the Pricing Model", expanded=True):
        st.markdown("""
        The minimum deal price is determined by several factors, aiming to balance risk, market conditions, and operational costs.
        """)
        st.markdown("#### Base Formula")
        st.latex(r"P_{\text{raw}} = H_k + \alpha\,M_r - \psi\,F + \gamma\,I_{\text{micro}}")
        st.markdown("#### Components")
        components_col1, components_col2 = st.columns(2)
        
        with components_col1:
            st.markdown("""
            - **Hk:** Base rate constant
            - **α:** Risk-slope coefficient
            - **Mr:** Risk multiplier
            - **ψ:** Cash-pressure coefficient
            """)
        
        with components_col2:
            st.markdown("""
            - **F:** Free-cash ratio %
            - **γ:** Micro-deal surcharge
            - **I_micro:** Micro deal indicator (0 or 1)
            """)
    st.markdown("### 📊 Pricing & IRR Analysis")
    
    # Generate matrices
    price_matrix, irr_matrices = generate_matrices(
        st.session_state.hk,
        st.session_state.alpha,
        st.session_state.psi,
        st.session_state.gamma,
        st.session_state.floor,
        principal,
        st.session_state.admin_fee_rate,
        consider_vat,
        repayment_type
    )
    
    # Display matrices side by side
    matrix_col1, matrix_col2 = st.columns([1, 1])
    
    with matrix_col1:
        st.markdown("#### Price Matrix (% / month)")
        st.dataframe(
            price_matrix.style.format("{:.3f}%")
            .highlight_max(axis=None, color='lightcoral')
            .highlight_min(axis=None, color='lightgreen'),
            height=300
        )
    
    with matrix_col2:
        st.markdown("#### IRR Matrix (%)")
        selected_tenure = st.select_slider(
            "Select Tenure",
            options=range(1, 7),
            value=1,
            format_func=lambda x: f"{x} Month{'s' if x > 1 else ''}"
        )
        
        st.dataframe(
            irr_matrices[selected_tenure].style.format("{:.2f}%", na_rep="N/A")
            .background_gradient(cmap='viridis', axis=None)
            .highlight_null(color='gray'),
            height=300
        )
    
    # Status text for IRR calculations
    status_parts = []
    if consider_vat: status_parts.append("includes 15% VAT reduction")
    if include_admin_fee and st.session_state.admin_fee_rate > 0: status_parts.append(f"includes {st.session_state.admin_fee_rate*100:.1f}% admin fee")
    status_parts.append(f"uses {repayment_type.replace('_', ' ')} repayment")
    st.caption(f"IRR calculations: {', '.join(status_parts)}")

# Footer
st.markdown("---")
st.caption("Deal Pricing Calculator v2.0 | Data Science PLC")