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
    'tenure': 3  # Default tenure to 3 months for more interesting initial IRR
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


# --- Sidebar for User Inputs ---
st.sidebar.image("https://datascienceplc.com/assets/img/logo-new.png", width=200) # Placeholder Logo
st.sidebar.title("⚙️ Deal Configuration")
st.sidebar.markdown("Adjust the parameters for your specific deal.")

st.sidebar.subheader("Deal Specifics")
# Use session state keys for widgets to preserve state across reruns
principal_input = st.sidebar.number_input(
    "💰 Principal Amount",
    min_value=1000.0, max_value=10000000.0,
    value=st.session_state.principal, step=10000.0,
    key="principal", # Link to session state
    format="%.2f"
)
tenure_input = st.sidebar.number_input(
    "⏳ Selected Tenure (months)",
    min_value=1, max_value=36,
    value=st.session_state.tenure, step=1,
    key="tenure" # Link to session state
)
is_micro_input = st.sidebar.checkbox("🐜 Is Micro deal?",
    value=st.session_state.is_micro,
    key="is_micro", # Link to session state
    help="Check if the deal principal is considered 'micro'."
)

st.sidebar.divider()

st.sidebar.subheader("Risk & Cash Context")
risk_tier_input = st.sidebar.selectbox(
    "🚦 Risk Tier",
    options=["r1", "r2", "r3"],
    index=["r1", "r2", "r3"].index(st.session_state.risk_tier),
    format_func=lambda x: f"{x} ({0.75 if x == 'r1' else 1.0 if x == 'r2' else 1.25}x Risk Multiplier)",
    key="risk_tier", # Link to session state
    help="Select the assessed risk category for the deal."
)
cash_bucket_input = st.sidebar.selectbox(
    "💧 Cash Bucket",
    options=["Tight", "Normal", "Flush", "Excess"],
    index=["Tight", "Normal", "Flush", "Excess"].index(st.session_state.cash_bucket),
    format_func=lambda x: f"{x} ({'0-5' if x == 'Tight' else '5-15' if x == 'Normal' else '15-30' if x == 'Flush' else '>30'}% Free Cash)",
    key="cash_bucket", # Link to session state
    help="Select the current cash availability situation."
)

st.sidebar.divider()

st.sidebar.subheader("IRR Calculation Options")
consider_vat_input = st.sidebar.toggle("📉 Include 15% VAT",
    value=st.session_state.consider_vat,
    key="consider_vat", # Link to session state
    help="Apply a 15% VAT reduction to calculated fees before IRR computation."
)
include_admin_fee_input = st.sidebar.toggle("🧾 Include Admin Fee",
    value=st.session_state.include_admin_fee,
    key="include_admin_fee", # Link to session state
    help="Include a one-time admin fee in the first repayment cash flow."
)

admin_fee_rate_input = 1.5 # Default if not included
if include_admin_fee_input:
    # Input as percentage, store as rate
    admin_fee_perc = st.sidebar.number_input("Admin Fee Rate (%)",
        min_value=0.0, max_value=10.0,
        value=st.session_state.admin_fee_rate * 100, # Display as %
        step=0.1, format="%.1f",
        key="admin_fee_rate_perc", # Use a different key for the widget if needed, or update session state directly
        help="One-time fee charged as a percentage of the principal."
    )
    # Update the session state rate value based on the percentage input
    st.session_state.admin_fee_rate = admin_fee_perc / 100
    admin_fee_rate_input = st.session_state.admin_fee_rate # Use the updated rate
else:
    # Ensure the rate is zero if the toggle is off
    st.session_state.admin_fee_rate = 0.0
    admin_fee_rate_input = 0.0


repayment_type_input = st.sidebar.selectbox(
    "🗓️ Repayment Type",
    options=["monthly", "bi_monthly", "bullet"],
    index=["monthly", "bi_monthly", "bullet"].index(st.session_state.repayment_type),
    format_func=lambda x: x.replace('_', ' ').title(), # Nicer display names
    key="repayment_type", # Link to session state
    help="Select how the principal and fees are repaid over the tenure."
)


# --- Main Page Layout ---

st.title("📈 Deal Pricing & IRR Calculator")
st.markdown("An interactive tool to determine minimum pricing and analyze potential IRR based on deal parameters.")
st.divider()

# --- Admin Settings (Conditional) ---
if is_admin:
    with st.expander("🔒 Admin Settings: Core Pricing Parameters", expanded=False):
        st.warning("⚠️ Caution: Modifying these values impacts all pricing calculations.")
        p_cols = st.columns(5)
        with p_cols[0]: st.number_input("Hk", min_value=1.0, max_value=3.0, value=st.session_state.hk, step=0.05, key="hk", help="Base rate constant.")
        with p_cols[1]: st.number_input("α (Risk Slope)", min_value=0.0, max_value=1.0, value=st.session_state.alpha, step=0.05, key="alpha", help="Sensitivity to risk multiplier.")
        with p_cols[2]: st.number_input("ψ (Cash Pressure)", min_value=0.0, max_value=0.1, value=st.session_state.psi, step=0.005, key="psi", help="Sensitivity to free cash ratio.")
        with p_cols[3]: st.number_input("γ (Micro Surcharge)", min_value=0.0, max_value=2.0, value=st.session_state.gamma, step=0.05, key="gamma", help="Additive factor for micro deals.")
        with p_cols[4]: st.number_input("Floor %", min_value=1.0, max_value=3.0, value=st.session_state.floor, step=0.05, key="floor", help="Minimum allowed price % per month.")
        st.caption("Changes here are saved in the session state and affect calculations immediately.")
    st.divider()

# --- Use session state values for calculations (retrieved from widgets/admin panel) ---
hk_val = st.session_state.hk
alpha_val = st.session_state.alpha
psi_val = st.session_state.psi
gamma_val = st.session_state.gamma
floor_val = st.session_state.floor
# User inputs are already in session state via their keys

# --- Perform Calculations ---
risk_multipliers = {"r1": 0.75, "r2": 1.0, "r3": 1.25}
cash_percentages = {"Tight": 5, "Normal": 10, "Flush": 22, "Excess": 35}

R_val = risk_multipliers[st.session_state.risk_tier]
F_val = cash_percentages[st.session_state.cash_bucket]
I_micro_val = 1 if st.session_state.is_micro else 0

P_raw_calc = hk_val + alpha_val * R_val - psi_val * F_val + gamma_val * I_micro_val
P_min_calc = max(P_raw_calc, floor_val)

# Calculate IRR for the *currently selected* scenario in the sidebar
current_irr_calc = calculate_irr(
    st.session_state.principal,
    P_min_calc / 100, # Pass rate as decimal
    st.session_state.tenure,
    admin_fee_rate=st.session_state.admin_fee_rate, # Use the potentially updated rate
    consider_vat=st.session_state.consider_vat,
    repayment_type=st.session_state.repayment_type
)

# --- Display Key Results ---
st.subheader("📊 Calculated Results for Current Deal")
res_cols = st.columns(3)
with res_cols[0]:
    st.metric(label="Minimum Price (% / month)", value=f"{P_min_calc:.3f}%")
with res_cols[1]:
    st.metric(label="Calculated Annual IRR", value=f"{current_irr_calc:.2f}%" if not np.isnan(current_irr_calc) else "N/A")
with res_cols[2]:
    delta_val = P_min_calc - P_raw_calc
    st.metric(label="Raw Calculated Price (% / month)", value=f"{P_raw_calc:.3f}%",
              delta=f"{delta_val:.3f}% vs Floor" if delta_val > 0.001 else None,
              help="Price before applying the minimum floor. Delta shows how much the floor increased the price.",
              delta_color="inverse") # Positive delta means floor was binding (higher price)

st.caption(f"Based on Principal: {st.session_state.principal:,.0f}, Tenure: {st.session_state.tenure}mo, "
           f"Risk: {st.session_state.risk_tier}, Cash: {st.session_state.cash_bucket}, Type: {'Micro' if st.session_state.is_micro else 'Core'}")

st.divider()

# --- Pricing Model Explanation ---
with st.expander("📖 Understanding the Pricing Model", expanded=False):
    st.markdown("""
    The minimum deal price is determined by several factors, aiming to balance risk, market conditions, and operational costs.
    """)
    formula_tabs = st.tabs(["Base Formula", "Risk Multiplier (Mr)", "Free Cash Ratio (F)", "Micro Deal (I_micro)"])
    with formula_tabs[0]:
        st.markdown("**Raw Price Formula:**")
        st.latex(r"P_{\text{raw}} = H_k + \alpha\,M_r - \psi\,F + \gamma\,I_{\text{micro}}")
        st.markdown("- **Hk:** Base rate (H's constant)\n"
                    "- **α:** Risk-slope coefficient\n"
                    "- **Mr:** Risk multiplier (see next tab)\n"
                    "- **ψ:** Cash-pressure coefficient\n"
                    "- **F:** Free-cash ratio % (see tab)\n"
                    "- **γ:** Micro-deal surcharge\n"
                    "- **I_micro:** Micro deal indicator (see tab)")
        st.markdown("**Final Price:**")
        st.latex(r"P_{\text{min}} = \max(P_{\text{raw}}, \text{Floor})")
        st.markdown("The final price is the higher of the raw calculated price and the predefined floor.")

    with formula_tabs[1]:
        st.markdown("**Risk Multiplier (Mr):** Adjusts price based on perceived risk.")
        st.latex(r"M_r = \begin{cases} 0.75 & \text{if Risk Tier = } r_1 \\ 1.00 & \text{if Risk Tier = } r_2 \\ 1.25 & \text{if Risk Tier = } r_3 \end{cases}")

    with formula_tabs[2]:
        st.markdown("**Free Cash Ratio (F %):** Reflects capital availability.")
        # st.latex(r"F = 100 \times \frac{\text{Cash on hand}}{\text{AUM}_{\text{deployed}}}") # Simplified explanation below
        st.markdown("""
        | Cash Bucket | Free Cash % (Approx) | Interpretation                 | Price Impact (via -ψF) |
        |-------------|----------------------|--------------------------------|------------------------|
        | Tight       | 0 – 5%               | Need funds, price increases    | Highest Price Offset   |
        | Normal      | 5 – 15%              | Business as usual              | Moderate Price Offset  |
        | Flush       | 15 – 30%             | Need to deploy, price decreases| Low Price Offset       |
        | Excess      | > 30%                | Hoarding cash, price decreases | Lowest Price Offset    |
        """)
        st.markdown(f"Current Model uses representative F values: Tight={cash_percentages['Tight']}%, Normal={cash_percentages['Normal']}%, Flush={cash_percentages['Flush']}%, Excess={cash_percentages['Excess']}%")


    with formula_tabs[3]:
        st.markdown("**Micro Deal Indicator (I_micro):** Applies a surcharge for smaller deals.")
        st.latex(r"I_{\text{micro}} = \begin{cases} 1 & \text{if deal is Micro} \\ 0 & \text{if deal is Core} \end{cases}")
        st.markdown(f"If checked, adds the γ value ({gamma_val:.2f}) directly to the raw price.")

st.divider()


# --- Generate and Display Matrices ---
st.subheader("📅 Pricing & IRR Matrices (1-6 Months Tenure)")

# Generate matrices based on current core parameters and IRR settings
price_matrix, irr_matrices = generate_matrices(
    hk_val, alpha_val, psi_val, gamma_val, floor_val,
    st.session_state.principal, # Pass current principal for context if needed by function
    st.session_state.admin_fee_rate,
    st.session_state.consider_vat,
    st.session_state.repayment_type
)

# Display Price Matrix
st.markdown("**Minimum Price Matrix (% / month)**")
st.caption("This matrix shows the calculated minimum monthly price percentage based on risk, deal type, and cash bucket. It is independent of tenure and IRR settings.")
st.dataframe(price_matrix.style.format("{:.3f}%").highlight_max(axis=None, color='lightcoral').highlight_min(axis=None, color='lightgreen'))

st.markdown("---") # Visual separator

# Display IRR Matrices
st.markdown("**Annual IRR Matrix (%)**")
# Build status text based on current IRR settings
status_parts = []
if st.session_state.consider_vat: status_parts.append("includes 15% VAT reduction on fees")
else: status_parts.append("excludes VAT reduction")
if st.session_state.include_admin_fee and st.session_state.admin_fee_rate > 0: status_parts.append(f"includes {st.session_state.admin_fee_rate*100:.1f}% admin fee")
else: status_parts.append("excludes admin fee")
status_parts.append(f"uses '{st.session_state.repayment_type.replace('_', ' ').title()}' repayment")
st.caption(f"Showing calculated Annual IRR based on the prices above and current sidebar settings: {', '.join(status_parts)}.")

irr_tabs = st.tabs([f"{t} Month{'s' if t > 1 else ''}" for t in irr_matrices.keys()])

for i, tab in enumerate(irr_tabs):
    tenure_key = list(irr_matrices.keys())[i]
    with tab:
        st.dataframe(
            irr_matrices[tenure_key].style.format("{:.2f}%", na_rep="N/A")
                                     .background_gradient(cmap='viridis', axis=None) # Add heatmap
                                     .highlight_null(color='gray')
        )

st.divider()

# --- Footer ---
st.markdown("---")
st.caption("Deal Pricing Calculator v1.1 | Use sidebar for configuration`")