import streamlit as st
import pandas as pd
import numpy as np
from scipy import optimize

def calculate_deal_metrics(principal, monthly_rate, tenure, admin_fee_rate, repayment_type, consider_vat=True):
    """
    Calculate IRR and deal metrics for a loan
    
    Parameters:
    - principal: Principal loan amount
    - monthly_rate: Monthly profit rate (decimal)
    - tenure: Loan duration in months
    - admin_fee_rate: One-time admin fee rate (decimal)
    - repayment_type: Type of repayment ('bullet', 'monthly', 'bi_monthly')
    - consider_vat: Whether to apply 15% VAT deduction to fees (default: True)
    
    Returns:
    - Dictionary with annual_irr and total_return
    """
    # Base amount (no VAT)
    base_amount = principal

    # Monthly fee
    monthly_fee = base_amount * monthly_rate * tenure

    # Admin fee
    admin_fee = base_amount * admin_fee_rate

    # Apply VAT if enabled
    if consider_vat:
        vat_rate = 0.15
        monthly_fee = monthly_fee * (1 - vat_rate)  # Reduce by VAT
        admin_fee = admin_fee * (1 - vat_rate)  # Reduce by VAT

    # Total income
    total_income = monthly_fee + admin_fee

    # Total sales
    total_sales = base_amount + total_income

    # Cash flow calculation for IRR
    cash_flows = []
    cash_flows.append(-principal)  # Initial outflow

    if repayment_type == 'bullet':
        # Single payment at the end
        for i in range(tenure - 1):
            cash_flows.append(0)
        cash_flows.append(total_sales)

    elif repayment_type == 'monthly':
        # Monthly payments with admin fee in first month
        monthly_principal = base_amount / tenure
        monthly_profit = monthly_fee / tenure

        for i in range(tenure):
            if i == 0:
                cash_flows.append(monthly_principal + monthly_profit + admin_fee)
            else:
                cash_flows.append(monthly_principal + monthly_profit)

    elif repayment_type == 'bi_monthly':
        # Bi-monthly payments with admin fee in first payment
        num_payments = tenure // 2
        remainder = tenure % 2

        # Calculate principal and profit per payment instance
        payment_instances = num_payments + (1 if remainder > 0 else 0)
        if payment_instances > 0:
            bi_monthly_principal = base_amount / payment_instances
            bi_monthly_profit = monthly_fee / payment_instances
        else:  # Handle tenure < 2 edge case
            bi_monthly_principal = 0
            bi_monthly_profit = 0

        payment_made = False
        for i in range(tenure):
            # Payment due on even months or the last month if tenure is odd
            is_payment_period = ((i + 1) % 2 == 0) or (i == tenure - 1 and remainder > 0)

            if is_payment_period and payment_instances > 0:
                current_payment = bi_monthly_principal + bi_monthly_profit
                if not payment_made:
                    # Add admin fee to the very first payment made
                    current_payment += admin_fee
                    payment_made = True
                cash_flows.append(current_payment)
            else:
                # No payment this month
                cash_flows.append(0)

    # Calculate IRR
    try:
        def npv(rate, cash_flows):
            # Ensure rate > -1 for the formula to be valid
            if rate <= -1:
                return float('inf')  # Or some large number to avoid math errors
            return sum(cf / (1 + rate) ** (i) for i, cf in enumerate(cash_flows))

        # Calculate monthly IRR using Newton-Raphson method
        monthly_irr = optimize.newton(lambda r: npv(r, cash_flows), x0=0.1, tol=1e-6, maxiter=100)

        # Convert to annual IRR
        if monthly_irr > -1:
            annual_irr = (1 + monthly_irr) ** 12 - 1
        else:
            annual_irr = float('nan')  # Or handle as an error condition

    except ImportError:
        st.warning("Scipy not found. Using approximation for IRR.")
        # Simple approximation if scipy fails or is not installed
        if principal > 0 and tenure > 0:
            annual_irr = ((total_sales / principal) ** (12 / tenure)) - 1
        else:
            annual_irr = 0  # Avoid division by zero or invalid exponent
    except Exception as e:
        st.error(f"IRR calculation error: {e}")
        # Fallback approximation or error value
        if principal > 0 and tenure > 0:
            annual_irr = ((total_sales / principal) ** (12 / tenure)) - 1
        else:
            annual_irr = 0

    # Total return percentage
    total_return = (total_sales / principal - 1) * 100 if principal > 0 else 0

    return {
        "annual_irr": annual_irr,
        "total_return": total_return,
        "total_income": total_income,
        "monthly_fee": monthly_fee,
        "admin_fee": admin_fee,
        "total_sales": total_sales,
        "cash_flows": cash_flows
    }

def format_percentage(value):
    """Format a decimal as a percentage with 2 decimal places"""
    return f"{value*100:.2f}%"

def format_currency(value):
    """Format a number as currency with commas"""
    return f"{value:,.2f}"

def display_cash_flows(cash_flows):
    """Display cash flows in a DataFrame"""
    cf_df = pd.DataFrame({
        "Month": range(len(cash_flows)),
        "Cash Flow": cash_flows
    })
    cf_df["Type"] = cf_df["Cash Flow"].apply(lambda x: "Outflow" if x < 0 else "Inflow")
    return cf_df

def main():
    st.set_page_config(page_title="BNPL IRR Calculator", layout="wide")
    
    st.title("BNPL IRR Calculator")
    st.markdown("Calculate the IRR for different loan structures and payment types")
    
    # Create columns for input fields
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        principal = st.number_input("Principal Amount", min_value=1000.0, max_value=10000000.0, value=100000.0, step=10000.0)
    
    with col2:
        monthly_rate = st.number_input("Monthly Rate (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100
    
    with col3:
        admin_fee_rate = st.number_input("Admin Fee Rate (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.1) / 100
    
    with col4:
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=36, value=3, step=1)
    
    # Payment structure selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        repayment_type = st.radio(
            "Repayment Structure",
            ["monthly", "bi_monthly", "bullet"],
            format_func=lambda x: {
                "monthly": "Monthly EMI (Equal Monthly Installments)",
                "bi_monthly": "Bi-Monthly EMI (Every 2 months)",
                "bullet": "Bullet Payment (Single payment at end)"
            }[x],
            horizontal=True
        )
    
    with col2:
        consider_vat = st.toggle("Include 15% VAT Impact", value=True,
            help="When enabled, IRR calculations will account for 15% VAT deduction from fees")
    
    # Calculate metrics
    metrics = calculate_deal_metrics(principal, monthly_rate, tenure, admin_fee_rate, repayment_type, consider_vat)
    
    # Display results in a nice format
    st.markdown("### Results")
    vat_status = "after 15% VAT" if consider_vat else "before VAT"
    st.caption(f"All values shown {vat_status}")
    
    # Create columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annual IRR", format_percentage(metrics["annual_irr"]))
    
    with col2:
        st.metric("Total Return", format_percentage(metrics["total_return"]/100))
    
    with col3:
        st.metric("Total Profit", format_currency(metrics["total_income"]))
    
    # Additional details in an expander
    with st.expander("Show Details"):
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.subheader("Financial Details")
            details_data = {
                "Principal": format_currency(principal),
                "Monthly Fee Total": format_currency(metrics["monthly_fee"]),
                "Admin Fee": format_currency(metrics["admin_fee"]),
                "Total Income": format_currency(metrics["total_income"]),
                "Total Sales": format_currency(metrics["total_sales"])
            }
            
            details_df = pd.DataFrame(details_data.items(), columns=["Item", "Value"])
            st.table(details_df)
            
            if consider_vat:
                st.caption("Note: Fee amounts shown are after 15% VAT deduction")
        
        with details_col2:
            st.subheader("Cash Flow Analysis")
            st.markdown(f"Initial Outflow: **{format_currency(abs(metrics['cash_flows'][0]))}**")
            
            # Show cash flows in a table
            cf_df = display_cash_flows(metrics["cash_flows"])
            st.dataframe(cf_df)
    
    # Add an explanation about IRR and VAT
    with st.expander("About IRR Calculation"):
        st.markdown("""
        ### Internal Rate of Return (IRR)
        
        The IRR is the discount rate that makes the net present value (NPV) of all cash flows equal to zero.
        It represents the annualized effective compounded return rate of the investment.
        
        For BNPL products, a higher IRR indicates a more profitable investment.
        
        ### VAT Impact
        
        When VAT is considered (15%):
        - Monthly fees are reduced by 15% (VAT portion)
        - Admin fees are reduced by 15% (VAT portion)
        - IRR and returns are calculated on the net amount after VAT
        
        This provides a more accurate view of actual returns since VAT must be paid on fee income.
        
        ### Calculation Method
        
        This calculator uses the Newton-Raphson method to calculate the IRR based on:
        - The initial outflow (principal)
        - Monthly payments (based on repayment structure)
        - Admin fee (added to the first payment)
        - Monthly fee (distributed according to repayment structure)
        
        The monthly IRR is then annualized using the formula: `(1 + monthly_irr)^12 - 1`
        """)

if __name__ == "__main__":
    main() 