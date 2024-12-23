import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class FinancialAnalyzer:
    def __init__(self):
        self.revenue = 0
        self.ebit = 0
        self.ebitda = 0
        self.net_income = 0
        self.depreciation_amortization = 0
        self.employees = 0
        self.accuracy_metrics = {}

    def extract_revenue(self, data: Dict) -> float:
        """
        Extract revenue based on company type and SOP rules
        Returns revenue value
        """
        if data.get('company_type') == 'general':
            # Case 1: Revenue / Sales/ Turnover
            revenue_items = ['revenue', 'sales', 'turnover']
            for item in revenue_items:
                if item in data:
                    # Don't include other income
                    return data[item]
            
        elif data.get('company_type') == 'bank_nbfc':
            # Net interest income + Fee commission + Trading income
            return (data.get('net_interest_income', 0) + 
                   data.get('fee_commission_income', 0) + 
                   data.get('trading_income', 0))
            
        elif data.get('company_type') == 'insurance':
            # Net earned premium + Reinsurance recoveries + Commission
            return (data.get('net_earned_premium', 0) + 
                   data.get('reinsurance_recoveries', 0) + 
                   data.get('reinsurance_commission', 0))
        
        return 0

    def calculate_ebit(self, data: Dict) -> float:
        """
        Calculate EBIT based on SOP cases
        Returns EBIT value
        """
        if 'operating_profit' in data:
            # Case 1: Direct operating profit
            return data['operating_profit']
        
        elif 'pbt' in data and 'interest' in data:
            # Case 5: PBT + Interest calculation
            return data['pbt'] + data['interest_expense'] - data['interest_income']
        
        elif 'pat' in data and 'tax' in data:
            # Case 6: PAT based calculation
            return (data['pat'] + data['tax'] + 
                   data['interest_expense'] - data['interest_income'])
            
        return 0

    def calculate_ebitda(self, data: Dict) -> float:
        """
        Calculate EBITDA based on SOP rules
        Returns EBITDA value
        """
        ebit = self.calculate_ebit(data)
        
        # Add back depreciation and amortization
        if 'depreciation' in data or 'amortization' in data:
            self.depreciation_amortization = (data.get('depreciation', 0) + 
                                            data.get('amortization', 0))
            
            # Don't include impairment in D&A
            if 'impairment' in data and data.get('has_impairment_detail', False):
                self.depreciation_amortization -= data['impairment']
                
        return ebit + self.depreciation_amortization

    def extract_net_income(self, data: Dict) -> float:
        """
        Extract net income based on SOP rules
        Returns net income value
        """
        if 'profit_before_minority' in data:
            # Case 1: Profit before minority interest
            return data['profit_before_minority']
            
        elif 'pat' in data and 'minority_interest' in data:
            # Case 2: PAT + Minority adjustment
            return data['pat'] + data['minority_interest']
            
        elif 'profit_equity_statement' in data:
            # Case 3: From equity statement
            return data['profit_equity_statement']
            
        return data.get('net_income', 0)

    def validate_data(self, data: Dict) -> Dict[str, bool]:
        """
        Validate data based on hard stops
        Returns dict of validation results
        """
        validations = {
            'currency_populated': bool(data.get('currency')),
            'currency_valid': data.get('currency') in self.valid_currencies,
            'period_date_populated': bool(data.get('period_date')),
            'period_date_valid': 1990 <= data.get('period_date', 0) <= 2024,
            'series_id_populated': bool(data.get('series_id')),
            'unit_populated': bool(data.get('unit'))
        }
        return validations

    def analyze_financials(self, data: Dict) -> Dict:
        """
        Main analysis function
        Returns dict with analyzed values and metrics
        """
        # Extract and calculate key metrics
        self.revenue = self.extract_revenue(data)
        self.ebit = self.calculate_ebit(data)
        self.ebitda = self.calculate_ebitda(data)
        self.net_income = self.extract_net_income(data)
        self.employees = data.get('employees', 0)
        
        # Validate data
        validations = self.validate_data(data)
        
        # Calculate accuracy metrics
        total_validations = len(validations)
        passed_validations = sum(validations.values())
        accuracy = (passed_validations / total_validations) * 100
        
        return {
            'revenue': self.revenue,
            'ebit': self.ebit,
            'ebitda': self.ebitda,
            'net_income': self.net_income,
            'employees': self.employees,
            'depreciation_amortization': self.depreciation_amortization,
            'validations': validations,
            'accuracy': accuracy
        }

def format_date(date_str: str) -> str:
    """
    Format date to MM-DD-YYYY as per SOP
    """
    try:
        date = pd.to_datetime(date_str)
        return date.strftime('%m-%d-%Y')
    except:
        return date_str

# Usage example
if __name__ == "__main__":
    # Sample financial data
    sample_data = {
        'company_type': 'general',
        'revenue': 1000000,
        'sales': 1000000,
        'operating_profit': 150000,
        'depreciation': 50000,
        'amortization': 20000,
        'pat': 80000,
        'minority_interest': -5000,
        'currency': 'USD',
        'period_date': 2023,
        'series_id': 'ABC123',
        'unit': 'thousands',
        'employees': 500
    }
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer()
    
    # Analyze financials
    results = analyzer.analyze_financials(sample_data)
    
    # Print results
    print("\nFinancial Analysis Results:")
    print(f"Revenue: {results['revenue']:,.2f}")
    print(f"EBIT: {results['ebit']:,.2f}")
    print(f"EBITDA: {results['ebitda']:,.2f}")
    print(f"Net Income: {results['net_income']:,.2f}")
    print(f"Employees: {results['employees']}")
    print(f"\nAccuracy: {results['accuracy']:.2f}%")
    print("\nValidations:")
    for key, value in results['validations'].items():
        print(f"{key}: {'Passed' if value else 'Failed'}")