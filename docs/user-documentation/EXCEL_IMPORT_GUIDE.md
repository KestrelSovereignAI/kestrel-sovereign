# How to Import KESTREL_ACCELERATED_GTM_SPREADSHEET_DATA.csv into Excel

## Step-by-Step Instructions

### Method 1: Direct Import (Recommended)
1. **Open Excel** on your computer
2. **Click File → Open**
3. **Navigate to** your Kestrel folder
4. **Select** `KESTREL_ACCELERATED_GTM_SPREADSHEET_DATA.csv`
5. **Choose** "Delimited" when prompted
6. **Select** "Comma" as the delimiter
7. **Click Finish**

### Method 2: Import as Text File
1. **Open Excel** (blank workbook)
2. **Click Data → From Text/CSV**
3. **Select** the CSV file from your Kestrel folder
4. **Choose** "Comma" as delimiter
5. **Click Load**

### Method 3: Copy-Paste Method
1. **Open** the CSV file in Notepad or any text editor
2. **Select All** (Ctrl+A) and **Copy** (Ctrl+C)
3. **Open Excel** and **Paste** (Ctrl+V) into cell A1
4. **Use Data → Text to Columns** to split the data properly

## Spreadsheet Structure

The CSV contains multiple "tabs" marked with `=== TAB NAME ===` headers:

### Tab 1: ASSUMPTIONS
- Core input variables for all three scenarios
- S-curve adoption parameters
- Year totals calculations

### Tab 2: GTM ACCELERATION
- Detailed budget allocation by scenario
- Hiring timelines
- GTM investment breakdown

### Tab 3: FINANCIAL PROJECTIONS
- Income statements for each scenario
- EBITDA calculations
- Cumulative tracking

### Tab 4: SCENARIOS COMPARISON
- Runway analysis
- Customer growth ramps
- Revenue impact
- CAC/LTV analysis
- Channel segmentation

### Tab 5: FORMULAS AND CALCULATIONS
- All the Excel formulas used
- Calculation methodologies

### Tab 6: RISK AND SENSITIVITY ANALYSIS
- Risk assessment by scenario
- Sensitivity factors
- Mitigation strategies

### Tab 7: NOTES AND ASSUMPTIONS
- Model assumptions
- Founder decision points
- Success milestones

## Key Formulas to Recreate in Excel

### Runway Calculation
```
= (Funding_Amount + Bridge_Capital) / Monthly_Burn
```

### Customer Growth
```
= Base_Customers * (1 + GTM_Acceleration_Factor)
GTM_Acceleration_Factor = (GTM_Budget_% / 45%) ^ 0.7
```

### Revenue Calculation
```
= Customers * ARPU * (1 - Churn_Rate_Monthly)
```

### CAC Payback
```
= CAC / (Monthly_ARPU * Gross_Margin)
LTV = Annual_ARPU / Churn_Rate
LTV_CAC_Ratio = LTV / CAC
```

## Tips for Using the Excel Version

1. **Create Separate Worksheets** for each tab section
2. **Use Cell References** instead of hard-coded numbers
3. **Add Data Validation** for scenario selection
4. **Create Charts** for runway, revenue, and customer growth
5. **Add Conditional Formatting** for risk indicators
6. **Use Scenario Manager** for sensitivity analysis

## Data Validation

- All percentages are formatted as percentages in Excel
- Currency values include dollar signs
- Numbers are formatted appropriately
- Dates are in MM/DD/YYYY format where applicable

## Need Help?

If you encounter any issues importing the CSV:
1. Make sure Excel is set to use commas as delimiters
2. Check that the file wasn't corrupted during download
3. Try opening in Google Sheets first, then export to Excel
4. Contact support if you need the formulas rebuilt

---

**File Location:** `c:\Users\gabri\Kestrel\KESTREL_ACCELERATED_GTM_SPREADSHEET_DATA.csv`
**Ready to import into Excel for full spreadsheet functionality!**</content>
<parameter name="filePath">c:\Users\gabri\Kestrel\EXCEL_IMPORT_GUIDE.md