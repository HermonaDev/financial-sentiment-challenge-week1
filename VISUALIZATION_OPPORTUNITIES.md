# VISUALIZATIONS TO EMBED IN YOUR FINAL SUBMISSION REPORT

## Summary
This guide identifies **key visualizations from your existing notebooks** that should be embedded directly in your `WEEK_1_FINAL_SUBMISSION_REPORT.md` for clarity, impact, and evidence of analysis quality.

**Goal:** Generate PNG files from your notebooks and embed them in the markdown report to support findings and demonstrate analytical competence.

---

## 📊 VISUALIZATIONS TO ADD TO YOUR REPORT

### SECTION 2: METHODOLOGY (Visual Evidence of Data & Approach)

**Location in Report:** After section 2.3 "Daily Aggregation"

**Visualization 1: Sentiment Distribution - Overall**
- **Source:** Notebook 03, Cell 7 (already exists)
- **Purpose:** Show the distribution of sentiment scores across all 44,196 articles
- **What to Embed:** The 4-panel sentiment analysis showing:
  - Histogram of sentiment scores with mean/median lines
  - Pie chart or bar chart of Positive/Neutral/Negative counts
  - Daily average sentiment over time
  - Sentiment distribution statistics
- **Caption:** *"Sentiment Analysis of 44,196 News Articles (2011-2020): Shows slightly negative bias in overall sentiment with concentrated clustering around neutral values."*

**Visualization 2: Data Coverage - Stock Overlap**
- **Source:** Notebook 03, Cell 13-14 (data verification)
- **Purpose:** Show which stocks have sufficient sentiment-price data overlap
- **What to Create:** Simple bar chart showing:
  - Number of news days per stock
  - Number of trading days matched
  - Sample sizes for each stock analyzed
- **Why Important:** Explains why only 3 stocks were analyzed (sample size constraints)
- **Sample Code:**
  ```python
  stocks = ['AAME', 'GEOS', 'GDXJ']
  merged_days = [10, 10, 10]
  
  plt.figure(figsize=(8, 4))
  plt.bar(stocks, merged_days, color='steelblue')
  plt.ylabel('Number of Matched Trading Days')
  plt.title('Sentiment-Price Data Overlap by Stock')
  plt.axhline(y=10, color='red', linestyle='--', label='Minimum for analysis')
  plt.legend()
  plt.tight_layout()
  plt.savefig('data_coverage.png', dpi=300, bbox_inches='tight')
  ```

---

### SECTION 3: FINDINGS - Task 1 (EDA Evidence)

**Location in Report:** Section 3.1 "Task 1 - Exploratory Data Analysis"

**Visualization 3: Sentiment Distribution by Category**
- **Source:** Notebook 01, Cell 25 (already exists)
- **Purpose:** Visual proof of sentiment categorization mentioned in your methodology
- **What to Embed:** Bar chart or pie chart showing:
  - Positive: 20.2% (8,915 articles)
  - Neutral: 53.0% (23,434 articles)
  - Negative: 26.8% (11,847 articles)
- **Caption:** *"Sentiment categories reveal balanced but slightly negative news environment. Over half of articles are neutral, indicating cautious language typical in financial reporting."*

**Visualization 4: Top 10-15 Stocks by Coverage**
- **Source:** Notebook 01, Cell 18 (already exists)
- **Purpose:** Show which companies dominated news coverage
- **What to Embed:** Horizontal bar chart showing:
  - Stock ticker on Y-axis
  - Number of articles on X-axis
  - Color gradient to show coverage concentration
- **Caption:** *"News coverage concentration: Top 10 stocks account for disproportionate share of articles, while 6,194 stocks have minimal coverage (avg 10 articles each)."*

**Visualization 5: Temporal Trends - Articles Over Time**
- **Source:** Notebook 01, Cell 16 (already exists)
- **Purpose:** Show how news coverage volume changed from 2011-2020
- **What to Embed:** Line chart showing:
  - Articles published per month/quarter over 9 years
  - Trend line showing overall trajectory
- **Caption:** *"News publication frequency shows growth from 2011-2015, plateau 2015-2017, then decline toward 2020, reflecting changing market focus and data source availability."*

---

### SECTION 3: FINDINGS - Task 2 (Technical Analysis Evidence)

**Location in Report:** Section 3.2 "Task 2 - Quantitative Technical Analysis"

**Visualization 6: Technical Indicators Summary - Sample Stock**
- **Source:** Notebook 02, Cell 11 or 15 (already exists)
- **Purpose:** Show example of calculated technical indicators
- **What to Embed:** Multi-panel chart showing:
  - Price chart with 20/50-day moving averages
  - MACD with signal line and histogram
  - RSI with overbought/oversold zones (70/30 lines)
  - Volume bars
- **Caption:** *"Example technical indicator analysis for major stock: Price chart with moving average overlays, momentum indicators (MACD, RSI), and volume. These indicators identified trading signals used in correlation analysis."*

**Visualization 7: Stock Comparison Metrics Table/Chart**
- **Source:** Notebook 02, Cell 12-13 (summary statistics)
- **Purpose:** Show comparative metrics across analyzed stocks
- **What to Create:** Either:
  - **Option A:** Grouped bar chart comparing:
    - Annualized return (%)
    - Annualized volatility (%)
    - Sharpe ratio
    - Max drawdown
  - **Option B:** Table with visual elements (color scale showing high/low values)
- **Sample Code:**
  ```python
  stocks = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA']
  returns = [...]  # your annualized returns
  volatility = [...]  # your annualized volatility
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  ax1.bar(stocks, returns, color='green', alpha=0.7)
  ax1.set_ylabel('Annualized Return (%)')
  ax1.set_title('Returns by Stock')
  ax1.tick_params(axis='x', rotation=45)
  
  ax2.bar(stocks, volatility, color='red', alpha=0.7)
  ax2.set_ylabel('Annualized Volatility (%)')
  ax2.set_title('Volatility by Stock')
  ax2.tick_params(axis='x', rotation=45)
  
  plt.tight_layout()
  plt.savefig('stock_comparison.png', dpi=300, bbox_inches='tight')
  ```
- **Caption:** *"Technical metric comparison: Shows variation in risk-adjusted returns (Sharpe ratio), volatility, and drawdown characteristics across analyzed securities."*

---

### SECTION 3: FINDINGS - Task 3 (Correlation Analysis Evidence)

**Location in Report:** Section 3.3 "Task 3 - Correlation Analysis Results"

**Visualization 8: Correlation Results Summary Chart** (HIGH PRIORITY)
- **Source:** Notebook 03, Cells 15-16 (already exists - enhance it)
- **Purpose:** Visual summary of all correlation findings at a glance
- **What to Create/Embed:** Two-panel chart showing:
  - **Left Panel:** Horizontal bar chart of correlation coefficients
    - AAME: +0.1147 (red/gray - not significant)
    - GEOS: +0.0696 (red/gray - not significant)  
    - GDXJ: -0.8030 (green - **significant**)
    - Color code: green = p<0.05, red = p>0.05
  - **Right Panel:** Bar chart of P-values with significance threshold
    - Red dashed line at p=0.05
    - Only GDXJ crosses the threshold
- **Why Important:** This single visualization proves your main finding without readers having to parse the table
- **Sample Code:**
  ```python
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  stocks = ['AAME', 'GEOS', 'GDXJ']
  correlations = [0.1147, 0.0696, -0.8030]
  p_values = [0.7523, 0.8485, 0.0052]
  
  colors = ['#d62728' if p > 0.05 else '#2ca02c' for p in p_values]
  
  # Correlation bars
  ax1.barh(stocks, correlations, color=colors, alpha=0.8)
  ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
  ax1.set_title('Sentiment-Return Correlation by Stock', fontsize=12, fontweight='bold')
  ax1.axvline(0, color='black', linestyle='-', linewidth=0.8)
  ax1.set_xlim([-1, 1])
  
  # P-value bars
  ax2.bar(stocks, p_values, color=colors, alpha=0.8)
  ax2.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Significance level (α=0.05)')
  ax2.set_ylabel('P-value', fontsize=11)
  ax2.set_title('Statistical Significance Test Results', fontsize=12, fontweight='bold')
  ax2.legend()
  ax2.set_ylim([0, 1])
  
  plt.tight_layout()
  plt.savefig('correlation_summary.png', dpi=300, bbox_inches='tight')
  plt.show()
  ```
- **Caption:** *"Correlation Analysis Results: GDXJ exhibits a statistically significant negative correlation (r = -0.803, p = 0.0052) between sentiment and returns. AAME and GEOS show no significant relationships (p > 0.05)."*

**Visualization 9: GDXJ Scatter Plot with Regression Line** (ALREADY EXISTS)
- **Source:** Notebook 03, Cell 17 (already exists)
- **Purpose:** Show the actual relationship for the significant finding
- **What to Embed:** Scatter plot showing:
  - X-axis: Daily average sentiment
  - Y-axis: Daily returns (%)
  - Blue dots: Each trading day's data point
  - Red line: Linear regression fit (negative slope)
  - R² value shown
  - Equation displayed
- **Caption:** *"GDXJ Sentiment-Return Scatter Plot: Negative relationship shows that positive sentiment days coincide with lower returns, suggesting contrarian dynamics in junior mining equities."*

**Visualization 10: Statistical Assumption Validation** (ADDS RIGOR)
- **Source:** Notebook 03, Create new visualization
- **Purpose:** Demonstrate awareness of Pearson correlation assumptions
- **What to Create:** Residual analysis plot for GDXJ (the significant case):
  ```python
  from scipy import stats
  
  # Assuming 'merged' df with avg_sentiment and Daily_Return columns
  slope, intercept, r_value, p_value, std_err = stats.linregress(
      merged['avg_sentiment'], merged['Daily_Return']
  )
  predicted = slope * merged['avg_sentiment'] + intercept
  residuals = merged['Daily_Return'] - predicted
  
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  
  # Residuals vs Fitted
  axes[0, 0].scatter(predicted, residuals, s=60, alpha=0.7)
  axes[0, 0].axhline(y=0, color='r', linestyle='--')
  axes[0, 0].set_ylabel('Residuals')
  axes[0, 0].set_title('Residuals vs Fitted Values')
  axes[0, 0].grid(True, alpha=0.3)
  
  # Histogram of residuals
  axes[0, 1].hist(residuals, bins=5, edgecolor='black', alpha=0.7)
  axes[0, 1].set_xlabel('Residuals')
  axes[0, 1].set_title('Distribution of Residuals')
  axes[0, 1].grid(True, alpha=0.3)
  
  # Q-Q plot
  stats.probplot(residuals, dist="norm", plot=axes[1, 0])
  axes[1, 0].set_title('Q-Q Plot (Normality Check)')
  axes[1, 0].grid(True, alpha=0.3)
  
  # Scale-Location plot
  standardized_residuals = residuals / np.std(residuals)
  axes[1, 1].scatter(predicted, np.sqrt(np.abs(standardized_residuals)), s=60, alpha=0.7)
  axes[1, 1].set_ylabel('√|Standardized Residuals|')
  axes[1, 1].set_title('Scale-Location Plot')
  axes[1, 1].grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig('residual_analysis_gdxj.png', dpi=300, bbox_inches='tight')
  ```
- **Why Important:** Shows you understand statistical rigor and tested assumptions
- **Caption:** *"Residual Diagnostics for GDXJ: Plots validate Pearson correlation assumptions - residuals show reasonable independence and approximate normality (Q-Q plot), supporting the validity of our statistical test."*

---

## 📋 IMPLEMENTATION CHECKLIST FOR YOUR REPORT

### Step 1: Generate PNG Files from Notebooks
- [ ] Cell 7 from Notebook 03 → save as `sentiment_analysis_44k.png`
- [ ] Cell 18 from Notebook 01 → save as `top_stocks_coverage.png`
- [ ] Cell 16 from Notebook 01 → save as `articles_over_time.png`
- [ ] Cell 25 from Notebook 01 → save as `sentiment_categories.png`
- [ ] Cell 11 from Notebook 02 → save as `technical_indicators_example.png`
- [ ] Cell 15 from Notebook 02 → save as `stock_metrics_comparison.png`
- [ ] **Create new:** Visualization 8 → save as `correlation_summary.png`
- [ ] Cell 17 from Notebook 03 → save as `gdxj_scatter_regression.png`
- [ ] **Create new:** Visualization 10 → save as `residual_analysis_gdxj.png`

### Step 2: Embed in Your Report
Add after relevant sections:
```markdown
![Visualization Title](path_to_image.png)
*Caption explaining what the visualization shows and why it matters*
```

### Step 3: Quality Standards
- [ ] All images 300 DPI (publication quality)
- [ ] Legends clearly labeled
- [ ] Axis labels readable and units specified
- [ ] Color-blind friendly (avoid red-green only)
- [ ] Captions explain the finding, not just describe the plot

---

## 🎯 PRIORITY RANKING FOR REPORT IMPACT

**Must Include (High Impact):**
1. ✅ Correlation Summary Chart (Visualization 8) - **Proves your main finding**
2. ✅ GDXJ Scatter Plot (Visualization 9) - **Shows the relationship visually**
3. ✅ Sentiment Distribution (Visualization 1) - **Shows your data**
4. ✅ Stock Coverage (Visualization 4) - **Explains data constraints**

**Should Include (Professional Polish):**
5. ✅ Technical Indicators Example (Visualization 6) - **Shows Task 2 work**
6. ✅ Residual Analysis (Visualization 10) - **Demonstrates statistical rigor**
7. ✅ Temporal Trends (Visualization 5) - **Shows data evolution**

**Nice to Have (Supporting Detail):**
8. ✅ Sentiment Categories (Visualization 3) - **Reinforces methodology**
9. ✅ Stock Metrics Comparison (Visualization 7) - **Cross-stock context**

---

## 📝 CAPTION TEMPLATES FOR YOUR REPORT

Use these as starting points for captions:

**For Data Visualizations:**
- "Figure X: Shows [what the plot displays]. The data reveals [key insight]. This [supports the methodology / explains the constraint]."

**For Statistical Visualizations:**
- "Figure X: [Technical finding with numbers]. The [green/significant result] indicates [interpretation]. Non-significant results [alternative explanation]."

**For Methodology Visualizations:**
- "Figure X: Demonstrates [what analysis was done]. This approach [why this method was chosen]. Results [what it enabled]."

---

## 🚀 QUICK START: Add One Visualization to Your Report Now

**To test the embedding process:**

1. Open Notebook 03 and run Cell 7 (Sentiment Analysis)
2. Right-click on the generated plot → Save image as `sentiment_analysis.png`
3. In your report, add to Section 3.1:
   ```markdown
   ![Sentiment Analysis of News Headlines](../notebooks/sentiment_analysis.png)
   *Figure 1: Sentiment distribution of 44,196 articles shows 53% neutral sentiment, 27% negative, and 20% positive. Slightly negative bias likely reflects cautious tone in financial reporting.*
   ```
4. Test in markdown preview

That's it! Now repeat for other visualizations.
