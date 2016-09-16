# Capstone Project
## Machine Learning Engineer Nanodegree
Carl Gosselin  
September, 2016

## I. Definition
[//]: # "_(approx. 1-2 pages)_"

### Project Overview

[//]: # "In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:"
[//]: # "- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_"
[//]: # "- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_"

**The problem domain**

I've chosen to apply my new-found knowledge in machine learning algorithms to the investment and trading domain. I keep hearing that many firms are using machine learning algorithms to gain an edge in the market.  For this project, I'd like to:<br>

A) understand how machine learning algorithms are currently used in the stock market <br>
B) understand the current challenges of using machine learning algorithms to help predict stock prices <br>
C) and finally, to make an attempt at resolving one of the challenges of applying machine learning algorithms to predicting stock prices <br>

**Project origin**

This project originated from Udacity's course - <a href="https://www.udacity.com/course/machine-learning-for-trading--ud501?_ga=1.193565739.763209811.1447720300"> Machine Learning for Trading </a>.  

I've always been interested in understanding the stock market.  I'm happy to have the opportunity to apply machine learning algorithms to stock data. <a href="https://www.youtube.com/watch?v=j-6pr72cves" target="_blank"> "_I am very happy to be here!_" </a>

**Related datasets**

CSV files captures the daily stock data for the following stocks from the last 5 years <br>
1. SPY (SPDR S&P 500 ETF) <br>
2. UPRO (ProShares UltraPro S&P500) <br>
3. GOOG (Google) <br>
4. AAPL (Apple) <br>
5. AMZN (Amazon) <br>
6. DIS (Disney) <br>
7. NFLX (Netflix) <br>
8. FB (Facebook) <br>
9. AXY (Alterra Power) <br>
10. VIX (Volatility Index) <br>
11. TSLA (Tesla) <br>
12. GWPH (GW Pharmaceuticals)<br>
13. MSFT (Microsoft) <br>
14. GLD (SPDR Gold Shares) <br>



### Problem Statement
[//]: # "In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:"
[//]: # " _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_"
[//]: # " _Have you thoroughly discussed how you will attempt to solve the problem?_"
[//]: # " _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_"

My problem statement is simple and straightforward - How can I increase my chances of picking "winning" stocks?

My current analysis involves tracking a handful of stocks on google.com/finance.  When time permits, I print-off quarterly and other reports and read them in detail.  I believe that another strategy that would complement my current analysis is creating predictive models on the price of the stock.  I feel that my feeble brain is no match for machine learning algorithms when it comes down to crunching numbers.


### Metrics
[//]: # "In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project."  
[//]: # "These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:"
[//]: # "- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_"
[//]: # "- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_"

I will be using a train/test split on the stock data.  A section of the data will be used to train the model and another section will be used to test the model on unseen data (out of sample data).  

## II. Analysis
[//]: # "_(approx. 2-4 pages)_"

### Data Exploration
[//]: # "In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:"
[//]: # "- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_"
[//]: # "- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_"
[//]: # "- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_"
[//]: # "- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_"


For this project, I will be using data in the form of datasets.  I have selected 12 stocks that are actively traded on the stock market.  I've also included data related to the volatility index (VIX).  This is also known as the fear gauge.  It is explained that when the VIX is high, the market moves lower as they are (supposedly) inversely related.  I plan to include the VIX data with other stocks in visual comparisons.

All stock data files have the same columns: <br>
- Date (date of the stock price) <br>
- Open (opening price of the stock <br>
- High (highest price of the stock for the day) <br> 
- Low (lowest price of the stock for the day) <br>
- Close (price of the stock at close) <br>
- Volume (the trading volume of the stock for the day) <br>
- Adj Close (the adjusted close price) <br>
note: the adjusted close price will be different from the "Close" price when a company chooses to split the stock, give dividends, etc... <br>

Other comments about the data: <br>
- By default, the csv files are indexed by a number starting from "0".  The code will need to explicitly identify the "Date" column as the index column. <br>
- The data in the csv files begin with the latest trade day.  Visualizing this data untouched, the graph will show a downward trend for stocks with increasing prices.  The data will need to be re-organized to show a proper linear progression through time. <br>
- It is possible that some of the stocks did not trade on a certain day.  Stock with no trades on a specific day will need a "nan" (or similar) inserted into the empty cell. <br>
- CSV files will need to be joined to combine data for comparison. In other words, csv files will need to be joined. <br>
- When joining csv files for different stocks, column names will need to be modified to prevent duplication of column titles.  To avoid processing errors, columns will be renamed to the stock ticker.  E.g. "Adj Close" -> "GOOG".  Updating column names will avoid overlapping of column names during csv/table joins. <br>
- To prevent duplicating code, a utility function will need to be built to process all csv stock files in an efficient manner. <br>
- abnormality -> dividends and stock splits. Need to use adjusted close.  Historical data gets adjusted for this purpose.
<br>
- talk about normalization... to be able to compare stocks traded at different price points, stocks will be normalized to start at $1.  This way, one can compare the magnitude of positive and negative direction.
<br>


### Exploratory Visualization
[//]: # "In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should"
[//]: # "adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:"
[//]: # "- _Have you visualized a relevant characteristic or feature about the dataset or input data?_"
[//]: # "- _Is the visualization thoroughly analyzed and discussed?_"
[//]: # "- _If a plot is provided, are the axes, title, and datum clearly defined?_"

The data has been visualized in many different ways and documented in a few files.  Files on github:
(note:  comments are included in the ipynb files)

1) 160_exploratory_visualization.ipynb
- Visuals in this file display normalized data in comparison to the benchmark stock (SPY)
- The visuals quickly show which stocks performed better than the benchmark stock (SPY)

2) 170_bollinger_bands.ipynb
- Visuals in this file display the famous bollinger bands for a selected number of stocks
- The width between the upper and lower bands indicate the amount of risk (aka standard deviation) in a stocks movement
 
3) 180_daily_returns.ipynb
- Visuals in this file show the daily returns of a stock in comparison to the benchmark stock (SPY)
- The visuals also displays how much a stock moves along (or against) the movement of the benchmark stock (SPY)

4) 200_fill_missing_values.ipynb
- The visuals in this file show the gaps in trading for stocks that are not traded every day
- Forward filling and back filling techniques were applied to be able to compare, as best as possible, against other stocks traded on a daily basis

5) 210_plot_histograms.ipynb
- Another way to visualize the data was to display the stock with histograms
- In this file, daily returns were compared to the benchmark stocks in the same graph

6) 220_scatterplots_beta_alpha_and_correlation.ipynb
- Scatterplots were created to compare stocks to the benchmark stock (SPY)
- This file also captured the beta and alpha variables.
- The beta variable indicates how much more reactive the stock is to the market than the benchmark stock (SPY)
- the alpha variable indicates how well the stock performs with respect to the benchmark stock (SPY).

7) 280_portfolio_statistics_visual.ipynb
- The visuals in this file shows the performance of selected stocks as a portfolio
- This portfolio of stocks was then compared to the benchmark stock (SPY)

8) 290_portfolio_allocation_optimization.ipynb
- This visual shows the performance of a portfolio of stock after portfolio optimization

9) 300_CAPM_with_optimized_portfolio_allocation.ipynb
- This visual takes the data from the previous file and display the CAPM visual
- This file also captured the beta and alpha variables.
- The beta variable indicates how much more reactive the stock is to the market than the benchmark stock (SPY)
- the alpha variable indicates how well the stock performs with respect to the benchmark stock (SPY).

10) 310_return_vs_risk_scatterplot.ipynb
- This visual displays a graph for the amount of risk vs return for a selected number of stocks



### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics
of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

For this project, I will be using two supervised learning algorithms to "predict" stock prices.  The first is linear regression and the second is k nearest neighbors.  



### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

The benchmark for linear regression is ....


The benchmark for k nearest neighbors is ...


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that
you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

For this project, I will be using data in the form of datasets.  I have selected 12 stocks that are actively traded on the stock market.  I've also included data related to the volatility index (VIX).  This is also known as the fear gauge.  It is explained that when the VIX is high, the market moves lower as they are (supposedly) inversely related.  I plan to include the VIX data with other stocks in visual comparisons.

All stock data files have the same columns: <br>
- Date (date of the stock price) <br>
- Open (opening price of the stock <br>
- High (highest price of the stock for the day) <br> 
- Low (lowest price of the stock for the day) <br>
- Close (price of the stock at close) <br>
- Volume (the trading volume of the stock for the day) <br>
- Adj Close (the adjusted close price) <br>
note: the adjusted close price will be different from the "Close" price when a company chooses to split the stock, give dividends, etc... <br>

Other comments about the data: <br>
- By default, the csv files are indexed by a number starting from "0".  The code will need to explicitly identify the "Date" column as the index column. <br>
- The data in the csv files begin with the latest trade day.  Visualizing this data untouched, the graph will show a downward trend for stocks with increasing prices.  The data will need to be re-organized to show a proper linear progression through time. <br>
- It is possible that some of the stocks did not trade on a certain day.  Stock with no trades on a specific day will need a "nan" (or similar) inserted into the empty cell. <br>
- CSV files will need to be joined to combine data for comparison. In other words, csv files will need to be joined. <br>
- When joining csv files for different stocks, column names will need to be modified to prevent duplication of column titles.  To avoid processing errors, columns will be renamed to the stock ticker.  E.g. "Adj Close" -> "GOOG".  Updating column names will avoid overlapping of column names during csv/table joins. <br>
- To prevent duplicating code, a utility function will need to be built to process all csv stock files in an efficient manner. <br>

Other comments that are also included in the exploratory visualization ipynb document...

The above graph display GOOG prices at close.
Challenge #1: The data is displayed backwards in time when ready the data directly from the csv file.
Challenge #2: Only one stock is displayed in the graph. We need to add more stocks for comparison.
Challenge #3: The graph shows 5 years worth of data.  Let's only view the last 2 years in the next graph.


Let's fix these challenges in the next visual...
-----

The above graph displays the list of stocks I've been tracking for a few years.


Progress from last visual...
- Overcame the challenge of showing more than one stock in a graph. The graph is now showing multiple stocks.
- Also, overcame the challenge of creating redundant code by creating a utility function to pull data from each csv file.
- Resolved the challenge of the data displaying stock prices in reverse order.
- Sliced the data to show 2 years worth of data.  The csv files cover 5 years worth of data.


Challenge: It is hard to visually assess the stocks at different price points.
This can be resolved by normalizing the data.  Let's fix this challenge in the next visual...
-----

The above graph normalizes the stock data at a starting point of 1 dollar.


Result: Whoah!  This graph is way too busy.  The same colour is applied to more than one stock.
The next series of graphs will divide the stock into smaller groups for more meaningful comparisons...
-----





### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear
how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this
section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_






### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for
certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate
results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In
addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s
solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You
should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a
significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are
expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this
section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be
made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and
compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?