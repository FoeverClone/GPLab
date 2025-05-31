# Voucher Policy Analysis

This project provides a modular framework for analyzing the results of consumption voucher policy simulations, focusing on different aspects of economic behavior and sentiment analysis.

## Project Structure

```
src/
  visualization/
    analyzer_core.py               # Core module for data retrieval and setup
    consumption_trend_analyzer.py  # Analyzes consumption trends over time
    heterogeneity_analyzer.py      # Analyzes consumption heterogeneity across demographics
    sentiment_analyzer.py          # Analyzes sentiment trends using normalized scores
    wordcloud_analyzer.py          # Generates word clouds for different policy periods
    main_analysis.py               # Orchestrates the execution of all analyzers
```

## Features

- **Enhanced Data Retrieval**: Robust data extraction from the simulation database with detailed error handling
- **Modular Analysis Pipeline**: Specialized analyzers for different aspects of policy evaluation
- **Visualization Improvements**:
  - Increased font sizes for better readability
  - Low-contrast color schemes for improved aesthetics
  - Normalized sentiment scores (0-1 scale)
  - Improved word clouds limited to 75 words per visualization
  - Box plots with improved styling and data point visualization
  - Data validation and informative logs during plotting
- **Policy Impact Assessment**: Specialized tools for measuring the voucher policy's effectiveness
- **Command-line Interface**: Flexible command options for running specific analyses

## Analysis Modules

1. **Consumption Trend Analyzer**: Tracks average consumption willingness over time and simulates different voucher scenarios.
2. **Heterogeneity Analyzer**: Examines consumption patterns across different demographic groups.
3. **Sentiment Analyzer**: Tracks sentiment evolution throughout the simulation with asynchronous processing.
4. **Word Cloud Analyzer**: Visualizes frequently discussed topics during different policy phases.

## Usage

To run the analysis, use the following command:

```
python -m src.visualization.main_analysis [options]
```

### Options:

- `--db_path PATH`: Specify the path to the simulation results database (default: uses latest results)
- `--analyzer TYPE`: Specify which analyzer to run (choices: consumption, heterogeneity, sentiment, wordcloud, all; default: all)

### Examples:

Run all analyzers:
```
python -m src.visualization.main_analysis
```

Run only consumption trend analysis:
```
python -m src.visualization.main_analysis --analyzer consumption
```

## Output

All visualizations are saved to the `visualizations` directory within the simulation results folder:

- **Consumption Trends**:
  - `avg_consumption_trend.png`: Line chart showing average consumption over time
  - `voucher_scenario_comparison.png`: Comparison of different voucher amount scenarios

- **Heterogeneity Analysis**:
  - `avg_consumption_by_[demographic].png`: Consumption patterns for different demographic groups
  - `heterogeneity_boxplot.png`: Box plot showing consumption distribution across groups

- **Sentiment Analysis**:
  - `sentiment_trends.png`: Evolution of positive and negative sentiment over time
  - `sentiment_distribution.png`: Box plots of sentiment distribution by policy phase

- **Word Clouds**:
  - `wordcloud_pre-policy.png`: Word cloud for the pre-policy period
  - `wordcloud_during_policy.png`: Word cloud for the policy implementation period
  - `wordcloud_post-policy.png`: Word cloud for the post-policy period

## Notes

- For best results, ensure the simulation database contains complete data for all epochs.
- The system is designed to handle missing or incomplete data gracefully.
- Each visualization includes detailed informational prints about the data being processed.
- Sentiment analysis is performed asynchronously with normalized scoring between 0-1.





