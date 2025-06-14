# Tax Incentive Policy Simulation

This simulation models the effects of tax incentive policies on economic behavior, social opinion, employment patterns, and consumption habits. The simulation is designed to evaluate how different tax policies can stimulate economic growth through consumption and investment.

## Subsystems

The simulation consists of four interconnected subsystems:

### 1. Economic Behavior System

This subsystem models how tax incentives affect agents' economic decisions regarding:
- Consumption levels
- Investment decisions
- Saving behaviors

Key metrics tracked include:
- GDP growth
- Investment growth
- Consumption growth
- Economic inequality (Gini coefficient)

### 2. Social Opinion System

This subsystem simulates how tax policies are perceived and discussed in society:
- Tracks public sentiment toward tax policies
- Models social media posts and discussions
- Analyzes opinion distribution across different demographic groups
- Evaluates the effectiveness of policy announcements

### 3. Employment Behavior System

This subsystem evaluates how tax incentives affect labor market dynamics:
- Employment rates
- Entrepreneurship rates
- Job creation
- Worker productivity
- Job satisfaction

### 4. Consumption Behavior System

This subsystem models detailed consumption patterns:
- Overall consumption levels
- Category-specific spending (essentials, leisure, durables, luxury)
- Response to targeted tax incentives for specific categories
- Consumer savings rates

## Policy Phases

The simulation runs through multiple policy phases:

1. **Baseline Phase (Epochs 0-1)**:
   - No special tax incentives
   - Standard consumption tax rate of 13%

2. **Investment Incentive Phase (Epochs 2-3)**:
   - 10% investment tax deduction
   - Tax-free savings up to 5,000 yuan
   - 8% entrepreneur tax credit
   - 3% employment income tax deduction
   - Targeted consumption tax reductions (2% for essentials, 5% for durables)

3. **Comprehensive Stimulus Phase (Epochs 4-5)**:
   - 5% general consumption tax reduction (to 11%)
   - 15% investment tax deduction
   - Tax-free savings up to 8,000 yuan
   - 12% entrepreneur tax credit
   - 5% employment income tax deduction
   - Enhanced category-specific consumption tax reductions

## Blackboard Communication

The subsystems communicate through a shared blackboard, sharing key metrics like:
- GDP estimates
- Employment rates
- Public sentiment
- Consumption levels

This allows for feedback effects between economic conditions, social perceptions, employment patterns, and consumption behaviors.

## Running the Simulation

To run the tax incentive policy simulation:

```bash
python src/main.py --config config/config_tax_incentive.yaml
```

## Evaluation

The simulation evaluates policy effectiveness through multiple metrics:
- Economic growth indicators
- Public sentiment and opinion distribution
- Employment and entrepreneurship rates
- Consumption patterns and category shifts
- Inequality measures

Time series data is collected for all key metrics to analyze trends over the course of the policy implementation. 