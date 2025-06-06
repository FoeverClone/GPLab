# Education Balance Subsystem

This subsystem simulates the dynamics of educational resource distribution, school choice, and housing market interactions in a society. It models how policy interventions in education can affect school quality, student achievement, residential segregation, and socioeconomic mobility.

## Components

The subsystem consists of two main components:

1. **EducationResourceSystem**: Manages school quality, student assignments, transfers, and educational outcomes.
2. **SchoolChoiceSystem**: Simulates housing market dynamics, residential choices, and their interaction with school quality.

## Key Features

### Education Resource System

- **School Quality Tracking**: Models different tiers of schools with varying quality levels
- **Student Achievement**: Tracks student performance based on school quality and individual factors
- **Policy Interventions**: Implements various policy levers:
  - Resource allocation modes (traditional, balanced, equalized)
  - Teacher rotation programs
  - Facility upgrade funds
  - Student subsidies
  - Transfer quotas
- **Socioeconomic Mobility**: Models how school quality affects future opportunities
- **Metrics**: Tracks quality gaps, achievement gaps, enrollment distributions, and more

### School Choice System

- **Housing Market**: Models how school quality affects housing prices in different districts
- **Residential Choices**: Simulates family decisions about where to live based on school quality, affordability, and commute
- **Segregation Dynamics**: Tracks residential segregation patterns over time
- **Housing Affordability**: Models how education policies affect housing affordability

## System Interactions

The two subsystems interact through the shared system state:

1. **EducationResourceSystem** shares:
   - School qualities for each district
   - Student achievement metrics
   - Enrollment pressures
   - Current education policies

2. **SchoolChoiceSystem** shares:
   - Housing prices for each district
   - District population distributions
   - Affordability metrics
   - Residential movement patterns

## Policy Simulation

The subsystem is designed to simulate the effects of different education policy approaches:

1. **Traditional Approach** (Epochs 0-1):
   - Resources concentrated in top schools
   - Open transfer policies
   - No subsidies or special programs

2. **Balanced Approach** (Epochs 2-3):
   - More equitable resource distribution
   - Teacher rotation programs
   - Subsidies for lower-performing schools
   - Limited transfer quotas

3. **Equalized Approach** (Epochs 4-5):
   - Aggressive quality equalization
   - Strict district enrollment (no transfers)
   - Increased subsidies
   - Facility upgrade funds for weaker schools

## Evaluation Metrics

The subsystem evaluates policy effectiveness through multiple dimensions:

- **Quality Metrics**: School quality gaps, facility improvements
- **Achievement Metrics**: Student performance, achievement gaps
- **Enrollment Metrics**: Distribution patterns, transfer activities
- **Housing Metrics**: Price gaps, affordability, residential segregation
- **Socioeconomic Metrics**: Mobility potential, opportunity distribution

## Usage

The education balance subsystem can be used to explore questions such as:

- How do different resource allocation policies affect educational equity?
- What are the trade-offs between school choice and residential segregation?
- How do housing markets respond to changes in school quality?
- What policy interventions are most effective at reducing achievement gaps?
- How do families make decisions about schools and housing?

## Configuration

The subsystem is configured through `config_education_balance.yaml`, which defines:

- School quality parameters
- Policy interventions by epoch
- Housing market parameters
- Agent decision attributes
- Required agent attributes
- Environment information provided to agents

## Implementation Details

- The subsystem uses a multi-agent simulation approach
- Agents make decisions about school choices and residential moves
- Policies evolve over multiple epochs
- The system tracks metrics and evaluates outcomes
- State is shared between subsystems through the system_state mechanism 