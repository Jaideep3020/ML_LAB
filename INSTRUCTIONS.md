# Business Entity Resolution - Usage Instructions

Welcome to the Business Entity Resolution project! This document outlines how to set up the environment, install dependencies, and run the codebase on your local machine.

## Prerequisites

- **Python**: Version 3.8 or higher is recommended.
- **Git**: To clone the repository.
- **Virtual Environment** (Optional but recommended): `venv` or `conda` to manage dependencies.

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Jaideep3020/ML_LAB.git
   cd ML_LAB
   ```

2. **Create a Virtual Environment (Optional)**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   Navigate to the core project directory and install the required packages:
   ```bash
   cd business_entity_resolution
   pip install -r requirements.txt
   ```

## Running the Code

The main entry point for this system is `run_pipeline.py`. 

1. **Configure Execution**
   You can adjust any runtime parameters, thresholds, or data paths in `config.json` before running.

2. **Execute the Pipeline**
   Run the following command from within the `business_entity_resolution` directory:
   ```bash
   python run_pipeline.py
   ```

3. **Check Results**
   Output metrics and results (such as `matching_results.tsv`) will be generated based on the configurations. 
   You can run analysis and checks on the results using the included helper scripts:
   ```bash
   python check_results.py
   python analyze_validation.py
   ```

## Repository Structure Overview

- `business_entity_resolution/`: The core source code directory.
  - `run_pipeline.py`: Main executable script for the resolution pipeline.
  - `requirements.txt`: Python package dependencies.
  - `config.json`: Configuration settings.
  - `ber/`: Core modules for data loading, metrics, model evaluation, and retrieval.
- `error_analysis_report.md` & `final_submission_execution_plan_1.md`: Project notes and performance analysis documentation.
