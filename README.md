# Prosperity-2

## Prerequisites

- Python 3.12

## Folder Structure

- **data**: This folder contains the project's data in CSV format.
- **analysis**: This folder is dedicated to the analysis of the data using Python notebooks.
- **datamodel.py**: This file contains a list of classes provided by Prosperity Wiki.
- **trader.py**: This file contains our trading solution.
- **runner.py**: This file is used to execute our trader. To start the runner, simply execute: 
    ```
    python runner.py
    ```

## Installation

To ensure a clean and isolated environment, it's recommended to use a virtual environment. Here's how you can set up the environment:

1. **Create a Virtual Environment**: Run the following command in your terminal to create a virtual environment named `venv`:

    ```
    python -m venv venv
    ```

2. **Activate the Virtual Environment**: Activate the virtual environment using the appropriate command for your operating system:

    For macOS/Linux:

    ```
    source venv/bin/activate
    ```

    For Windows:

    ```
    venv\Scripts\activate
    ```

3. **Install Requirements**: Once the virtual environment is activated, install the project dependencies using pip:

    ```
    pip install -r requirements.txt
    ```