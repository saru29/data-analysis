

# Project 11 FIT File Handling and Data Pipeline

This repository contains the documentation and code for a project related to Wahoo fitness devices. The project involves working with the Wahoo Cloud API and establishing a connection with the Wahoo Kickr trainer to retrieve live data.

## Contents

The repository is organized into the following directories:

### Project Documentation

The `Project Documentation` directory contains project documentation, including:

- Project Scope: This document outlines the objectives and scope of the project, defining what is in and out of its scope.
- Handover Documentation: A comprehensive guide for handing over the project to stakeholders or other team members.
- Data Pipeline / Workflow: Documentation describing the data pipeline and workflow involved in the project.

### Research

The `Research` directory contains code and findings from a trial run of the Wahoo API. It serves as a testing ground for exploring the capabilities of the Wahoo Cloud API and understanding its functionality.

### Wahoo_Cloud_API

The `Wahoo_Cloud_API` directory contains the core implementation of the project. It focuses on interacting with the Wahoo Cloud API to retrieve FIT files. FIT files are commonly used to store fitness-related data, such as workouts, activities, and health metrics.

### Wahoo_Kickr_Connection

The `Kickr_Connection` directory contains a Python script that establishes a connection with the Wahoo Kickr trainer. It enables the retrieval of live data from the trainer, such as speed, power, and cadence. This script can be used to monitor and analyze real-time performance during workouts.

### Google Cloud Platform Project & Bigquery Sandbox data environment
The `GCP_Bigquery` directory contains infomation about the GCP Project `sit-23t1-fit-data-pipe-ee8896e` created and the Bigquery data warehouse. The Bigquery data warehouse under pins many of Data/AI's Trimester 2 projects. 

## Getting Started

To get started with this project, please follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/project-documentation.git`
2. Navigate to the relevant directory for the component you wish to explore (`Documentation`, `Research`, `Wahoo_Cloud_API`, `GCP_Bigquery` or `Kickr_Connection`).
3. Follow the instructions provided in each directory's respective README.md file to set up and run the code.

## Requirements

Make sure you have the following requirements fulfilled before using this project:

- Python (version 3.11)
- Wahoo Cloud API credentials (API key, secret)
- Wahoo Kickr trainer
- A road bike
- Additional dependencies (specified in each component's readme.md)

Please refer to the individual readme.md files or pdf documented in each directory for details concerning set up and project scope. 

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code according to the terms and conditions specified in the license.

## Contributors

- [Mark Telley](https://github.com/marktelley)

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Make sure to adhere to the project's coding conventions and guidelines.
