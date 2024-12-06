# Seamless Route Navigation for Wheelchair Users in Toronto
ECE1724H F3 (Fall) Special Topics in Software Engineering: Bio-inspired Algorithms for Smart Mobility
Course Project - Group 12

## Authors
Kaicheng Xu, Zhengan Du

## Project Overview
This project aims to solve multimodal pathfinding problems by:
- Generating multiple candidate paths using **Yen's K-Shortest Paths Algorithm**.
- Optimizing the generated paths using a **Genetic Algorithm (GA)** to find the most balanced route considering factors like **travel time**, **safety**, and **mode transitions**.

The project leverages data on **pedestrian and public transit networks** and incorporates real-world considerations like safety scores and incident data.

## Installation

To run this project, you'll need **Python 3.8+** and the following packages. You can easily install the required dependencies by following these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine:

```
git clone https://github.com/KaichengXu007/Bio-1724-course-project-Group-12.git
cd Bio-1724-course-project-Group-12
```

### Step 2: Install Required Packages
All the necessary packages are listed in `requirements.txt`. Run the following command to install them:

```
pip install -r requirements.txt
```

## Usage

### Running the Code
Ensure that your **network data** (e.g., `connected_network_cleaned.geojson`) is in the appropriate directory.
Change the start_node and end_node variables to "u" or "v" values in the **network data** to find paths from start_node to end_node.
To run the multimodal pathfinding solution, use the following command:

```
python multimodal_path_finding.py
```

This script will:

1. Load the graph data from the `connected_network_cleaned.geojson` file.
2. Use **Yen's K-Shortest Paths Algorithm** to generate candidate paths between the source and destination nodes.
3. Apply the **Genetic Algorithm** to optimize the generated paths based on multiple criteria.
4. Save and visualize the paths.
