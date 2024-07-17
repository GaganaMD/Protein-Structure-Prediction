# 🧬 Esmfold

Esmfold is a Gradio application designed to provide an interactive user interface for protein folding predictions and visualizations. This application leverages the `EsmForProteinFolding` model from Hugging Face, making advanced protein structure prediction accessible and user-friendly.

## Project Metadata

| Key            | Value                  |
|----------------|------------------------|
| **title**      | Esmfold                |
| **emoji**      | 🧬                     |
| **colorFrom**  | gray                   |
| **colorTo**    | blue                   |
| **sdk**        | gradio                 |
| **sdk_version**| 3.39.0                 |
| **app_file**   | app.py                 |
| **pinned**     | false                  |
| **license**    | Apache License 2.0     |

## Description

ESMFold is a state-of-the-art tool for protein folding predictions, utilizing the power of deep learning models developed by Meta AI Research. This application is built on the robust Gradio framework, providing an intuitive and interactive interface for researchers and enthusiasts in the field of bioinformatics and computational biology.

### Key Features

- **Accurate Protein Folding Predictions**: Utilizes the ESMForProteinFolding model to predict the 3D structure of proteins with high accuracy.
- **Interactive Interface**: Gradio-based interface allows users to input protein sequences and visualize their folded structures in 3D.
- **Embeddings Generation**: Provides options to generate embeddings for protein sequences using ESM and ESMFold models.
- **Sample Protein Suggestions**: Offers predefined protein sequences for quick testing and demonstration.
- **Code Integration**: Sample code snippets are provided to facilitate integration with other projects and workflows.

### Unique Properties

- **High Performance**: Powered by advanced deep learning models for precise and reliable predictions.
- **User-Friendly**: The Gradio interface simplifies the interaction, making it accessible even for users with limited programming experience.
- **Flexibility**: Can be used locally or through a hosted web application, offering flexibility based on user needs.
- **Visualization**: Integrated 3D visualization of protein structures helps in better understanding and analysis.

### Why ESMFold?

- **Cutting-Edge Technology**: Incorporates the latest advancements in protein folding from leading AI research, ensuring top-tier performance.
- **Ease of Use**: With an emphasis on usability, ESMFold lowers the barrier to entry for protein folding analysis, enabling more researchers to utilize this powerful tool.
- **Comprehensive Toolset**: Beyond folding predictions, ESMFold provides additional features such as embeddings generation and sample protein suggestions, making it a comprehensive tool for protein analysis.

## Setup

### Prerequisites

- Python 3.8+
- `streamlit`
- `stmol==0.0.9`
- `py3Dmol`
- `biotite`
- `ipywidgets`
- `ipython_genutils`
- `gradio-client`
- `gradio`
- `transformers`
- `torch`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/GaganaMD/Protein-Structure-Prediction.git
    cd Protein-Structure-Prediction
    ```

2. Install the required libraries:
    ```sh
    pip install torch transformers gradio
    ```
## Verifying the Installation

After installing the required libraries, you can verify the installation by running the following command to check if Gradio and other dependencies are correctly installed:

```sh
pip list | grep -E "torch|transformers|gradio"

## Accessing the Gradio Interface

Once you have installed the necessary libraries and verified their installation, you can access the Gradio interface to start using Esmfold for protein folding predictions and visualizations.

1. **Launch the Application**:
   - Ensure you are in the project directory where `app.py` is located.
   - Run the following command to start the Gradio interface:
     ```sh
     python app.py
     ```

2. **Interacting with the Interface**:
   - Open your web browser and navigate to the URL displayed in the terminal after launching the application (`localhost:XXXX`).
   - You will see the interactive interface where you can input protein sequences, visualize their folded structures in 3D, and explore other functionalities provided by Esmfold.

3. **Explore Sample Proteins**:
   - Use the dropdown menu in the interface to select from predefined protein sequences such as "Plastic degradation protein", "Antifreeze protein", "AI Generated protein", or "7-bladed propeller fold" for quick testing and demonstration purposes.

4. **Custom Integration**:
   - Utilize the provided sample code snippets to integrate Esmfold into your own projects or workflows for advanced protein structure prediction tasks.

5. **Feedback and Support**:
   - If you have any questions, feedback, or encounter issues while using Esmfold, please refer to the [Issues section](https://github.com/GaganaMD/Protein-Structure-Prediction/issues) of the GitHub repository for support.
By following these steps, you can effectively leverage Esmfold for protein folding predictions and explore its capabilities through an intuitive Gradio-based interface.
