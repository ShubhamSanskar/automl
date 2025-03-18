import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def dataset(path_to_dataset, split_in_test_and_train=0.8, **kwargs):
    """
    Loads dataset, splits data, and visualizes specified graphs.

    Parameters:
    - path_to_dataset (str): Path to the dataset file (CSV format recommended).
    - split_in_test_and_train (float): Proportion of data to use for training (default is 0.8).
    - **kwargs: Graph configurations for visualizations (e.g., bargraph, lineplot, etc.).

    Returns:
    - train_df (DataFrame): Training dataset.
    - test_df (DataFrame): Testing dataset.
    """

    # Error Handling for Dataset Loading
    try:
        df = pd.read_csv(path_to_dataset)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{path_to_dataset}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: The file '{path_to_dataset}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"Error: The file '{path_to_dataset}' is not properly formatted.")

    # Split the dataset
    try:
        if not (0 < split_in_test_and_train < 1):
            raise ValueError("Error: 'split_in_test_and_train' must be between 0 and 1.")
        train_df, test_df = train_test_split(df, test_size=1 - split_in_test_and_train, random_state=42)
    except Exception as e:
        raise RuntimeError(f"Dataset split error: {str(e)}")

    # Graph Visualizations
    graph_functions = {
        'bargraph': plot_bar_graph,
        'lineplot': plot_line_plot,
        'scatterplot': plot_scatter_plot,
        'histogram': plot_histogram,
        'boxplot': plot_box_plot,
        'piechart': plot_pie_chart,
        'heatmap': plot_heatmap,
        'confusion_matrix': plot_confusion_matrix,
        'pairplot': plot_pair_plot,
        'violinplot': plot_violin_plot
    }

    for graph_type, graph_config in kwargs.items():
        if graph_type in graph_functions:
            try:
                graph_functions[graph_type](df, graph_config)
            except Exception as e:
                print(f"Warning: Error in {graph_type} - {str(e)}")

    return train_df, test_df

# Visualization Functions with Enhanced Error Handling
def plot_bar_graph(df, config):
    try:
        x, y = config.get("xtitle"), config.get("ytitle")
        check_columns(df, x, y)
        plt.bar(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{x} vs {y}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Bar Graph Error: {str(e)}")

def plot_line_plot(df, config):
    try:
        x, y = config.get("xtitle"), config.get("ytitle")
        check_columns(df, x, y)
        plt.plot(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{x} vs {y}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Line Plot Error: {str(e)}")

def plot_scatter_plot(df, config):
    try:
        x, y = config.get("xtitle"), config.get("ytitle")
        check_columns(df, x, y)
        plt.scatter(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{x} vs {y}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Scatter Plot Error: {str(e)}")

def plot_histogram(df, config):
    try:
        column = config.get("column")
        check_column(df, column)
        plt.hist(df[column], bins=config.get("bins", 10), color='skyblue', edgecolor='black')
        plt.title(f"Histogram of {column}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Histogram Error: {str(e)}")

def plot_box_plot(df, config):
    try:
        column = config.get("column")
        check_column(df, column)
        sns.boxplot(x=df[column])
        plt.title(f"Box Plot of {column}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Box Plot Error: {str(e)}")

def plot_pie_chart(df, config):
    try:
        column = config.get("column")
        check_column(df, column)
        plt.pie(df[column].value_counts(), labels=df[column].unique(), autopct='%.1f%%')
        plt.title(f"Pie Chart of {column}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Pie Chart Error: {str(e)}")

def plot_heatmap(df, config):
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Heatmap Error: {str(e)}")

def plot_confusion_matrix(df, config):
    try:
        actual = config.get("actual")
        predicted = config.get("predicted")
        check_columns(df, actual, predicted)
        cm = confusion_matrix(df[actual], df[predicted])
        ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Confusion Matrix Error: {str(e)}")

def plot_pair_plot(df, config):
    try:
        sns.pairplot(df)
        plt.title("Pair Plot")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Pair Plot Error: {str(e)}")

def plot_violin_plot(df, config):
    try:
        column = config.get("column")
        check_column(df, column)
        sns.violinplot(x=df[column])
        plt.title(f"Violin Plot of {column}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Violin Plot Error: {str(e)}")

# Utility Functions
def check_columns(df, *columns):
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Error: Column '{col}' not found in dataset.")

def check_column(df, column):
    if column not in df.columns:
        raise ValueError(f"Error: Column '{column}' not found in dataset.")
