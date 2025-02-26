import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
from IPython.display import Image
import pandas as pd
import os 
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re
import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph
import pydot
import io
import tempfile
from causal_analysis_module.analysis import data_preprocessing
from dowhy import CausalModel



df = data_preprocessing()

def create_causal_graph_spec():
    """
    Create the causal graph specification for DoWhy
    """
    graph = """
    digraph {
        # Direct Effects on Rating
        food_rating -> rating;
        service_rating -> rating;
        age_group -> rating;
        activity -> rating;
        personality -> rating;
        User_cuisine -> rating;
        
        # Demographic Effects
        age_group -> budget;
        age_group -> User_cuisine;
        age_group -> drink_level;
        age_group -> dress_preference;
        
        activity -> budget;
        activity -> transport;
        
        marital_status -> hijos;
        marital_status -> budget;
        marital_status -> user_ambience;
        
        # Restaurant Selection Effects
        transport -> accessibility;
        transport -> area;
        budget -> price;
        
        # Personal Attribute Effects
        weight -> food_rating;
        personality -> food_rating;
        height -> food_rating;
        color -> food_rating;
        hijos -> food_rating;
        
        # Service Rating Influences
        color -> service_rating;
        personality -> service_rating;
        
        # Transport and Accessibility Chain
        transport -> smoker;
        transport -> dress_preference;
        transport -> weight;
    }
    """
    return graph


def create_and_visualize_graph(graph_spec, plot=True, treatment_nodes=None, outcome_node="rating"):
    """
    Creates a directed acyclic graph (DAG) from a graphviz specification and visualizes it.
    Uses pydot instead of pygraphviz for better compatibility.
    
    Args:
        graph_spec: A string containing the graphviz graph specification.
        plot: If True, plots the graph using matplotlib.
        treatment_nodes: List of treatment nodes to highlight in a single color.
        outcome_node: The outcome node to highlight in red.
    
    Returns:
        A networkx.DiGraph object representing the DAG.
    """
    # Default treatment nodes if None provided
    if treatment_nodes is None:
        treatment_nodes = ['food_rating', 'service_rating', 'age_group', 'activity', 
                          'personality', 'User_cuisine']
    
    # Parse with pydot
    graphs = pydot.graph_from_dot_data(graph_spec)
    if not graphs:
        raise ValueError("Could not parse the DOT graph specification")
    
    # Convert to networkx graph
    graph = nx.nx_pydot.from_pydot(graphs[0])
    
    if plot:
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph, seed=42)  # Consistent layout
        
        # Define colors for node categories
        treatment_color = 'green'  # Single color for all treatment nodes
        outcome_color = 'red'      # Red for outcome
        other_color = 'skyblue'    # Sky blue for all other nodes
        
        # Create a node list and corresponding color list
        all_nodes = list(graph.nodes())
        node_colors = []
        
        for node in all_nodes:
            if node == outcome_node:
                node_colors.append(outcome_color)
            elif node in treatment_nodes:
                node_colors.append(treatment_color)
            else:
                node_colors.append(other_color)
        
        # Draw the graph with node colors
        nx.draw(graph, pos, 
                nodelist=all_nodes,
                node_color=node_colors,
                with_labels=True, 
                node_size=700, 
                font_size=8,
                arrowstyle='-|>', 
                arrowsize=15)
        
        # Create simplified legend with categories
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=treatment_color, 
                      markersize=10, label='Treatments'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=outcome_color, 
                      markersize=10, label='Outcome'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=other_color, 
                      markersize=10, label='Other Variables')
        ]
        
        # Place legend
        plt.legend(handles=legend_handles,
                  title="Variable Categories",
                  loc='lower right',
                  frameon=True,
                  framealpha=0.8)
        
        plt.title("Causal Graph with Treatment Variables and Outcome")
        plt.tight_layout()
        plt.show()
    
    return graph
