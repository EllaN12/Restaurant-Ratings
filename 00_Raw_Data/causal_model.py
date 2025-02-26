#%%
import networkx as nx
from dowhy import CausalModel
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
import pandas_flavor as pf
#import causal_analysis_module as cam
from causal_analysis_module.analysis import data_preprocessing
from causal.bin.causal_graph import webscrape_report, causal_digraph, create_causal_visualization

#%%


df = data_preprocessing()


#correlation_dict = webscrape_report()
G = create_causal_visualization(
    cgm= causal_digraph(correlation_dict=webscrape_report())
)


#%%
#model = 









#%%
@pf.register_dataframe_method
def find_common_causes(G, treatment, outcome):
    """
   # Identifies common causes between treatment and outcome
    """
    treatment_ancestors = nx.ancestors(G, treatment)
    outcome_ancestors = nx.ancestors(G, outcome)
    return list(treatment_ancestors.intersection(outcome_ancestors))

treatment = ['food_rating', 'service_rating', 'age_group', 'activity', 'lifestyle_cluster', 'personality', 'User_cuisine']
cgm = causal_digraph(correlation_dict=webscrape_report())

find_common_causes (
    G= cgm.dag,
    treatment = 'User_cuisine',
    outcome= 'rating'

)
#%%

@pf.register_dataframe_method
def create_causal_model(data, graph_definition):
    """
    #Creates a causal model with multiple treatment-outcome relationships based on the graph
    
    #Parameters:
    #data (pd.DataFrame): The dataset containing all variables
    #graph_definition (str): The graph definition in DOT format
    
    #Returns:
    #dict: Dictionary of CausalModel objects for different treatment-outcome pairs
    """
    # Convert DOT format to NetworkX graph
    G = nx.DiGraph(nx.nx_pydot.read_dot(graph_definition))
    
    # Find all direct effects on rating
    direct_effects = [edge for edge in G.edges() if edge[1] == 'rating']
    
    # Create separate causal models for each treatment
    causal_models = {}
    
    for treatment_edge in direct_effects:
        treatment = treatment_edge[0]
        
        # Create model for each treatment-outcome pair
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome='rating',
            graph=graph_definition,
            common_causes=find_common_causes(G, treatment, 'rating'),
            instruments=find_instruments(G, treatment, 'rating')
        )
        
        causal_models[treatment] = model
    
    return causal_models



@pf.register_dataframe_method
def find_instruments(G, treatment, outcome):
    """
    #Identifies potential instrumental variables
    """
    # Find nodes that affect treatment but not outcome directly
    treatment_ancestors = nx.ancestors(G, treatment)
    outcome_ancestors = nx.ancestors(G, outcome)
    
    potential_instruments = treatment_ancestors - outcome_ancestors
    
    # Verify instrumental variable assumptions
    valid_instruments = []
    for instrument in potential_instruments:
        # Check if instrument only affects outcome through treatment
        paths_to_outcome = list(nx.all_simple_paths(G, instrument, outcome))
        valid = all(treatment in path for path in paths_to_outcome)
        
        if valid:
            valid_instruments.append(instrument)
    
    return valid_instruments




@pf.register_dataframe_method
def identify_effects(causal_models):
    """
    #Identifies causal effect for each treatment-outcome pair
    """""
    Parameters:
    causal_models (dict): Dictionary of CausalModel objects
    
    Returns:
    dict: Dictionary of identified estimands
    """
    identified_effects = {}
    
    for treatment, model in causal_models.items():
        try:
            identified_estimand = model.identify_effect()
            identified_effects[treatment] = identified_estimand
        except Exception as e:
            print(f"Could not identify effect for {treatment}: {str(e)}")
            identified_effects[treatment] = None
    
    return identified_effects


@pf.register_dataframe_method
def estimate_effects(causal_models, identified_effects, method='backdoor.linear_regression'):
    """""
    Estimates causal effects for each treatment-outcome pair
    
    Parameters:
    causal_models (dict): Dictionary of CausalModel objects
    identified_effects (dict): Dictionary of identified estimands
    method (str): Estimation method to use
    
    Returns:
    dict: Dictionary of estimated effects
    """
    estimates = {}
    
    for treatment, model in causal_models.items():
        if identified_effects[treatment] is not None:
            try:
                estimate = model.estimate_effect(
                    identified_effects[treatment],
                    method_name=method
                )
                estimates[treatment] = estimate
            except Exception as e:
                print(f"Could not estimate effect for {treatment}: {str(e)}")
                estimates[treatment] = None
    
    return estimates



# Create causal models
models = create_causal_model(df, graph_definition)

# Identify effects
identified_effects = identify_effects(models)

# Estimate effects
estimates = estimate_effects(models, identified_effects)

# Access results for specific treatment
food_rating_effect = estimates.get('food_rating')

# %%
