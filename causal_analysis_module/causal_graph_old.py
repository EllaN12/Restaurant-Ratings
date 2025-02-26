#%%
import os
import os
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

#for causal_grapgh
#from causalgraphicalmodels import CausalGraphicalModel
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
import logging, sys
import pandas_flavor as pf



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

#@pf.register_dataframe_method
def webscrape_report():
    """
    Web scrapes a Ydata EDA HTML report to extract information about data quality alerts,
    specifically focusing on high correlations.

    Returns:
        A dictionary where keys are variable names with high correlation alerts,
        and values are the corresponding "other fields" information.
    """

    file_path = os.path.abspath("Reports/data_report.html")
    file_url = "file://" + file_path
    print(f"File URL: {file_url}")  # Print for debugging

    executable_path = ChromeDriverManager().install()
    service = Service(executable_path)
    options = Options()

    # Options to try to avoid detection (might not always be successful)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=service, options=options)
    correlation_dict = {}

    try:
        driver.get(file_url)
        html = driver.page_source
        soupy = soup(html, 'html.parser')

        alerts_table = soupy.find('div', id='tab-pane-overview-alerts').find('table', class_='table-striped')

        if alerts_table:
            for row in alerts_table.find_all('tr'):
                cells = row.find_all('td')
                if cells:  # Check if the row has cells
                    link = cells[0].find('a')
                    if link:
                        variable_name = link.text.strip()
                        message = cells[0].text.replace(variable_name, "").replace(" has constant value '?'", "").replace(" is highly overall correlated with ", "").replace(" is highly imbalanced (", "").replace(" has ", "").replace(" missing values", "").replace(" is uniformly distributed", "").strip()
                        badge = cells[1].find('span', class_='badge')
                        alert_type = badge.text.strip() if badge else None
                        other_fields_span = cells[0].find('span', attrs={'data-bs-toggle': 'tooltip'})
                        other_fields = other_fields_span['data-bs-title'] if other_fields_span else None

                        print(f"Variable: {variable_name}")  # Print each field for debugging
                        print(f"Message: {message}")
                        print(f"Alert Type: {alert_type}")
                        print(f"Other Fields: {other_fields}")
                        print("-" * 20)

                        if alert_type == "High correlation":
                            correlation_dict[variable_name] = other_fields

        else:
            print("Alerts table not found.")

    except Exception as e: # Handle potential exceptions
        print(f"An error occurred: {e}")

    finally:
        driver.quit()  # Ensure driver quits in all cases

    # Print summary of high correlations after scraping
    print("\nHigh Correlation Summary:")
    for variable_name, fields in correlation_dict.items():
        print(f"Variable: {variable_name}, Other Fields: {fields}")

    return correlation_dict



#%%
def causal_digraph_for_dowhy(correlation_dict: Dict[str, str]) -> nx.DiGraph:
    """
    Creates a causal graph for use with DoWhy.
    
    Args:
        correlation_dict: A dictionary of correlations, where keys are source variables
                          and values are comma-separated target variables.
    
    Returns:
        A networkx DiGraph object ready for use with DoWhy's CausalModel.
    """
    def parse_relationships_dict(correlation_dict: Dict[str, str]) -> List[Tuple[str, str]]:
        """Parse the relationships dictionary to create edges with error handling."""
        
        additional_edges = []

        if not correlation_dict:
            print("Warning: Empty correlation dictionary provided")
            return additional_edges

        for source, targets in correlation_dict.items():
            if not targets:
                print(f"Warning: No targets found for source '{source}'")
                continue

            try:
                target_list = [t.strip() for t in targets.split(',') if t.strip()]

                for target in target_list:
                    if target:
                        additional_edges.append((str(source), str(target)))

            except Exception as e:
                print(f"Error processing source '{source}': {str(e)}")
                continue

        print(f"Successfully created {len(additional_edges)} edges from correlation dictionary")
        return additional_edges

    # Base causal relationships
    base_edges = [
        # Direct Effects on Rating
        ('food_rating', 'rating'),
        ('service_rating', 'rating'),
        ('age_group', 'rating'),
        ('activity', 'rating'),
        ('personality', 'rating'),
        ('User_cuisine', 'rating'),
        
        # Demographic Effects
        ('age_group', 'budget'),
        ('age_group', 'User_cuisine'),
        ('age_group', 'drink_level'),
        ('age_group', 'dress_preference'),
        ('activity', 'budget'),
        ('activity', 'transport'),
        ('marital_status', 'hijos'),
        ('marital_status', 'budget'),
        ('marital_status', 'user_ambience'),
        
        # Restaurant Selection Effects
        ('transport', 'accessibility'),
        ('transport', 'area'),
        ('budget', 'price'),
        
        # Personal Attribute Effects
        ('weight', 'food_rating'),
        ('personality', 'food_rating'),
        ('height', 'food_rating'),
        ('color', 'food_rating'),
        ('hijos', 'food_rating'),
        
        # Service Rating Influences
        ('color', 'service_rating'),
        ('personality', 'service_rating'),
        
        # Transport and Accessibility Chain
        ('transport', 'smoker'),
        ('transport', 'dress_preference'),
        ('transport', 'weight')
    ]

    try:
        correlation_edges = parse_relationships_dict(correlation_dict)

        # Create a DiGraph directly
        G = nx.DiGraph()
        G.add_edges_from(base_edges)
        logging.info(f"Added {len(base_edges)} base edges")

        if correlation_edges:
            G.add_edges_from(correlation_edges)
            logging.info(f"Added {len(correlation_edges)} correlation edges")

        # Log graph information
        logging.info(f"Created causal graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        return G  # Return the DiGraph directly

    except Exception as e:
        logging.error(f"Error creating causal graph: {str(e)}")
        # Fallback: Basic graph with only base edges
        G = nx.DiGraph()
        G.add_edges_from(base_edges)
        logging.warning("Returning fallback causal graph with only base edges")
        return G




#%%
def convert_nx_to_dowhy_format(G: nx.DiGraph) -> str:
    """
    Convert a NetworkX DiGraph to DoWhy's DOT string format.
    
    Args:
        G: NetworkX DiGraph representing the causal model
        
    Returns:
        A DOT format string compatible with DoWhy
    """
    dot_string = "digraph {\n"
    
    # Add all edges to the DOT string
    for source, target in G.edges():
        dot_string += f"    {source} -> {target};\n"
    
    dot_string += "}"
    return dot_string









#%%

#@pf.register_dataframe_method
def create_causal_visualization(cgm: CausalGraphicalModel, 
                              figsize: tuple = (20, 16)) -> plt.Figure:
    """
    Create a visualization of the causal graph with node categories
    """
    # Get the NetworkX graph
    G = cgm.dag
    
    # Define node categories and their colors
    node_categories = {
        'rating': ['rating'],
        'direct_effects': ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine'],
        'demographics': [ 'marital_status'],
        'preferences': ['budget', 'drink_level', 'dress_preference'],
        'transport': ['transport', 'accessibility', 'area'],
        'personal': ['weight', 'height', 'color', 'hijos', 'smoker'],
        'restaurant': ['Rcuisine_y', 'price', 'user_ambience']
    }
    

    colors = {
        'rating': '#ff6666',        # Red
        'direct_effects': '#ffa366', # Orange
        'demographics': '#66b3ff',   # Blue
        'preferences': '#66ff66',    # Green
        'transport': '#ffff66',      # Yellow
        'personal': '#ff66ff',       # Pink
        'restaurant': '#a366ff'      # Purple
    }
    
    # Set up the plot
    plt.figure(figsize=figsize)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes by category
    node_colors = []
    for node in G.nodes():
        # Find which category the node belongs to
        category = 'other'
        for cat, nodes in node_categories.items():
            if node in nodes:
                category = cat
                break
        node_colors.append(colors.get(category, '#gray'))
    
    # Draw the network
    nx.draw(G, pos,
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            font_weight='bold',
            with_labels=True,
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            arrowstyle='->',
            node_shape='o')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], 
                                 marker='o', 
                                 color='w',
                                 markerfacecolor=color,
                                 markersize=10,
                                 label=category.replace('_', ' ').title())
                      for category, color in colors.items()]
    
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              title='Node Categories',
              title_fontsize=12,
              fontsize=10)
    
    # Add title
    plt.title("Restaurant Rating Causal Graph", pad=20, size=16)
    
    # Remove axes
    plt.axis('off')
    
    # Add graph statistics as text
    stats_text = f"Nodes: {len(G.nodes())}\n"
    stats_text += f"Edges: {len(G.edges())}\n"
    stats_text += f"Average degree: {sum(dict(G.degree()).values())/len(G):0.2f}"
    
    plt.text(0.95, 0.05, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='bottom',
            horizontalalignment='right')
    
    plt.tight_layout()
    return plt


#%%







#%%
