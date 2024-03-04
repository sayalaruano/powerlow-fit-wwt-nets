# Importing libraries
import os
import powerlaw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import mpld3
import streamlit.components.v1 as components

# Function to load the data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to plot power law fit with Plotly
def plot_power_law_plotly(df):
    # Fit the data to a power law distribution
    data = df['Degree'].values
    fit = powerlaw.Fit(data)

    # Sorting data for plotting
    sorted_data = np.sort(data)
    
    # Prepare plot data
    fig = go.Figure()

    # Empirical CCDF
    empirical_ccdf = np.array([fit.ccdf(datum) for datum in sorted_data])
    fig.add_trace(go.Scatter(x=sorted_data, y=empirical_ccdf, mode='lines', name='Empirical Data', line=dict(width=3)))

    # Power law fit CCDF
    # Using the fitted power law distribution to calculate CCDF
    theoretical_ccdf = np.array([fit.power_law.ccdf(datum) for datum in sorted_data])
    fig.add_trace(go.Scatter(x=sorted_data, y=theoretical_ccdf, mode='lines', name='Power law fit', line=dict(color='red', dash='dash')))

    # Update plot layout
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='p(X≥x)',
        yaxis_type='log', 
        xaxis_type='log',
        legend=dict(y=0.5, traceorder='reversed', font_size=16)
    )
    
    return fig

def plot_power_law_matplotlib(df, compare_distribution):
    # Fit the data to a power law distribution
    data = df['Degree'].values
    data = data[data > 0]

    # Check if data is not empty after filtering
    if len(data) == 0:
        raise ValueError("No positive data available to plot after filtering.")
    
    fit = powerlaw.Fit(data)

    # Create a figure with a specific size
    fig, ax = plt.subplots(figsize=(7, 3.5))  # Adjust figure size if needed

    # Plot the empirical data
    fit.plot_ccdf(linewidth=3, label='Empirical Data', ax=ax)
    
    # Plot the power law fit
    fit.power_law.plot_ccdf(color='orange', linestyle='--', label='Power law fit', ax=ax)

    # Plot the selected comparison distribution dynamically
    if hasattr(fit, compare_distribution):
        getattr(fit, compare_distribution).plot_ccdf(color = 'g', linestyle='--', label=f'{compare_distribution.capitalize()} fit', ax=ax)

    # Add labels and title
    ax.set_ylabel(u"p(X≥x)")
    ax.set_xlabel("x")

    # Adjust the legend
    ax.legend(loc=3)

    return fig

def compare_distributions(df, distribution1, distribution2):
    # Filter data for positive values
    data = df['Degree'].values
    data = data[data > 0]

    if len(data) == 0:
        return None, None  # Return None if no positive data

    fit = powerlaw.Fit(data)
    R, p = fit.distribution_compare(distribution1, distribution2, normalized_ratio=True)
    return R, p

# Streamlit app layout and functionality
st.set_page_config(page_title="App to fit power law distributions", layout="wide")
st.title('App to check the goodness of fit of network\'s degree distributions')

biomes = ['Wastewater', 'Water and sludge', 'Industrial wastewater', 'Activated sludge']
biome_folder_names = ["Wastewater", "Wastewater_Water_and_sludge", "Wastewater_Industrial_wastewater", "Wastewater_Activated_Sludge"]
biome = st.selectbox('Select the biome to analyze', biomes)

# List of available distributions for comparison
available_distributions = ['lognormal', 'exponential', 'truncated_power_law', 'stretched_exponential', 'lognormal_positive']
selected_distribution = st.selectbox('Select a distribution to compare with power law fit', available_distributions)

st.write("Select the threshold and the complementary cumulative distribution function (CCDF) plots will be displayed.")

# Define file paths
file_paths = [
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_010_degree_distribution.csv',
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_020_degree_distribution.csv',
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_030_degree_distribution.csv',
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_040_degree_distribution.csv',
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_045_degree_distribution.csv',
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_047_degree_distribution.csv',
    f'Data/{biome_folder_names[biomes.index(biome)]}/{biome_folder_names[biomes.index(biome)]}_cclasso_050_degree_distribution.csv'
]

# List of thresholds
thresholds = [0.10, 0.20, 0.30, 0.40, 0.45, 0.47, 0.50]

# Create tabs for each plot adding the threshold to the tab name
tab_names = [f"CCLasso_ {threshold}" for threshold in thresholds]
# tab_names = [f"Plot {idx+1}" for idx, _ in enumerate(file_paths)]
tabs = st.tabs(tab_names)

for idx, file_path in enumerate(file_paths):
    with tabs[idx]:
        if os.path.exists(file_path):
            df = load_data(file_path)
            try:
                fig = plot_power_law_matplotlib(df, selected_distribution)
                st.pyplot(fig, use_container_width=False)
                # fig_html = mpld3.fig_to_html(fig)
                # components.html(fig_html, height=500, width=900)

                # Perform and display the distribution comparison
                R, p = compare_distributions(df, 'power_law', selected_distribution)
                if R is not None and p is not None:
                    st.write(f"Loglikelihood ratio between Power Law and {selected_distribution.capitalize()}: {R:.2f}")
                    st.write(f"Significance of the comparison (p-value): {p:.2f}")
                else:
                    st.write("Not enough positive data for distribution comparison.")
                
            except ValueError as e:
                st.error(str(e))
        else:
            st.error(f"Data file not found for {tab_names[idx]}. Please check the file path or the data availability.")







