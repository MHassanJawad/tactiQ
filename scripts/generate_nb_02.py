import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell('# 02: Visualizing Generated Graphs\nLet\'s physically inspect what the Graph Neural Network sees.'),
    nbf.v4.new_code_cell('import torch\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport os\n\ndata_path = r"d:\\NUST\\6th sem\\Machine Learning\\project\\graphs\\la_liga_2015_16_train.pt"\nif os.path.exists(data_path):\n    graphs = torch.load(data_path)\n    print(f"Loaded {len(graphs)} graphs!")\nelse:\n    print("Graphs file not found yet!")'),
    nbf.v4.new_markdown_cell('## Plotting a Single Graph on the Pitch\nThe GNN sees a normalized [0,1] pitch. Let\'s scale it back to 120x80 yards and draw the connections.'),
    nbf.v4.new_code_cell('''def draw_pitch():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_facecolor('#2b8a3e') # Green grass
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color='white') # Pitch outline
    ax.plot([60, 60], [0, 80], color='white') # Halfway line
    ax.scatter(60, 40, color='white', marker='o') # Center dot
    return fig, ax

def plot_graph(graph_data):
    fig, ax = draw_pitch()
    
    # 1. Scale coordinates up from [0, 1] back to [0, 120] and [0, 80]
    coords = graph_data.x[:, :2].numpy()
    x_coords = coords[:, 0] * 120
    y_coords = coords[:, 1] * 80
    
    # 2. Extract teams (Team 0=Possession, Team 1=Defending, 0.5=Ball)
    team_flags = graph_data.x[:, 2].numpy()
    
    # 3. Draw Edges (spatial mathematical connections)
    edge_index = graph_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        start_idx = edge_index[0, i]
        end_idx = edge_index[1, i]
        ax.plot([x_coords[start_idx], x_coords[end_idx]], 
                [y_coords[start_idx], y_coords[end_idx]], 
                color='yellow', alpha=0.15, linewidth=1.0)
                
    # 4. Draw Nodes
    for i in range(len(x_coords)):
        if team_flags[i] == 0.0:
            color = '#3399ff' # Blue (Possession)
        elif team_flags[i] == 1.0:
            color = '#ff3333' # Red (Defending)
        else:
            color = 'white'   # Ball

        ax.scatter(x_coords[i], y_coords[i], color=color, s=150, zorder=5, edgecolors='black')
        
        # Highlight the actual ball
        if color == 'white':
            ax.scatter(x_coords[i], y_coords[i], color='white', s=250, zorder=6, edgecolors='black', marker='*')
        
    event_type = getattr(graph_data, 'event_type', 'Unknown')
    title = f"Graph Event: {event_type} | Next Action Target: {graph_data.y.item() if graph_data.y is not None else 'N/A'}"
    ax.set_title(title, fontsize=14, color='white', pad=20)
    
    # Make plot background dark to look professional
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for val in ax.spines.values():
        val.set_edgecolor('white')
        
    plt.show()

if 'graphs' in locals() and len(graphs) > 0:
    sample_graph = graphs[min(100, len(graphs)-1)] 
    plot_graph(sample_graph)'''),
    nbf.v4.new_markdown_cell('### Interpretation Guide\n- **Blue Nodes**: The team in possession, maintaining their 4-3-3 tactical shape around the ball.\n- **Red Nodes**: The defending team, forced backward.\n- **White Star**: The exact location of the ball according to StatsBomb.\n- **Faint Yellow Lines (Edges)**: The mathematical distance limits. This forms a "Graph." The GNN will pass information along these yellow lines to learn how open a player is.')
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/02_graph_construction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 02 created.")
