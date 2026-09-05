import plotly.graph_objects as go

# Extract neuron activities and weights
neuron_activities = [sim.data[ensemble] for ensemble in ensembles]
weights = [sim.data[conn].weights for conn in model.connections]

# Calculate average activity of each neuron
average_activities = [np.mean(activity, axis=0) for activity in neuron_activities]

# Create 3D scatter plot
fig = go.Figure()

# Add scatter points for each neuron
for i in range(len(ensembles)):
    activity = average_activities[i]
    x, y, z = np.random.randn(3)
    fig.add_trace(go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode='markers',
        marker=dict(
            size=10,
            color=activity,  # Use average activity for color
            colorscale='Viridis',
            opacity=0.8
        ),
        name=f'Neuron {i+1}'
    ))

# Add connections between neurons
for i in range(len(ensembles)):
    for j in range(len(ensembles)):
        if i != j:
            weight = weights[i][j]
            x0, y0, z0 = np.random.randn(3)
            x1, y1, z1 = np.random.randn(3)
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(
                    color='rgb(125,125,125)',
                    width=1,
                ),
                opacity=weight,  # Use weight for opacity
                showlegend=False
            ))

# Set layout
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    title='3D Scatter Plot of Neuron Network',
)

# Show plot
fig.show()
