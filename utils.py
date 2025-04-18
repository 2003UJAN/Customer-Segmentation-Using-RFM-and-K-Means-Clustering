import plotly.express as px

def plot_segments(components, labels):
    fig = px.scatter(
        x=components[:, 0],
        y=components[:, 1],
        color=labels.astype(str),
        title="Customer Segments (PCA View)",
        labels={'x': 'PCA 1', 'y': 'PCA 2'}
    )
    return fig
