import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import plotly.graph_objs as go

# Updated Metrics Table
data = {
    'Metric': ['Accuracy', 'F1 Score Macro', 'Recall Macro', 'Precision Macro'],
    'Linear SVC': [0.7726, 0.4648, 0.3706, 0.6886],
    'LSTM': [0.6753, 0.2304, 0.5235, 0.182],
    'BiLSTM': [0.8482, 0.3364, 0.6669, 0.2498],
    'GRU': [0.8494, 0.306, 0.6508, 0.2262]
}
evaluation_tab = pd.DataFrame(data).set_index('Metric')

# Precomputed Confusion Matrices (placeholders)
confusion_matrices = {

    'LSTM': np.array([
        [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [2, 58, 1, 2, 0, 1, 3, 0, 0, 3, 0, 0, 21, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2],
        [0, 3, 12, 0, 2, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 2, 0, 8, 26, 2, 4, 1, 3, 2, 0, 1, 2, 0, 5, 29, 1, 4, 0, 2, 1, 3],
        [1, 0, 0, 1, 3, 3, 31, 1, 1, 2, 1, 0, 0, 0, 0, 0, 5, 5, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 2, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 0, 0, 20, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 5, 0, 2, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 24, 1, 0, 2, 0, 0, 2, 0, 0, 0, 3, 8, 0, 3],
        [0, 0, 1, 0, 1, 2, 1, 1 ,1, 2, 6, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 7, 2],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0],
        [3, 0, 0, 0, 3, 4, 0, 1, 0, 0, 0, 0, 1, 1, 0, 4, 27, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 13, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 13, 0],
        [333, 324, 638, 12, 202, 91, 37, 70, 301, 257, 25, 147, 55, 20, 9, 48, 29, 5, 21, 37,11,18,5867],
    ]),
    'BiLSTM': np.array([
    [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 75, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0, 13, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 2, 15, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 3],
    [0, 0, 0, 10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 2, 0, 2, 56, 2, 1, 1, 4, 4, 0, 1, 0, 0, 0, 14, 0, 1, 0 ,1, 2, 3],
    [0, 0, 0, 0, 0, 3, 47, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2],
    [0, 0, 2, 0, 0, 1, 0, 0, 0, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 5],
    [0, 0, 1, 0, 0, 1, 4, 0, 0, 1, 14, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 1, 0, 0, 0 ,1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 27, 0, 2, 1, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 13, 0],
    [78, 131, 110, 8, 48, 104, 48, 19, 107, 195, 55, 23, 31, 35, 4, 8, 56, 9, 45, 36, 36, 19, 7352],

]),

    'GRU': np.array([
    [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 73, 1, 1, 0, 2, 3, 0, 0, 3, 1, 1, 5, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2],
    [0, 3, 13, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 3, 0, 4, 51, 4, 3, 2, 2, 3, 0, 1, 0, 0, 0, 13, 0, 1, 0, 1, 0, 6],
    [0, 0, 0, 1, 0, 2, 44, 1, 0, 2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 11, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, 1, 0, 0, 25, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1],
    [1, 2, 0, 0, 0, 1, 0, 1, 0, 28, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 8, 0, 4],
    [0, 0, 2, 0, 0, 2, 5, 1, 0, 1, 10, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 2, 1, 1, 1, 0, 2, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [5, 0, 0, 0, 0, 6, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 28, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 16, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 9, 0],
    [104, 130, 105, 18, 31, 66, 27, 56, 69, 145, 28, 32, 57, 41, 6, 10, 56, 14, 25, 46, 73, 32, 7386],
])

}

class_labels = ['B-Anatomical_system','B-Cell','B-Cellular_component','B-Developing_anatomical_structure',
 'B-Immaterial_anatomical_entity','B-Multi-tissue_structure', 'B-Organ','B-Organism_subdivision',
 'B-Organism_substance','B-Pathological_formation','B-Tissue','I-Anatomical_system','I-Cell',
 'I-Cellular_component','I-Developing_anatomical_structure','I-Immaterial_anatomical_entity',
 'I-Multi-tissue_structure', 'I-Organ','I-Organism_subdivision', 'I-Organism_substance',
 'I-Pathological_formation', 'I-Tissue','O']

with open('history_lstm.pkl', 'rb') as f:
    history_lstm = pickle.load(f)

with open('history_bilstm.pkl', 'rb') as f:
    history_bilstm = pickle.load(f)

with open('history_gru.pkl', 'rb') as f:
    history_gru = pickle.load(f)

# Store all histories in a dictionary for easier access
histories = {
    'LSTM': history_lstm,
    'BiLSTM': history_bilstm,
    'GRU': history_gru
}
# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Model Performance Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Model Performance Dashboard", style={'textAlign': 'center'}),

    # Dropdown to select metric
    html.Div([
        html.Label("Select Metric:", style={'fontSize': '18px'}),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[{'label': metric, 'value': metric} for metric in evaluation_tab.index],
            value='Accuracy'
        )
    ], style={'width': '50%', 'margin': '20px auto'}),

    # Bar chart for metrics
    html.Div([
        html.H3("Model Comparison Metrics", style={'textAlign': 'center'}),
        dcc.Graph(id='metric-bar-chart')
    ]),

    # Dropdown to select model for confusion matrix
    html.Div([
        html.Label("Select Model for Confusion Matrix:", style={'fontSize': '18px'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': model, 'value': model} for model in confusion_matrices.keys()],
            value='Linear SVC'
        )
    ], style={'width': '50%', 'margin': '20px auto'}),

    # Confusion matrix visualization
    html.Div([
        html.H3("Confusion Matrix", style={'textAlign': 'center'}),
        dcc.Graph(id='confusion-matrix-plot')
    ]),

    # Dropdown to select model for Loss and Accuracy
    html.Div([
        html.Label("Select Model for Training and Validation Metrics:", style={'fontSize': '18px'}),
        dcc.Dropdown(
            id='history-model-dropdown',
            options=[
                {'label': 'LSTM', 'value': 'LSTM'},
                {'label': 'BiLSTM', 'value': 'BiLSTM'},
                {'label': 'GRU', 'value': 'GRU'}
            ],
            value='LSTM'  # Default value
        )
    ], style={'width': '50%', 'margin': '20px auto'}),

    # Loss Graph
    html.Div([
        html.H3("Training and Validation Loss", style={'textAlign': 'center'}),
        dcc.Graph(id='loss-graph')
    ]),

    # Accuracy Graph
    html.Div([
        html.H3("Training and Validation Accuracy", style={'textAlign': 'center'}),
        dcc.Graph(id='accuracy-graph')
    ]),
])

# Callback for Metrics Bar Chart
@app.callback(
    Output('metric-bar-chart', 'figure'),
    Input('metric-dropdown', 'value')
)
def update_metrics_chart(selected_metric):
    metrics_data = evaluation_tab.loc[selected_metric]
    fig = px.bar(
        x=metrics_data.index,
        y=metrics_data.values,
        color=metrics_data.index,
        labels={'x': 'Models', 'y': selected_metric},
        title=f"Comparison of {selected_metric} Across Models"
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1]),  # Metrics are between 0 and 1
        showlegend=False,
        template='plotly_white'
    )
    return fig

# Callback for Confusion Matrix
@app.callback(
    Output('confusion-matrix-plot', 'figure'),
    Input('model-dropdown', 'value')
)
def update_confusion_matrix(selected_model):
    matrix = confusion_matrices[selected_model]
    fig = px.imshow(
        matrix,
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        x=class_labels,
        y=class_labels,
        text_auto=True,
        title=f"Confusion Matrix for {selected_model}"
    )
    fig.update_layout(
        autosize=False,
        width=1400,  # Increased size
        height=1400,
        coloraxis_colorbar=dict(title="Count")
    )
    return fig

# Callback for Loss and Accuracy Graphs
@app.callback(
    [Output('loss-graph', 'figure'),
     Output('accuracy-graph', 'figure')],
    Input('history-model-dropdown', 'value')
)
def update_history_graphs(selected_model):
    history = histories[selected_model]
    print(f"Lengths - Loss: {len(history['loss'])}, Val Loss: {len(history['val_loss'])}, "
          f"Accuracy: {len(history['accuracy'])}, Val Accuracy: {len(history['val_accuracy'])}")
    epochs = list(range(1, len(history['loss']) + 1))

    # Generate loss figure
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=history['loss'], name='Training Loss', line=dict(color='#FF5733')))
    loss_fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss', line=dict(color='#C70039')))
    loss_fig.update_layout(title=f"Training and Validation Loss ({selected_model})", xaxis_title="Epochs",
                           yaxis_title="Loss")

    # Generate accuracy figure
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], name='Training Accuracy', line=dict(color='#33FFBD')))
    acc_fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='#2E86C1')))
    acc_fig.update_layout(title=f"Training and Validation Accuracy ({selected_model})", xaxis_title="Epochs",
                          yaxis_title="Accuracy")

    # Debug prints for figures
    print("Loss Figure:", loss_fig)
    print("Accuracy Figure:", acc_fig)

    return loss_fig, acc_fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
