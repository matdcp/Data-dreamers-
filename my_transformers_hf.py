from dash import Dash, dcc, html, Input, Output, State, dash_table
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import confusion_matrix
import plotly.express as px
import torch
import pandas as pd
import numpy as np

# Load the fine tuned model and tokenizer from huggingface repository https://huggingface.co/lorenzop14/fine_tuned_model where i stored
# all the necessary files to compile the dashboard of the transformers, all the files have been downoloaded from TRANSFORMER_ufficiale.ipynb
MODEL_PATH = "lorenzop14/fine_tuned_model"
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Mapping of tag IDs to labels
id2tag = {
    0: "I-Multi-tissue_structure", 1: "B-Cellular_component", 2: "O",
    3: "I-Immaterial_anatomical_entity", 4: "I-Cell", 5: "B-Cell",
    6: "B-Multi-tissue_structure", 7: "I-Anatomical_system", 8: "I-Organism_substance",
    9: "B-Tissue", 10: "B-Organism_substance", 11: "B-Organ",
    12: "I-Cellular_component", 13: "I-Organ", 14: "I-Tissue",
    15: "B-Developing_anatomical_structure", 16: "I-Pathological_formation",
    17: "I-Organism_subdivision", 18: "B-Pathological_formation",
    19: "B-Anatomical_system", 20: "B-Organism_subdivision", 21: "B-Immaterial_anatomical_entity",
}

# Hardcoded confusion matrix (values from the image)
confusion_matrix_data = np.array([
    [23, 0, 2, 1, 0, 1, 9, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 1],
    [0, 9, 27, 0, 0, 2, 0, 0, 0, 0, 2, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [39, 5, 7793, 0, 4, 14, 24, 1, 1, 0, 12, 0, 4, 0, 9, 0, 9, 1, 12, 15, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 14, 0, 0, 6, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0],
    [2, 15, 14, 0, 1, 40, 4, 0, 0, 4, 4, 1, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0],
    [10, 0, 21, 0, 0, 2, 34, 0, 0, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 17, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 7, 0, 0, 13, 0, 2, 0, 3, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 3, 0, 0, 2, 2, 0, 0, 0, 26, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 6, 0, 9, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 2, 0, 0, 1, 0, 0, 10, 13, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 4, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 3, 0, 0, 1, 1, 0, 0, 1, 0, 0, 12, 0, 15, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Class labels for the confusion matrix
class_labels = ['I-Multi-tissue_structure','B-Cellular_component','O','I-Immaterial_anatomical_entity',
 'I-Cell','B-Cell','B-Multi-tissue_structure','I-Anatomical_system','I-Organism_substance',
 'B-Tissue','B-Organism_substance','B-Organ','I-Cellular_component','I-Organ','I-Tissue','B-Developing_anatomical_structure',
 'I-Pathological_formation','I-Organism_subdivision','B-Pathological_formation','B-Anatomical_system',
 'B-Organism_subdivision','B-Immaterial_anatomical_entity']

# Prediction function
def predict_ner(sentence):
    inputs = tokenizer.encode_plus(
        sentence, return_tensors="pt", truncation=True, max_length=128
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    # Decode tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    labels = [id2tag[pred] for pred in predictions]

    # Filter out special tokens
    filtered_tokens, filtered_labels = [], []
    for token, label in zip(tokens, labels):
        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            filtered_tokens.append(token)
            filtered_labels.append(label)

    return filtered_tokens, filtered_labels

# Function to highlight entities
def highlight_entities(tokens, labels):
    highlighted_text = ""
    for token, label in zip(tokens, labels):
        if label.startswith("B-") or label.startswith("I-"):
            color = "yellow" if label.startswith("B-") else "lightblue"
            highlighted_text += f'<span style="background-color: {color}; padding: 2px; margin: 2px;">{token} ({label})</span> '
        else:
            highlighted_text += f"{token} "
    return highlighted_text

# Dash app setup
app = Dash(__name__)

app.layout = html.Div([
    html.H1("NER Model Dashboard with Confusion Matrix", style={"textAlign": "center"}),

    # NER Prediction Section
    html.Div([
        dcc.Input(
            id="input-sentence",
            type="text",
            placeholder="Enter a sentence...",
            style={"width": "80%", "padding": "10px", "margin": "10px"}
        ),
        html.Button("Predict", id="predict-button", n_clicks=0, style={"padding": "10px"})
    ], style={"textAlign": "center"}),

    html.Div(id="highlighted-output", style={"margin": "20px", "fontSize": "18px"}),

    html.Div([
        html.H3("Predicted Tokens Table"),
        dash_table.DataTable(
            id="token-table",
            columns=[
                {"name": "Token", "id": "Token"},
                {"name": "Predicted Label", "id": "Predicted Label"}
            ],
            style_table={"margin": "20px"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"fontWeight": "bold"},
        )
    ]),

    html.H2("Confusion Matrix for Test Set", style={"textAlign": "center"}),
    dcc.Graph(id="confusion-matrix-plot")
])

# Callback for NER predictions
@app.callback(
    [Output("highlighted-output", "children"),
     Output("token-table", "data")],
    [Input("predict-button", "n_clicks")],
    [State("input-sentence", "value")]
)
def update_output(n_clicks, sentence):
    if not sentence:
        return "", []

    # Get predictions
    tokens, labels = predict_ner(sentence)

    # Highlight entities
    highlighted_text = highlight_entities(tokens, labels)

    # Prepare table data
    token_data = pd.DataFrame({"Token": tokens, "Predicted Label": labels})
    return highlighted_text, token_data.to_dict("records")

# Callback for displaying confusion matrix
@app.callback(
    Output("confusion-matrix-plot", "figure"),
    Input("predict-button", "n_clicks")
)
def update_confusion_matrix(n_clicks):
    # Create confusion matrix heatmap
    fig = px.imshow(
        confusion_matrix_data,
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        x=class_labels,
        y=class_labels,
        text_auto=True,  # Display exact values
        title="Confusion Matrix for Test Set"
    )

    fig.update_layout(
        width=1000,
        height=1000,
        coloraxis_colorbar=dict(title="Count")
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
