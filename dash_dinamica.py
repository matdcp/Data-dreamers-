import plotly.graph_objects as go
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from wordcloud import WordCloud
from collections import Counter
import spacy

# ======== Data Preparation ========
merged_dataset_path = "merged_dataset.csv"
ds = pd.read_csv(merged_dataset_path)
ds.dropna(inplace=True)
Ner_counts = ds['Label'].value_counts()
most_common_tokens = pd.read_csv("most_common_tokens.csv")
cleaned_ds = pd.read_csv("cleaned_dataset_NN.csv")

# Mapped labels
mapping = {
    'B-Multi-tissue_structure': 'Multi-tissue_structure',
    'I-Multi-tissue_structure': 'Multi-tissue_structure',
    'B-Organism_substance': 'Organism_substance',
    'I-Organism_substance': 'Organism_substance',
    'B-Organism_subdivision': 'Organism_subdivision',
    'I-Organism_subdivision': 'Organism_subdivision',
    'B-Organ': 'Organ',
    'I-Organ': 'Organ',
    'B-Cellular_component': 'Cellular_component',
    'I-Cellular_component': 'Cellular_component',
    'B-Cell': 'Cell',
    'I-Cell': 'Cell',
    'B-Immaterial_anatomical_entity': 'Immaterial_anatomical_entity',
    'I-Immaterial_anatomical_entity': 'Immaterial_anatomical_entity',
    'B-Tissue': 'Tissue',
    'I-Tissue': 'Tissue',
    'B-Pathological_formation': 'Pathological_formation',
    'I-Pathological_formation': 'Pathological_formation',
    'B-Anatomical_system': 'Anatomical_system',
    'I-Anatomical_system': 'Anatomical_system',
    'B-Developing_anatomical_structure': 'Developing_anatomical_structure',
    'I-Developing_anatomical_structure': 'Developing_anatomical_structure',
    'O': 'O'
}


mapped_ds = ds.copy()
mapped_ds["Label"] = mapped_ds["Label"].map(mapping)
mapped_ds_without_O = mapped_ds[mapped_ds["Label"] != "O"]

label_counts = mapped_ds["Label"].value_counts()
label_counts_without_O = mapped_ds_without_O["Label"].value_counts()

# ======== LDA Preparation ========
texts = [str(doc).split() for doc in cleaned_ds["id"].dropna()]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42)
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# ======== Sentence Length Distribution ========
sentence_ending_tokens = [".", "!", "?"]
ds['Sentence_ID'] = (ds['id'].isin(sentence_ending_tokens)).cumsum()
sentence_lengths = ds.groupby('Sentence_ID').size()


# function for ploty
def generate_sentence_length_plot_interactive():
    # Calcola l'istogramma
    bin_counts, bin_edges = np.histogram(sentence_lengths, bins=15, range=(0, 32))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # smooth line
    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
    spline = make_interp_spline(bin_centers, bin_counts, k=3)
    y_smooth = spline(x_smooth)

    # ploty figure
    fig = go.Figure()

    # hist
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=bin_counts,
        name="Frequenza",
        marker_color="skyblue"
    ))

    # shoothed curve
    fig.add_trace(go.Scatter(
        x=x_smooth,
        y=y_smooth,
        mode='lines',
        name="Curva di Distribuzione",
        line=dict(color='red', width=2)
    ))

    # Layout del grafico
    fig.update_layout(
        title="Distribuzione Interattiva della Lunghezza delle Frasi",
        xaxis_title="Lunghezza della Frase (Numero di Token)",
        yaxis_title="Frequenza",
        bargap=0.2,
        template="plotly_white"
    )

    return fig

# ======== POS Tagging Distribution ========
#  spaCy model loading
nlp = spacy.load('en_core_web_sm')
def pos_tagging(text):
    doc = nlp(text)
    return [token.pos_ for token in doc]

if 'Sentence_ID' in ds.columns:
    ds['sentence'] = ds.groupby('Sentence_ID')['id'].transform(lambda x: ' '.join(x))

unique_sentences = ds[['sentence']].drop_duplicates().dropna()
unique_sentences['POS'] = unique_sentences['sentence'].apply(pos_tagging)

pos_counts = Counter([pos for pos_list in unique_sentences['POS'] for pos in pos_list])

def generate_pos_plot():
    fig = go.Figure()

    # plot
    fig.add_trace(go.Bar(
        x=list(pos_counts.keys()),
        y=list(pos_counts.values()),
        name='Frequenza',
        marker_color='lightskyblue'
    ))


    fig.add_trace(go.Scatter(
        x=list(pos_counts.keys()),
        y=list(pos_counts.values()),
        mode='lines+markers',
        name='Curva di distribuzione',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title="Distribuzione Interattiva delle Part-of-Speech (POS)",
        xaxis_title="POS",
        yaxis_title="Frequenza",
        template="plotly_white"
    )
    return fig

# ======== Dependencies Parsing Preparation ========

def extract_dependencies(text):
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

unique_sentences['Dependencies'] = unique_sentences['sentence'].apply(extract_dependencies)

print(unique_sentences['Dependencies'].head())

dependency_counts = Counter([dep[1] for dep_list in unique_sentences['Dependencies'] for dep in dep_list])

def generate_dependencies_plot():
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(dependency_counts.keys()),
        y=list(dependency_counts.values()),
        name='Frequenza',
        marker_color='skyblue'
    ))

    fig.update_layout(
        title="Distribuzione Interattiva delle Relazioni Grammaticali",
        xaxis_title="Relazione",
        yaxis_title="Frequenza",
        xaxis=dict(tickangle=75),
        template="plotly_white",
        bargap=0.2
    )

    return fig



# ======== Cumulative Area Plot Preparation ========
ds['token_length'] = ds['id'].astype(str).apply(len)
token_length_distribution = ds.groupby(['token_length', 'Label']).size().unstack(fill_value=0)
token_length_distribution_cumsum = token_length_distribution.cumsum()
token_length_distribution_percent = (token_length_distribution_cumsum / token_length_distribution_cumsum.max()) * 100

# ======== Word Cloud ========
def generate_wordcloud_image(selected_class):
    filtered_data = most_common_tokens[most_common_tokens['Label'] == selected_class]
    word_freq = dict(zip(filtered_data['id'], filtered_data['Frequency']))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    buf = BytesIO()
    wordcloud.to_image().save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ======== Dashboard Layout ========
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Data Exploratory Analysis (DEA) - by DataDreamers"),

    html.H2("1. Labels Distribution"),
    html.H3("1.1 NER-Tag Distribution"),
    dcc.Graph(figure=px.pie(names=Ner_counts.index, values=Ner_counts.values, title="NER-Tag Distribution")),

    html.H3("1.2 Mapped NER-Tag Distribution"),
    dcc.Graph(figure=px.pie(names=label_counts.index, values=label_counts.values, title="Mapped NER-Tag Distribution")),
    dcc.Graph(figure=px.bar(x=label_counts_without_O.index, y=label_counts_without_O.values,
                            title="Mapped NER-Tag Distribution without 'O'", labels={"x": "Label", "y": "Frequency"})),

    html.H2("2. Sentence Length Distribution"),
    dcc.Graph(figure=generate_sentence_length_plot_interactive()),

    html.H2("3. POS Tagging Distribution"),
    dcc.Graph(figure=generate_pos_plot()),


    html.H2("4. Dependencies Parsing"),
    dcc.Graph(figure=generate_dependencies_plot()),

    html.H2("5. Most Common Tokens per Class"),
    dcc.Dropdown(
        id='class-dropdown',
        options=[{'label': label, 'value': label} for label in most_common_tokens['Label'].unique()],
        value=most_common_tokens['Label'].unique()[0]
    ),
    dcc.Graph(id='pie-chart'),

    html.H2("6. Cumulative Area Plot of Token Length"),
    dcc.Graph(figure=px.area(
        token_length_distribution_percent.reset_index(),
        x="token_length",
        y=token_length_distribution_percent.columns,
        title="Cumulative Percentage Distribution of Token Length by NER Category",
        labels={"value": "Cumulative Frequency (%)", "token_length": "Token Length"}
    )),

    html.H2("7. Word Cloud"),
    dcc.Dropdown(
        id='wordcloud-dropdown',
        options=[{'label': label, 'value': label} for label in most_common_tokens['Label'].unique()],
        value=most_common_tokens['Label'].unique()[0]
    ),
    html.Img(id='wordcloud-image', style={"width": "100%", "height": "auto"}),

    html.H2("8. LDA - Topic Visualization"),
    html.Iframe(
        srcDoc=pyLDAvis.prepared_data_to_html(lda_vis),
        style={"width": "100%", "height": "600px"}
    )
])

# ======== Callbacks ========
@app.callback(
    Output('pie-chart', 'figure'),
    Input('class-dropdown', 'value')
)
def update_pie_chart(selected_class):
    filtered_data = most_common_tokens[most_common_tokens['Label'] == selected_class]
    return px.pie(filtered_data, names='id', values='Frequency', title=f'Top Tokens for Class: {selected_class}')

@app.callback(
    Output('wordcloud-image', 'src'),
    Input('wordcloud-dropdown', 'value')
)
def update_wordcloud(selected_class):
    return generate_wordcloud_image(selected_class)

# ======== Run Server ========
if __name__ == '__main__':
    app.run_server(debug=True)

