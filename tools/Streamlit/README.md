# Streamlit — Fastest ML Demo Apps in Pure Python

[Streamlit](https://streamlit.io) turns Python scripts into interactive web apps with no front-end experience required. Ideal for model demos, internal tools, and data dashboards.

## Installation

```bash
pip install streamlit
```

Verify:

```bash
streamlit --version
```

## Quick Start — Hello World

```python
# app.py
import streamlit as st

st.title("Hello from Streamlit")
st.write("This is the fastest way to build an ML demo.")

name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}!")
```

Run:

```bash
streamlit run app.py
```

## Loading & Displaying Data

```python
import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(20, 4), columns=["A", "B", "C", "D"])
st.dataframe(df, use_container_width=True)
st.line_chart(df)
```

## Widgets & Interactivity

```python
import streamlit as st
import plotly.express as px
import pandas as pd

df = px.data.iris()

species = st.selectbox("Filter by species", df["species"].unique())
threshold = st.slider("Sepal length threshold", 4.0, 8.0, 5.0)

filtered = df[(df["species"] == species) & (df["sepal_length"] > threshold)]
fig = px.scatter(filtered, x="sepal_width", y="petal_length", color="species")
st.plotly_chart(fig, use_container_width=True)
```

## Caching — `@st.cache_data`

Avoid recomputing expensive operations:

```python
import streamlit as st
import pandas as pd

@st.cache_data
def load_large_dataset():
    # Simulate expensive load
    return pd.read_parquet("large_file.parquet")

df = load_large_dataset()
st.dataframe(df.head())
```

## Session State — Persisting Across Reruns

```python
import streamlit as st

if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1

st.write(f"Count = {st.session_state.count}")
```

## LLM Chat Interface — `st.chat_message`

```python
import streamlit as st
import random

st.title("Simple Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    response = f"You said: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
```

## ML Model Demo (scikit-learn)

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

st.title("Random Forest Classifier Demo")

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
model = RandomForestClassifier()
model.fit(X, y)

sepal_length = st.slider("Feature 1", -3.0, 3.0, 0.0)
sepal_width = st.slider("Feature 2", -3.0, 3.0, 0.0)
petal_length = st.slider("Feature 3", -3.0, 3.0, 0.0)
petal_width = st.slider("Feature 4", -3.0, 3.0, 0.0)

input_vec = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred = model.predict(input_vec)[0]
prob = model.predict_proba(input_vec)[0]

st.write(f"Prediction: **Class {pred}**")
st.write(f"Confidence: {prob[pred]:.2%}")
```

## Deployment

Push to GitHub and deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (free) or HuggingFace Spaces.
