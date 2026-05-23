# Label Studio — Data Labeling & Annotation Platform

Platform for labeling text, image, and audio data with a web interface and REST API.

## Installation

```bash
pip install label-studio
label-studio start  # http://localhost:8080
```

## Programmatic Project Setup

```python
import label_studio_sdk

ls = label_studio_sdk.Client("http://localhost:8080", "YOUR_API_KEY")
project = ls.create_project(
    title="NER Annotation",
    description="Annotate named entities",
    label_config="""
    <View>
      <Labels name="label" toName="text">
        <Label value="PERSON" background="red"/>
        <Label value="ORG" background="blue"/>
        <Label value="LOC" background="green"/>
      </Labels>
      <Text name="text" value="$text"/>
    </View>
    """,
)
```

## Labeling Configurations

```xml
<!-- Text classification -->
<View>
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="Positive"/><Choice value="Negative"/>
  </Choices>
  <Text name="text" value="$text"/>
</View>
```

```xml
<!-- Image bounding boxes -->
<View>
  <Image name="image" value="$image_url"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Car"/><Label value="Pedestrian"/>
  </RectangleLabels>
</View>
```

## API-Based Labeling

```python
project.import_tasks([
    {"text": "Apple acquired Beats in 2014."},
    {"text": "Google is headquartered in Mountain View."},
])

# Export annotations
export_result = project.export_tasks(export_type="JSON")
print(export_result)
```

## ML Backend

```python
from label_studio_ml.model import LabelStudioMLBase

class MyModel(LabelStudioMLBase):
    def predict(self, tasks, context=None, **kwargs):
        predictions = []
        for task in tasks:
            text = task["data"]["text"]
            entities = self.run_ner(text)
            predictions.append({"result": entities, "score": 0.95})
        return predictions

    def fit(self, annotations, **kwargs):
        pass  # Training loop
```

## Export Formats

```python
project.export_tasks(export_type="JSON")
project.export_tasks(export_type="COCO")       # Images
project.export_tasks(export_type="CONLL2003")  # NER
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Project | Container for tasks + config |
| Label Config (XML) | Annotation schema |
| Task | Data item to annotate |
| Annotation | Completed labeling result |
| ML Backend | Model server for predictions |

## Integration

Use the SDK to manage projects. Export in model-expected formats. ML backend enables active learning loops with pre-labeling.
