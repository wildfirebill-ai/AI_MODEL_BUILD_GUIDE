# Rasa — Open-Source Conversational AI Framework

Build contextual chatbots with NLU, dialogue management, and custom actions.

## Installation
```bash
pip install rasa && rasa init --no-prompt && rasa train && rasa shell
```

## NLU Training Data
```yml:data/nlu.yml
version: "3.1"
nlu:
  - intent: greet
    examples: |-
      - hey
      - hello
  - intent: ask_weather
    examples: |-
      - what's the weather in [Berlin](city)
      - is it raining in [London](city)
```

## Pipeline & Policies
```yml:config.yml
pipeline:
  - name: WhitespaceTokenizer
  - name: CountVectorsFeaturizer
  - name: DIETClassifier
    epochs: 100
  - name: FallbackClassifier
    threshold: 0.3
policies:
  - name: MemoizationPolicy
  - name: RulePolicy
  - name: TEDPolicy
    epochs: 100
```

## Stories & Rules
```yml:data/stories.yml
version: "3.1"
stories:
  - story: happy path
    steps:
      - intent: greet
      - action: utter_greet
      - intent: ask_weather
      - action: action_check_weather
```
```yml:data/rules.yml
version: "3.1"
rules:
  - rule: say goodbye
    steps:
      - intent: goodbye
      - action: utter_goodbye
```

## Domain
```yml:domain.yml
version: "3.1"
intents: [greet, ask_weather, goodbye]
entities: [city]
slots:
  city:
    type: text
    mappings:
      - type: from_entity
        entity: city
responses:
  utter_greet: [{text: "Hey! How are you?"}]
  utter_goodbye: [{text: "Goodbye!"}]
actions: [action_check_weather]
```

## Custom Actions
```python:actions/actions.py
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionCheckWeather(Action):
    def name(self) -> str:
        return "action_check_weather"

    def run(self, dispatcher, tracker: Tracker, domain: dict) -> list:
        city = next(tracker.get_latest_entity_values("city"), None)
        if not city:
            dispatcher.utter_message(text="Which city?")
            return []
        weather = {"Berlin": "15°C, cloudy", "London": "12°C, rainy"}
        dispatcher.utter_message(text=f"Weather in {city}: {weather.get(city, 'unknown')}.")
        return []
```
## Key Concepts
| Concept | Description |
|---------|-------------|
| NLU Pipeline | Intent + entity extraction |
| Stories | Dialogue flow examples |
| Rules | Fixed dialogue paths |
| Custom Actions | Python dynamic responses |
| Slots | Conversation context |
| DIETClassifier | Intent + entity transformer |

## Integration
Serve via `rasa run` (HTTP API). Custom actions in separate SDK server (`rasa run actions`). Use custom actions for LLM integration.
