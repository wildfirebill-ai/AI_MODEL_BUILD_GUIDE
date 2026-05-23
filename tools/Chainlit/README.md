# Chainlit — Production-Ready Conversational AI Apps

[Chainlit](https://docs.chainlit.io) is an open-source Python framework for building conversational AI interfaces. It handles streaming, history, auth, and deployment out of the box.

## Installation

```bash
pip install chainlit
```

Verify:

```bash
chainlit --version
```

## Quick Start — Echo Bot

```python
# app.py
import chainlit as cl

@cl.on_message
async def on_message(msg: cl.Message):
    await cl.Message(content=f"Echo: {msg.content}").send()
```

Run:

```bash
chainlit run app.py -w
```

## Chat Start Hook

```python
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome! Ask me anything.").send()
    cl.user_session.set("counter", 0)

@cl.on_message
async def on_message(msg: cl.Message):
    counter = cl.user_session.get("counter") + 1
    cl.user_session.set("counter", counter)
    await cl.Message(content=f"Message #{counter}: {msg.content}").send()
```

## Asking the User for Input

```python
import chainlit as cl

@cl.on_message
async def on_message(msg: cl.Message):
    res = await cl.AskUserMessage(content="What is your favorite color?", timeout=30).send()
    if res:
        await cl.Message(content=f"Your favorite color is {res['output']}!").send()
```

## Streaming LLM Responses (OpenAI)

```python
import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI()

@cl.on_message
async def on_message(msg: cl.Message):
    msg = cl.Message(content="")
    async for part in await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": msg.content}],
        stream=True,
    ):
        if part.choices[0].delta.content:
            await msg.stream_token(part.choices[0].delta.content)
    await msg.send()
```

## Using Steps for Intermediate Results

```python
import chainlit as cl

@cl.on_message
async def on_message(msg: cl.Message):
    async with cl.Step(name="Thinking"):
        await cl.sleep(1)
        intermediate = "Processing complete."

    async with cl.Step(name="Generating"):
        final = f"Result: {msg.content}"

    await cl.Message(content=final).send()
```

## Authentication

```python
# app.py
import chainlit as cl

@cl.password_auth_callback
def auth_callback(email: str, password: str):
    if email == "admin@example.com" and password == "secret":
        return cl.User(identifier=email, metadata={"role": "admin"})
    return None
```

Configure in `chainlit.md` or CLI flags.

## Data Persistence with SQLAlchemy

```python
import chainlit as cl
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()
engine = create_engine("sqlite:///conversations.db")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)
    content = Column(Text)

Base.metadata.create_all(engine)

@cl.on_message
async def on_message(msg: cl.Message):
    with Session(engine) as session:
        session.add(Conversation(id=msg.id, content=msg.content))
        session.commit()
    await cl.Message(content="Saved!").send()
```

## RAG Demo Skeleton

```python
import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI()

@cl.on_chat_start
async def start():
    cl.user_session.set("context", "You are a helpful assistant.")

@cl.on_message
async def on_message(msg: cl.Message):
    context = cl.user_session.get("context")
    async with cl.Step(name="Searching documents"):
        await cl.sleep(0.5)
        retrieved = "Relevant doc snippet..."

    async with cl.Step(name="Generating answer"):
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Context: {retrieved}"},
                {"role": "user", "content": msg.content},
            ],
        )
    await cl.Message(content=response.choices[0].message.content).send()
```

## Deployment

Deploy with Docker or on [Chainlit Cloud](https://cloud.chainlit.io). Set `CHAINLIT_AUTH_SECRET` in production.
