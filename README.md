# DecodeLabs AI Internship - Project 1

## Rule-Based AI Chatbot

A small command-line chatbot built with Python. It uses explicit rules and a
dictionary of responses; it does not call an online AI service.

## Requirements

- Python 3.10 or newer
- No third-party packages

## Run it

Open a terminal in this folder and run:

```powershell
py chatbot.py
```

If the `py` command is unavailable, use:

```powershell
python chatbot.py
```

## Features

- A continuous `while` loop for conversation
- Case, surrounding whitespace, punctuation, and repeated-space normalization
- Eight response intents: greetings, identity, capabilities, wellbeing, AI,
  machine learning, project information, gratitude, plus a joke response
- A fallback for messages the bot does not recognize
- Clean exit with `bye`, `goodbye`, `exit`, or `quit`
- An explicit `if`/`else` response path and dictionary `.get()` lookup

## Example

```text
LogicBot: Hello! I'm LogicBot. Type 'help' to see what I can answer.
LogicBot: Type 'bye', 'goodbye', 'exit', or 'quit' to end the chat.
You:   HELLO!!
LogicBot: Hi there! How can I help you today?
You: What is AI?
LogicBot: AI is the field of building computer systems that perform tasks associated with human intelligence.
You: exit
LogicBot: Goodbye! It was nice chatting with you.
```

Try an unrecognized question too; the bot will show its fallback response and
continue listening.

## Project checklist

- [x] Continuous input loop and clean exit
- [x] Input normalization and more than five response intents
- [x] Dictionary lookup and fallback response
- [ ] Create the empty public GitHub repository `DecodeLabs-Internship` when
      signed in to the GitHub account; keep this project local until mentors
      provide upload instructions.
