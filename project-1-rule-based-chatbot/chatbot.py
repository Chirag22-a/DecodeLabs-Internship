"""A small rule-based chatbot for DecodeLabs AI Internship Project 1."""

import string


# Each group below is one intent. Multiple phrases can trigger the same reply.
RESPONSES = {
    # Greeting
    "hello": "Hi there! How can I help you today?",
    "hi": "Hi there! How can I help you today?",
    "hey": "Hi there! How can I help you today?",

    # Bot identity
    "what is your name": "I'm LogicBot, a simple rule-based chatbot.",
    "who are you": "I'm LogicBot, a simple rule-based chatbot.",

    # Capabilities
    "help": "I can answer a few questions about myself, AI, and this project.",
    "what can you do": "I can answer a few questions about myself, AI, and this project.",

    # Wellbeing
    "how are you": "I'm working well. Thanks for asking!",
    "how are you doing": "I'm working well. Thanks for asking!",

    # Artificial intelligence
    "what is ai": "AI is the field of building computer systems that perform tasks associated with human intelligence.",
    "what is artificial intelligence": "AI is the field of building computer systems that perform tasks associated with human intelligence.",

    # Machine learning
    "what is machine learning": "Machine learning is a way for computers to learn patterns from data.",

    # Project information
    "what is this project": "This project is a rule-based chatbot built with Python control flow and a response dictionary.",
    "what are you built with": "I'm built with Python rules, a dictionary, and a continuous input loop.",

    # Gratitude
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",

    # A little personality
    "tell me a joke": "Why did the computer go to the doctor? It had a virus!",
}

EXIT_COMMANDS = {"bye", "goodbye", "exit", "quit"}
FALLBACK_MESSAGE = "I'm not sure how to answer that yet. Try 'help' or ask me about AI."


def normalize_input(message: str) -> str:
    """Normalize case, surrounding whitespace, punctuation, and repeated spaces."""
    cleaned_message = message.lower().strip()
    cleaned_message = cleaned_message.strip(string.punctuation + string.whitespace)
    return " ".join(cleaned_message.split())


def main() -> None:
    """Run the chatbot until the user enters an exit command."""
    print("LogicBot: Hello! I'm LogicBot. Type 'help' to see what I can answer.")
    print("LogicBot: Type 'bye', 'goodbye', 'exit', or 'quit' to end the chat.")

    while True:
        user_message = input("You: ")
        clean_message = normalize_input(user_message)

        if clean_message in EXIT_COMMANDS:
            print("LogicBot: Goodbye! It was nice chatting with you.")
            break

        if not clean_message:
            print("LogicBot: Please type a message so I can respond.")
        else:
            response = RESPONSES.get(clean_message)
            if response is None:
                print(f"LogicBot: {FALLBACK_MESSAGE}")
            else:
                print(f"LogicBot: {response}")


if __name__ == "__main__":
    main()
