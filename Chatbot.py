import random
import difflib
import re

# 🏦 Knowledge base with expanded operations
knowledge_base = {
    "savings account": {
        "available": True,
        "interest_rate": 2.5,
        "minimum_balance": 1000,
        "description": "A flexible account that helps you grow your savings with annual interest and easy access to funds.",
    },
    "checking account": {
        "available": True,
        "interest_rate": 0.5,
        "minimum_balance": 500,
        "description": "A simple account designed for everyday transactions with low fees and instant transfers.",
    },
    "credit card": {
        "available": False,
        "interest_rate": 15.0,
        "minimum_balance": 0,
        "description": "A convenient card for payments and purchases, currently unavailable for new applications.",
    },
    "loan": {
        "available": True,
        "interest_rate": 6.5,
        "minimum_balance": 0,
        "description": "Flexible personal loans with competitive rates and fast approval.",
    },
}

# --- Bank service operations ---
operations = {
    "check balance": "To check your balance, log in to your online banking or visit a nearby branch with your ID. For privacy, I can’t display your balance here.",
    "close account": "To close your account, you’ll need to visit a branch with valid identification. Ensure your balance is cleared and any active loans are settled.",
    "open account": "To open an account, bring your Emirates ID or passport and proof of address to any branch, or apply online through our website.",
    "apply for loan": "Loan applications can be submitted online or in-branch. You’ll need valid ID, salary proof, and 3 months of bank statements.",
}

# Greetings and responses
greetings = [
    "Welcome! What would you like to know about our services?",
    "Hello! How can I assist you with our banking services today?",
]

unknown_query_responses = [
    "I didn’t quite get that. Could you please clarify or repeat what you meant?",
    "Sorry, I didn’t understand that. Could you try rephrasing?",
]

# --- Context tracking ---
context = {
    "active_service": None,
    "service_history": {},
    "awaiting_service": False,
}

# --- Common phrases ---
exit_phrases = ["exit", "quit", "bye", "goodbye", "done", "finished", "thank you", "thanks"]
availability_keywords = ["available", "open", "apply", "offer", "signup", "sign up", "start", "join"]
interest_keywords = ["interest", "rate", "percentage", "earnings", "return", "profit", "yield"]
minimum_keywords = ["minimum", "balance", "requirement", "deposit", "need to keep"]
description_keywords = ["describe", "about", "information", "details", "tell me", "what is", "overview"]
balance_keywords = ["balance", "check balance", "how much do i have", "my balance"]
closing_keywords = ["close account", "close my account", "end account", "delete account"]
open_keywords = ["open account", "create account", "start account", "register account"]
loan_keywords = ["loan application", "apply loan", "get a loan", "apply for loan"]

# Generic service phrases
generic_queries = [
    "account", "accounts", "service", "services", "loan", "loans",
    "i want an account", "i want to open an account", "i would like an account", "open account"
]

# --- Helper: fuzzy service detection ---
def find_service(query):
    query = query.lower()
    for service in knowledge_base:
        if service in query:
            return service
    best_match, best_ratio = None, 0
    for service in knowledge_base:
        ratio = difflib.SequenceMatcher(None, query, service).ratio()
        if ratio > best_ratio:
            best_match, best_ratio = service, ratio
    if best_ratio >= 0.65 and any(word in query for word in ["savings", "checking", "loan", "credit"]):
        return best_match
    return None

# --- Helper: with polite follow-up ---
def with_hint(response):
    hints = [
        "Would you like to ask about another service?",
        "Is there anything else you'd like to know?",
        "You can also ask about account opening, checking balance, or closing an account.",
        "Would you like details about a different service?",
    ]
    return f"{response} {random.choice(hints)}"

# --- Helper: context logging ---
def log_state(message):
    print(f"(Context: {message})")

# --- Main chatbot logic ---
def chatbot(query):
    query = query.lower().strip()

    # Exit check
    if any(phrase in query for phrase in exit_phrases):
        return "exit"

    # Greetings
    if query in ["hi", "hello", "hey"]:
        return random.choice(greetings)

    # Detect service
    service = find_service(query)

    # Handle specific operations first
    if any(phrase in query for phrase in balance_keywords):
        log_state("user requested balance check")
        return with_hint(operations["check balance"])

    if any(phrase in query for phrase in closing_keywords):
        log_state("user requested account closure")
        return with_hint(operations["close account"])

    if any(phrase in query for phrase in open_keywords):
        log_state("user requested account opening")
        return with_hint(operations["open account"])

    if any(phrase in query for phrase in loan_keywords):
        log_state("user requested loan application")
        return with_hint(operations["apply for loan"])

    # Handle generic queries like "open an account" or "services"
    if any(phrase in query for phrase in generic_queries) and not service:
        available = [s for s, info in knowledge_base.items() if info["available"]]
        context["awaiting_service"] = True
        log_state("awaiting service selection")
        return with_hint(
            f"We currently offer the following services: {', '.join(available)}. "
            "Please choose one to know more about (like interest, minimum balance, or availability)."
        )

    # Handle service selection
    if context["awaiting_service"]:
        if service and service in knowledge_base:
            context["active_service"] = service
            context["service_history"][service] = "selected"
            context["awaiting_service"] = False
            log_state(f"switching context → {service}")
            return with_hint(
                f"You selected {service}. You can ask about interest rates, minimum balance, availability, or how to apply."
            )
        else:
            return "I didn’t quite catch that. Could you name the service you’re interested in? (e.g., savings account, loan)"

    # Switch service mid-conversation
    if service:
        previous = context.get("active_service")
        if previous != service:
            log_state(f"switching context → {service}")
        context["active_service"] = service
        context["service_history"][service] = "selected"
        return with_hint(
            f"You're now asking about {service}. You can ask about interest rates, minimum balance, availability, or how to apply."
        )

    # Respond within an active context
    active = context.get("active_service")
    if active:
        info = knowledge_base[active]

        if any(word in query for word in availability_keywords):
            log_state(f"answering availability for {active}")
            return with_hint(
                f"Yes, our {active} is currently available. You can apply online or at any branch. "
                "Would you like a list of documents needed?"
            )

        elif any(word in query for word in interest_keywords):
            log_state(f"answering interest for {active}")
            return with_hint(f"The current interest rate for our {active} is {info['interest_rate']}%.")

        elif any(word in query for word in minimum_keywords):
            log_state(f"answering minimum balance for {active}")
            return with_hint(f"You’ll need to maintain a minimum balance of ${info['minimum_balance']} for the {active}.")

        elif any(word in query for word in description_keywords):
            log_state(f"describing {active}")
            return with_hint(info["description"])

        else:
            log_state(f"unclear input under {active}")
            return with_hint(
                f"You're asking about {active}. You can ask about interest, minimum balance, availability, or how to apply."
            )

    log_state("no recognized service or context")
    return random.choice(unknown_query_responses)

# --- Run chatbot ---
print(random.choice(greetings))
while True:
    user = input("You: ")
    bot_reply = chatbot(user)
    if bot_reply == "exit":
        print("Chatbot: Goodbye! Thank you for banking with us.")
        break
    print(f"Chatbot: {bot_reply}")
