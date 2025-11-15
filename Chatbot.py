import random
import difflib
import re

# ------------------------------------------------------
# KNOWLEDGE BASE
# ------------------------------------------------------
knowledge_base = {
    "savings account": {
        "available": True,
        "interest_rate": 2.5,
        "minimum_balance": 1000,
        "requirements": "Emirates ID and proof of address.",
        "apply": "You can apply online or at any branch.",
        "description": "A flexible savings account ideal for long-term saving."
    },
    "checking account": {
        "available": True,
        "interest_rate": 0.5,
        "minimum_balance": 500,
        "requirements": "Emirates ID and address verification.",
        "apply": "Visit a branch or apply online.",
        "description": "A daily-use account with low fees."
    },
    "student account": {
        "available": True,
        "interest_rate": 0.2,
        "minimum_balance": 0,
        "requirements": "Valid University ID and Emirates ID.",
        "apply": "Apply in branch with student documents.",
        "description": "A no-fee account designed for students."
    },
    "business account": {
        "available": True,
        "interest_rate": 1.0,
        "minimum_balance": 5000,
        "requirements": "Trade license, Emirates ID, and business documents.",
        "apply": "Apply through business banking services.",
        "description": "A corporate account for business operations."
    },
    "personal loan": {
        "available": True,
        "interest_rate": 6.5,
        "minimum_balance": 0,
        "requirements": "Salary certificate and bank statements.",
        "apply": "Apply online or at a branch.",
        "description": "A flexible loan for personal needs."
    }
}


# ------------------------------------------------------
# HELP / GUIDE MENU
# ------------------------------------------------------
def help_message():
    return (
        "Here’s what I can help you with:\n\n"
        "**Accounts & Services:**\n"
        "- Savings Account\n"
        "- Checking Account\n"
        "- Student Account\n"
        "- Business Account\n"
        "- Personal Loan\n\n"
        "**You can ask about:**\n"
        "- Interest rate\n"
        "- Minimum balance\n"
        "- Requirements\n"
        "- Availability\n"
        "- How to apply\n"
        "- General description\n\n"
        "**Try examples like:**\n"
        "- 'I want to open a savings account'\n"
        "- 'Interest rate for student account'\n"
        "- 'What are the requirements for a business account?'\n"
        "- 'How do I apply for a personal loan?'"
    )


# ------------------------------------------------------
# SPAM DETECTION
# ------------------------------------------------------
def looks_like_spam(text):
    if len(text) >= 8 and re.fullmatch(r"[a-zA-Z]+", text):
        vowels = sum(c in "aeiou" for c in text)
        if vowels == 0 or vowels / len(text) < 0.15:
            return True
    if re.search(r"(.)\1{2,}", text):
        return True
    if re.fullmatch(r"[0-9a-zA-Z!@#$%^&*()_+=\-{}\[\]:;\"'<>,.?/\\|`~]{10,}", text):
        return True
    return False


# ------------------------------------------------------
# ULTRA-ACCURATE MATCHING ENGINE
# ------------------------------------------------------
def match_service(text):
    text = text.lower().strip()
    services = list(knowledge_base.keys())

    # Normalize input
    text_clean = re.sub(r"\s+", " ", text).strip()

    # -----------------------------------------
    # 1) STRONG PARTIAL WORD MATCH
    # -----------------------------------------
    for service in services:
        service_words = service.split()
        input_words = text_clean.split()

        strong_hits = sum(
            1 for w in input_words
            if len(w) >= 3 and any(w in sw for sw in service_words)
        )

        if strong_hits >= 1:
            return service, None  # direct match

    # -----------------------------------------
    # 2) HIGH-CONFIDENCE FUZZY SEARCH (>=0.70)
    # -----------------------------------------
    best_match = None
    best_score = 0.0

    for service in services:
        score = difflib.SequenceMatcher(None, text_clean, service).ratio()

        # Check word-level
        for w1 in text_clean.split():
            for w2 in service.split():
                score = max(score, difflib.SequenceMatcher(None, w1, w2).ratio())

        if score > best_score:
            best_score = score
            best_match = service

    # Only suggest if VERY accurate
    if best_score >= 0.70:
        return None, best_match  # “Did you mean?”

    # -----------------------------------------
    # 3) PREFIX MATCH (only if unique)
    # -----------------------------------------
    prefix_matches = [s for s in services if s.startswith(text_clean[:3])]
    if len(prefix_matches) == 1:
        return prefix_matches[0], None

    return None, None


# ------------------------------------------------------
# INTENT DETECTION
# ------------------------------------------------------
def detect_intent(text):
    text = text.lower()

    intents = {
        "interest": ["interest", "rate", "percentage"],
        "minimum": ["minimum", "min balance", "minimum balance"],
        "availability": ["available", "availability"],
        "requirements": ["requirements", "documents", "docs", "needed"],
        "apply": ["apply", "open", "open account", "application"],
        "description": ["describe", "explain", "details", "information"],
    }

    for intent, keywords in intents.items():
        for k in keywords:
            if k in text:
                return intent

    return None


# ------------------------------------------------------
# CONTEXT MEMORY
# ------------------------------------------------------
context = {
    "active_service": None,
    "awaiting_confirmation": None,
    "last_intent": None
}


# ------------------------------------------------------
# INTENT RESPONSES
# ------------------------------------------------------
def answer_intent(service, intent):
    info = knowledge_base[service]

    if intent == "interest":
        return f"The interest rate for **{service}** is **{info['interest_rate']}%**."

    if intent == "minimum":
        return f"The minimum balance for **{service}** is **{info['minimum_balance']} AED**."

    if intent == "availability":
        status = "available" if info["available"] else "not available"
        return f"The **{service}** is currently **{status}**."

    if intent == "requirements":
        return f"To open **{service}**, you need: {info['requirements']}"

    if intent == "apply":
        return f"You can apply for **{service}** this way:\n{info['apply']}"

    if intent == "description":
        return info["description"]

    return "What else would you like to know?"


# ------------------------------------------------------
# MAIN CHATBOT LOGIC
# ------------------------------------------------------
def chatbot(user_input):
    text = user_input.lower().strip()

    # Exit
    if text in ["bye", "quit", "exit", "thanks"]:
        return "exit"

    # Greetings
    if text in ["hi", "hello", "hey"]:
        return "Hello! How can I help you with our banking services today?"

    # Help menu
    if text in ["help", "guide", "?", "options", "features"]:
        return help_message()

    # Spam → show help menu
    if looks_like_spam(text):
        return help_message()

    # YES / NO for "Did you mean?"
    if context["awaiting_confirmation"]:
        if text in ["yes", "yep", "yeah", "ok"]:
            chosen = context["awaiting_confirmation"]
            context["awaiting_confirmation"] = None
            context["active_service"] = chosen
            return f"Great! Continuing with **{chosen}**.\nWhat would you like to know?"
        if text in ["no", "nope"]:
            context["awaiting_confirmation"] = None
            return "Okay! Which service are you asking about?"
        return "Please answer with **yes** or **no**."

    # Service detection
    service, suggestion = match_service(text)
    intent = detect_intent(text)

    # Exact / partial match
    if service:
        context["active_service"] = service
        return (
            f"You selected **{service}**.\n"
            f"{knowledge_base[service]['description']}\n\n"
            "What would you like to know?"
        )

    # Suggest match
    if suggestion:
        context["awaiting_confirmation"] = suggestion
        return f"Did you mean **{suggestion}**?"

    # Intent without selecting service
    if intent:
        if not context["active_service"]:
            return "Sure — which account or loan are you asking about?"
        return answer_intent(context["active_service"], intent)

    # Default fallback
    return help_message()


# ------------------------------------------------------
# RUN CHATBOT
# ------------------------------------------------------
print("Hello! How can I help you with our banking services today?")
while True:
    user = input("You: ")
    reply = chatbot(user)
    if reply == "exit":
        print("Chatbot: Thank you for banking with us. Goodbye!")
        break
    print("Chatbot:", reply)
