import streamlit as st
from transformers import pipeline
from datetime import datetime
import random


# -----------------------------
# Load AI model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


classifier = load_model()


# -----------------------------
# Intent detection
# -----------------------------
def detect_intent(message):
    message = message.lower()

    if any(word in message for word in [
        "refund", "money back", "charged", "payment", "billing"
    ]):
        return "BILLING/REFUND"

    elif any(word in message for word in [
        "password", "login", "log in", "account"
    ]):
        return "ACCOUNT"

    elif any(word in message for word in [
        "order", "delivery", "shipping", "package"
    ]):
        return "ORDER"

    else:
        return "GENERAL"


# -----------------------------
# Agent tools
# -----------------------------
def generate_response(message):
    return (
        "Thank you for contacting customer support. "
        "We understand your concern and are happy to help."
    )


def create_support_ticket(message):
    ticket_id = "TKT-" + str(random.randint(1000, 9999))

    return {
        "ticket_id": ticket_id,
        "status": "Created",
        "priority": "High",
        "message": message,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def escalate_to_human(message):
    return {
        "status": "Escalated",
        "department": "Human Support",
        "priority": "High",
        "message": message
    }


# -----------------------------
# Customer Support Agent
# -----------------------------
def customer_support_agent(message):

    # AI model analyzes sentiment
    result = classifier(message)[0]

    sentiment = result["label"]
    confidence = result["score"]

    # Agent identifies intent
    intent = detect_intent(message)

    # Agent makes decision
    if confidence < 0.80:

        decision = "UNCERTAIN"

        action_result = {
            "action": "Ask for clarification",
            "response": (
                "I'm not completely sure I understand your issue. "
                "Could you please provide more details?"
            )
        }

    elif intent == "BILLING/REFUND":

        decision = "ESCALATE"

        ticket = create_support_ticket(message)
        escalation = escalate_to_human(message)

        action_result = {
            "action": "Create ticket and escalate",
            "ticket": ticket,
            "escalation": escalation
        }

    elif intent == "ACCOUNT":

        decision = "ACCOUNT_SUPPORT"

        action_result = {
            "action": "Send account support response",
            "response": (
                "Please use the password reset option on the login page "
                "to recover your account."
            )
        }

    elif intent == "ORDER":

        decision = "ORDER_SUPPORT"

        ticket = create_support_ticket(message)

        action_result = {
            "action": "Create order support ticket",
            "ticket": ticket
        }

    else:

        decision = "RESPOND"

        response = generate_response(message)

        action_result = {
            "action": "Send customer response",
            "response": response
        }

    return {
        "message": message,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "intent": intent,
        "decision": decision,
        "action": action_result
    }


# -----------------------------
# Streamlit Web Interface
# -----------------------------
st.title("🤖 AI Customer Support Decision Agent")

st.write(
    "Enter a customer message. The AI model analyzes the message "
    "and the agent decides what action to take."
)

message = st.text_area(
    "Customer Message",
    placeholder="Example: I was charged twice and want my money back."
)

if st.button("Analyze Message"):

    if message.strip():

        result = customer_support_agent(message)

        st.subheader("🧠 AI Model Analysis")

        st.write("**Sentiment:**", result["sentiment"])
        st.write("**Confidence:**", result["confidence"])

        st.subheader("🎯 Customer Intent")

        st.write(result["intent"])

        st.subheader("🤖 Agent Decision")

        st.write(result["decision"])

        st.subheader("🛠️ Agent Action")

        action = result["action"]

        st.write("**Action:**", action["action"])

        if "response" in action:
            st.success(action["response"])

        if "ticket" in action:

            ticket = action["ticket"]

            st.info(
                f"Ticket Created: {ticket['ticket_id']}\n\n"
                f"Priority: {ticket['priority']}\n\n"
                f"Status: {ticket['status']}"
            )

        if "escalation" in action:

            st.warning(
                "⚠️ The issue has been escalated to Human Support."
            )

    else:

        st.warning("Please enter a customer message.")
