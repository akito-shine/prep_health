import streamlit as st
from chatbot_run import normalize_answer
from chatbot_flow import QUESTIONS, classify_review

st.set_page_config(page_title="PrEP Chatbot", page_icon="💬", layout="centered")
st.title("💬 PrEP Pre-Screening Chatbot")

# --- Initialize session state ---
if "step" not in st.session_state:
    st.session_state.step = 0
if "conversation" not in st.session_state:
    # Start with the first question
    st.session_state.conversation = [{"sender": "bot", "message": QUESTIONS[0]["text"]}]
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "followup_active" not in st.session_state:
    st.session_state.followup_active = False
if "followup_step" not in st.session_state:
    st.session_state.followup_step = 0

# --- Display conversation ---
for turn in st.session_state.conversation:
    if turn["sender"] == "bot":
        st.chat_message("assistant").markdown(turn["message"])
    else:
        st.chat_message("user").markdown(turn["message"])

# --- Function to handle input ---
def handle_input(user_input):
    # Determine current question
    if not st.session_state.followup_active:
        current_q = QUESTIONS[st.session_state.step]
    else:
        current_q = st.session_state.followup_questions[st.session_state.followup_step]

    # Save user answer
    st.session_state.conversation.append({"sender": "user", "message": user_input})
    st.session_state.answers[current_q["id"]] = user_input

    # Normalize if applicable
    if current_q["id"] != "whatsapp":
        normalized = normalize_answer(current_q["id"], user_input)
        st.session_state.answers[current_q["id"] + "_normalized"] = normalized
    else:
        normalized = user_input

    # Determine next step
    if not st.session_state.followup_active:
        st.session_state.step += 1
        # Check if we need to trigger follow-up questions
        if current_q["id"] == "prep_history":
            if normalized == "yes":
                st.session_state.followup_active = True
                st.session_state.followup_questions = [
                    {"id": "prep_followup", "text": "Do you have a PrEP prescription or recent HIV blood test?"}
                ]
                st.session_state.followup_step = 0
    else:
        st.session_state.followup_step += 1

    # Determine next bot message
    if not st.session_state.followup_active:
        if st.session_state.step < len(QUESTIONS):
            next_q = QUESTIONS[st.session_state.step]["text"]
            st.session_state.conversation.append({"sender": "bot", "message": next_q})
        else:
            # End of questions, show final decision
            needs_review, reason = classify_review(st.session_state.answers)
            final_message = (
                f"Doctor review REQUIRED\n**Reason:** {reason}" 
                if needs_review 
                else f"ℹ No doctor review needed\n**Reason:** {reason}"
            )
            st.session_state.conversation.append({"sender": "bot", "message": final_message})
    else:
        if st.session_state.followup_step < len(st.session_state.followup_questions):
            next_q = st.session_state.followup_questions[st.session_state.followup_step]["text"]
            st.session_state.conversation.append({"sender": "bot", "message": next_q})
        else:
            st.session_state.followup_active = False
            # After follow-ups, check if final message needed
            needs_review, reason = classify_review(st.session_state.answers)
            final_message = (
                f"Doctor review REQUIRED\n**Reason:** {reason}" 
                if needs_review 
                else f"ℹ No doctor review needed\n**Reason:** {reason}"
            )
            st.session_state.conversation.append({"sender": "bot", "message": final_message})

# --- Display input box for current question ---
if (
    (not st.session_state.followup_active and st.session_state.step < len(QUESTIONS)) 
    or (st.session_state.followup_active and st.session_state.followup_step < len(st.session_state.followup_questions))
):
    user_input = st.chat_input("Type your answer here...")
    if user_input:
        handle_input(user_input)
