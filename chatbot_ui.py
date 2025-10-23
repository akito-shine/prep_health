# chatbot_ui.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chatbot_flow import QUESTIONS
from chatbot_run import normalize_answer, generate_summary

# ------------------ MODEL LOAD ------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"  # use flan-t5-large for better speed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model, device


tokenizer, model, device = load_model()

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="PrEP Chatbot Prototype", layout="centered")
st.title("💬 PrEP Pre-Screening Chatbot")
st.caption("Powered by FLAN-T5 | Supports full pre-screening + follow-ups")

# ------------------ STATE INIT ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "question_texts" not in st.session_state:
    st.session_state.question_texts = {}
if "current_question" not in st.session_state:
    st.session_state.current_question = QUESTIONS[0]
if "completed" not in st.session_state:
    st.session_state.completed = False

# ------------------ DISPLAY INITIAL QUESTION ------------------
if len(st.session_state.messages) == 0:
    first_q = QUESTIONS[0]
    st.session_state.current_question = first_q
    st.session_state.messages.append({
        "role": "assistant",
        "content": first_q["text"]
    })

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------ FUNCTION: ASK NEXT QUESTION ------------------
def get_next_question():
    a = st.session_state.answers

    # 1️⃣ Next question logic identical to chatbot_run.py
    if "prep_history" not in a:
        return QUESTIONS[0]

    if "partners" not in a:
        return QUESTIONS[1]

    if "condom_use" not in a:
        return QUESTIONS[2]

    if "partner_type" not in a:
        return QUESTIONS[3]

    if "preventive_options" not in a:
        return QUESTIONS[4]

    if "vaccinations" not in a:
        return QUESTIONS[5]

    if "whatsapp" not in a:
        return QUESTIONS[6]

    # Branching logic after main questions
    if a.get("prep_history_normalized") == "yes":
        if "prep_followup" not in a:
            return {"id": "prep_followup", "text": "Do you have a PrEP prescription or recent HIV blood test?"}

        if a.get("prep_followup_normalized") == "no":
            if "prep_nextstep" not in a:
                return {
                    "id": "prep_nextstep",
                    "text": (
                        "You will need to schedule an HIV blood test. Please choose an option:\n"
                        "1. Schedule a virtual doctor\n"
                        "2. Send me an HIV self-test kit\n"
                        "3. Full Sexual Health Screening"
                    )
                }

            if a.get("prep_nextstep_normalized") in [
                "send me an hiv self test kit"
            ] or a.get("prep_nextstep") == "2":
                if "hiv_test_upload" not in a:
                    return {"id": "hiv_test_upload", "text": "Please upload your HIV test result (or type 'uploaded' once done):"}

                if "prep_plan" not in a:
                    return {"id": "prep_plan", "text": "Do you plan to use PrEP daily or on-demand?"}

                if "prep_refills" not in a:
                    return {"id": "prep_refills", "text": "What’s the most convenient way to receive your PrEP refills? Quarterly subscription or ordered as needed?"}

            if a.get("prep_nextstep_normalized") in [
                "full sexual health screening"
            ] or a.get("prep_nextstep") == "3":
                if "schedule_time" not in a:
                    return {"id": "schedule_time", "text": "Please schedule a time for your full Sexual Health Screening at Shield Clinic:"}

        elif a.get("prep_followup_normalized") == "yes":
            if "upload_bloodtest" not in a:
                return {"id": "upload_bloodtest", "text": "Please upload your HIV blood test results or prescription:"}

            if "prep_plan" not in a:
                return {"id": "prep_plan", "text": "Do you plan to use PrEP daily or on-demand?"}

            if "prep_refills" not in a:
                return {"id": "prep_refills", "text": "What’s the most convenient way to receive your PrEP refills? Quarterly subscription or ordered as needed?"}

    else:
        if "see_doctor" not in a:
            return {
                "id": "see_doctor",
                "text": (
                    "It is recommended for first-time PrEP users to consult a doctor:\n"
                    "A. Speak virtually with a sexual health doctor and get PrEP prescribed.\n"
                    "B. Self-check using AI Sexual Health Doctor for PrEP prescription."
                )
            }

    if "email" not in a:
        return {"id": "email", "text": "Please provide your email address for follow-up:"}

    return None  # no more questions


# ------------------ CHAT INPUT HANDLER (normalized)------------------
# if not st.session_state.completed:
#     if user_input := st.chat_input("Type your answer..."):
#         q = st.session_state.current_question
#         qid = q["id"]
#         qtext = q["text"]

#         # Record user message
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         # Save and normalize
#         st.session_state.answers[qid] = user_input.strip()
#         st.session_state.question_texts[qid] = qtext

#         # Normalize (except for contact info)
#         if qid not in ["whatsapp", "email", "schedule_time", "upload_bloodtest", "hiv_test_upload"]:
#             norm = normalize_answer(qid, user_input)
#             st.session_state.answers[qid + "_normalized"] = norm
#             bot_reply = f"(Normalized answer: **{norm}**)"
#         else:
#             bot_reply = "(Recorded.)"

#         st.session_state.messages.append({"role": "assistant", "content": bot_reply})

#         # Determine next question
#         next_q = get_next_question()
#         if next_q:
#             st.session_state.current_question = next_q
#             st.session_state.messages.append({"role": "assistant", "content": next_q["text"]})
#         else:
#             # Done! Generate summary
#             st.session_state.completed = True
#             summary = generate_summary(st.session_state.answers, st.session_state.question_texts)
#             summary = summary.replace("This user", "You")
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": "You've completed the PrEP pre-screening!\n\n**Summary:**\n" + summary
#             })

#         st.rerun()

# ------------------ CHAT INPUT HANDLER ------------------
if not st.session_state.completed:
    if user_input := st.chat_input("Type your answer..."):
        q = st.session_state.current_question
        qid = q["id"]
        qtext = q["text"]

        # Record user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Save answer internally
        st.session_state.answers[qid] = user_input.strip()
        st.session_state.question_texts[qid] = qtext

        # Normalize internally (do NOT display)
        if qid not in ["whatsapp", "email", "schedule_time", "upload_bloodtest", "hiv_test_upload"]:
            norm = normalize_answer(qid, user_input)
            st.session_state.answers[qid + "_normalized"] = norm

        # For bot reply, just a placeholder message (or empty) if you want no extra message
        bot_reply = ""  # keeps NameError away

        # Determine next question
        next_q = get_next_question()
        if next_q:
            st.session_state.current_question = next_q
            st.session_state.messages.append({"role": "assistant", "content": next_q["text"]})
        else:
            # Done! Generate summary
            st.session_state.completed = True
            summary = generate_summary(st.session_state.answers, st.session_state.question_texts)
            summary = summary.replace("This user", "You")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "You've completed the PrEP pre-screening!\n\n**Summary:**\n" + summary
            })

        st.rerun()

else:
    if st.button("Restart Chat"):
        for key in ["messages", "answers", "question_texts", "completed"]:
            st.session_state[key] = {} if key != "messages" else []
        st.session_state.current_question = QUESTIONS[0]
        # Immediately show the first question again
        st.session_state.messages.append({
            "role": "assistant",
            "content": QUESTIONS[0]["text"]
        })
        st.rerun()

