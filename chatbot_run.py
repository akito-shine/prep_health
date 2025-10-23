

# chatbot_run.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from chatbot_flow import QUESTIONS, classify_review  # your existing questions & logic

# --- Set device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load model & tokenizer ---
model_name = "google/flan-t5-small"   # or "google/flan-t5-large" if still too big
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# --- Normalize free-text answer ---
def normalize_answer(question_id, user_input):
    options_map = {
        "prep_history": ["yes", "no"],
        "partners": ["0-3", "3-6", ">7", "prefer not to say"],
        "condom_use": ["always", "sometimes", "never"],
        "partner_type": ["injectable", "gay", "bisexual", "transgender", "none"],
        "preventive_options": ["doxycycline", "at-home hiv test kit", "anal care", "hsv antivirals", "none"],
        "vaccinations": ["hpv", "hepatitis a", "hepatitis b", "mpox", "gonorrhea", "none"],
        "prep_followup": ["yes", "no"],
        "prep_next_step": [
            "schedule a virtual doctor",
            "send me an hiv self test kit",
            "full sexual health screening"
        ]
    }

    # Build prompt
    if question_id in ["prep_history", "partners", "condom_use","prep_followup","prep_next_step"]:
        prompt = (
            f'User answer: "{user_input}"\n'
            f'Options: {options_map[question_id]}\n'
            'Interpret the user meaning carefully and choose the correct keyword from the list based on meaning, not exact words. Respond ONLY with one keyword.'
        )
    elif question_id in ["partner_type", "preventive_options", "vaccinations"]:
        prompt = (
            f'You are helping to classify the user\'s answer into clear medical keywords.\n'
            f'Question category: {question_id}\n'
            f'User answer: "{user_input}"\n'
            f'Valid keywords: {options_map[question_id]}\n\n'
            'Identify which of the valid keywords are mentioned. Respond ONLY with the keywords, separated by commas if multiple. If none match, respond exactly with "none".'
        )
    else:
        prompt = f'Question: {question_id}\nUser answer: "{user_input}"\nRespond with a simple keyword suitable for rule checking.'

    # Run model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=20)
    normalized = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

    # Post-processing
    if question_id in options_map:
        valid_options = options_map[question_id]
        if question_id in ["prep_history", "partners", "condom_use", "prep_followup", "prep_next_step"]:
            normalized_clean = normalized.lower().strip()
            matched = False
            for kw in valid_options:
                kw_clean = kw.lower().strip()
                if kw_clean in normalized_clean or normalized_clean in kw_clean:
                    normalized = kw
                    matched = True
                    break
            if not matched:
                normalized = "none"
        else:
            normalized_clean = normalized.lower().strip()
            found = [kw for kw in valid_options if kw.lower() in normalized_clean]
            normalized = ", ".join(found) if found else "none"

    return normalized
def generate_summary(answers, question_texts):
    """
    Generates a PrEP recommendation with clear justification.
    """

    # Step 1: Prepare readable Q&A
    qa_text = ""
    for key, value in answers.items():
        if not key.endswith("_normalized"):
            question_text = question_texts.get(key, key.replace('_', ' ').title())
            qa_text += f"Question: {question_text}\nAnswer: {value}\n"

    #Step 2: Strong prompt to enforce reasoning
    summary_prompt = (
        "You are a digital sexual health assistant. Carefully read the user's answers below. "
        "Your task is to provide a single **recommendation**: either the user can self-check/self-initiate PrEP (eligible) "
        "or they should see a doctor.\n\n"
        "Immediately after the recommendation, provide a **clear, concise justification** for your recommendation. "
        "Cite key answers such as PrEP history, HIV test results, number of partners, condom use, partner types, preventive measures, and vaccinations. "
        "Do NOT list all answers; only refer to the relevant points supporting your recommendation. "
        "Write in one paragraph, supportive tone, and in the second person ('You'). "
        "Do NOT give multiple options or hedged statements.\n\n"
        f"User's answers:\n{qa_text}\n\n"
        "Write the recommendation and justification paragraph:"
    )

    # Step 3: Generate
    inputs = tokenizer(summary_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=400)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    summary = summary.replace("This user", "You").replace("the user", "you")
    return summary




def run_chatbot():
    print("Hi! Let's go through a short PrEP pre-screening.\n")
    answers = {}
    question_texts={}
    # 1) Ask all main questions first
    for q in QUESTIONS:
        user_input = input(q["text"] + "\nYour answer: ").strip()
        answers[q["id"]] = user_input
        question_texts[q["id"]] = q["text"]  # store question text
        if q["id"] != "whatsapp":
            normalized = normalize_answer(q["id"], user_input)
            answers[q["id"] + "_normalized"] = normalized
            print(f"(Normalized answer: {normalized})\n")

    # 2) Conditionally ask follow-ups after main questions
    if answers.get("prep_history_normalized") == "yes":
        # First follow-up
        followup_q = {
            "id": "prep_followup",
            "text": "Do you have a PrEP prescription or recent HIV blood test?"
        }

        user_input = input(followup_q["text"] + "\nYour answer: ").strip()
        answers[followup_q["id"]] = user_input
        question_texts[followup_q["id"]] = followup_q["text"]
        normalized = normalize_answer(followup_q["id"], user_input)
        answers[followup_q["id"] + "_normalized"] = normalized
        print(f"(Normalized answer: {normalized})\n")

        # Second follow-up if "no"
        if normalized == "no":
            final_q = {
                "id": "prep_nextstep",
                "text": (
                    "You will need to schedule an HIV blood test. Please choose an option by typing 1 or 2 or 3:\n"
                    "1. Schedule a virtual doctor\n"
                    "2. Send me an HIV self-test kit\n"
                    "3. Full Sexual Health Screening"
                )
            }

            user_input = input(final_q["text"] + "\nYour response: ").strip()
            answers[final_q["id"]] = user_input
            question_texts[final_q["id"]] = final_q["text"]
            normalized = normalize_answer(final_q["id"], user_input)
            answers[final_q["id"] + "_normalized"] = normalized
            print(f"(Normalized next step: {normalized})\n")

            # Trigger upload prompt if user chose HIV self-test kit (either by typing 2 or text)
            if normalized == "send me an hiv self test kit" or user_input.strip() == "2":
                upload_q = {
                    "id": "hiv_test_upload",
                    "text": "Please upload your HIV test result (or type 'uploaded' once done):"
                }
                user_input = input(upload_q["text"] + "\nYour response: ").strip()
                answers[upload_q["id"]] = user_input
                question_texts[upload_q["id"]] = upload_q["text"]
                print("(HIV test upload recorded)\n")

                # --- Ask PrEP usage plan ---
                prep_plan_q = {
                    "id": "prep_plan",
                    "text": "Do you plan to use PrEP daily or on-demand?"
                }

                user_input = input(prep_plan_q["text"] + "\nYour answer: ").strip()
                answers[prep_plan_q["id"]] = user_input
                question_texts[prep_plan_q["id"]] = prep_plan_q["text"]
                normalized = normalize_answer(prep_plan_q["id"], user_input)
                answers[prep_plan_q["id"] + "_normalized"] = normalized
                print(f"(Normalized answer: {normalized})\n")

                prep_refills = {
                    "id": "prep_refills",
                    "text": "What’s the most convenient way for you to receive your PrEP refills?Quartly scription or Ordered as needed?"
                }

                user_input = input(prep_refills["text"] + "\nYour answer: ").strip()
                answers[prep_refills["id"]] = user_input
                question_texts[prep_refills["id"]] = prep_refills["text"]
                normalized = normalize_answer(prep_refills["id"], user_input)
                answers[prep_refills["id"] + "_normalized"] = normalized
                print(f"(Normalized answer: {normalized})\n")
                

            if normalized == "Full Sexual Health Screening" or user_input.strip() == "3":
                schedule_time = {
                    "id": "schedule_time",
                    "text": "Please schedule a time for your full Sexual Health Screening at Shield Clinic:"
                }
                user_input = input(schedule_time["text"] + "\nYour response: ").strip()
                answers[schedule_time["id"]] = user_input
                question_texts[schedule_time["id"]] = schedule_time["text"]
                print("(Schedule a time for Full Sexual Health Screening)\n")
            
        else:
            upload_bloodtest = {
                    "id": "upload_bloodtest",
                    "text": "Please upload your HIV blood test results or prescription:"
                }
            user_input = input(upload_bloodtest["text"] + "\nYour response: ").strip()
            answers[upload_bloodtest["id"]] = user_input
            question_texts[upload_bloodtest["id"]] = upload_bloodtest["text"]
            print("(Upload succeeded\n")

            
            prep_plan_q = {
                    "id": "prep_plan",
                    "text": "Do you plan to use PrEP daily or on-demand?"
                }

            user_input = input(prep_plan_q["text"] + "\nYour answer: ").strip()
            answers[prep_plan_q["id"]] = user_input
            question_texts[prep_plan_q["id"]] = prep_plan_q["text"]
            normalized = normalize_answer(prep_plan_q["id"], user_input)
            answers[prep_plan_q["id"] + "_normalized"] = normalized
            print(f"(Normalized answer: {normalized})\n")



            prep_refills = {
                    "id": "prep_refills",
                    "text": "What’s the most convenient way for you to receive your PrEP refills?Quartly scription or Ordered as needed?"
                }

            user_input = input(prep_refills["text"] + "\nYour answer: ").strip()
            answers[prep_refills["id"]] = user_input
            question_texts[prep_refills["id"]] = prep_refills["text"]
            normalized = normalize_answer(prep_refills["id"], user_input)
            answers[prep_refills["id"] + "_normalized"] = normalized
            print(f"(Normalized answer: {normalized})\n")
    else:
        see_doctor = {
            "id": "see_doctor",
            "text": "It is recommended for first time PrEP users to consult a doctor: \n"
            "A. Speak virtually with a sexual helth doctor and get PrEP prescribed. \n"
            "B. Self check using AI Sexual Health Doctor for PrEP prescription. \n"
        }

        user_input = input(see_doctor["text"] + "\nYour answer: ").strip()
        answers[see_doctor["id"]] = user_input
        question_texts[see_doctor["id"]] = see_doctor["text"]
        normalized = normalize_answer(see_doctor["id"], user_input)
        answers[see_doctor["id"] + "_normalized"] = normalized
        print(f"(Normalized answer: {normalized})\n")
    
    email = {
            "id": "email",
            "text": "Do you have a PrEP prescription or recent HIV blood test?"
        }

    user_input = input(email["id"] + "\nYour answer: ").strip()
    answers[email["id"]] = user_input
    normalized = normalize_answer(email["id"], user_input)
    answers[email["id"] + "_normalized"] = normalized
    print(f"(Normalized answer: {normalized})\n")

    # After collecting all answers
    summary = generate_summary(answers,question_texts)
    print("\n--- User Summary ---")
    print(summary)

    # Done
    print("Submitted!")
    for k, v in answers.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_chatbot()
