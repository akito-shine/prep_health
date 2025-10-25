# chatbot_run.py
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
# import torch
# from chatbot_flow import QUESTIONS, classify_review  # your existing questions & logic


# model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # smaller free model
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model with auto device mapping
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto" if device == "cuda" else None,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# )

# model.eval()

# chatbot_run.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
import torch
from chatbot_flow import QUESTIONS, classify_review  # your existing questions & logic

# --- Set device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load model & tokenizer ---
model_name = "google/flan-t5-large" #"google/flan-t5-small"   # or "google/flan-t5-large" if still too big
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# --- Normalize free-text answer ---
# def normalize_answer(question_id, user_input):
#     options_map = {
#         "prep_history": ["yes", "no"],
#         "partners": ["0-3", "3-6", ">7", "prefer not to say"],
#         "condom_use": ["always", "sometimes", "never"],
#         "partner_type": ["injectable", "gay", "bisexual", "transgender", "none"],
#         "preventive_options": ["doxycycline", "at-home hiv test kit", "anal care", "hsv antivirals", "none"],
#         "vaccinations": ["hpv", "hepatitis a", "hepatitis b", "mpox", "gonorrhea", "none"],
#         "prep_followup": ["yes", "no"],
#         "prep_next_step": [
#             "schedule a virtual doctor",
#             "send me an hiv self test kit",
#             "full sexual health screening"
#         ]
#     }

#     # Build prompt
#     if question_id in ["prep_history", "partners", "condom_use","prep_followup","prep_next_step"]:
#         prompt = (
#             f'User answer: "{user_input}"\n'
#             f'Options: {options_map[question_id]}\n'
#             'Interpret the user meaning carefully and choose the correct keyword from the list based on meaning, not exact words. Respond ONLY with one keyword.'
#             #'When the user say number for example 4, take the number and compare against options. When the user says number in text, try to correct interpret, for example five, it should be 3-5 and if 6 it should be >7. Only when users say nothing or any number in text or real number , just choose prefer not to say'
#         )
        
#     elif question_id in ["partner_type", "preventive_options", "vaccinations"]:
#         prompt = (
#             f'You are helping to classify the user\'s answer into clear medical keywords.\n'
#             f'Question category: {question_id}\n'
#             f'User answer: "{user_input}"\n'
#             f'Valid keywords: {options_map[question_id]}\n\n'
#             'Identify which of the valid keywords are mentioned. Respond ONLY with the keywords, separated by commas if multiple. If none match, respond exactly with "none".'
#         )
#     else:
#         prompt = f'Question: {question_id}\nUser answer: "{user_input}"\nRespond with a simple keyword suitable for rule checking.'

#     # Run model
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.inference_mode():
#         outputs = model.generate(**inputs, max_new_tokens=20)
#     normalized = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

#     # Post-processing
#     if question_id in options_map:
#         valid_options = options_map[question_id]
#         if question_id in ["prep_history", "partners", "condom_use", "prep_followup", "prep_next_step"]:
#             normalized_clean = normalized.lower().strip()
#             matched = False
#             for kw in valid_options:
#                 kw_clean = kw.lower().strip()
#                 if kw_clean in normalized_clean or normalized_clean in kw_clean:
#                     normalized = kw
#                     matched = True
#                     break
#             if not matched:
#                 normalized = "none"
#         else:
#             normalized_clean = normalized.lower().strip()
#             found = [kw for kw in valid_options if kw.lower() in normalized_clean]
#             normalized = ", ".join(found) if found else "none"

QUESTION_PROMPTS = {
    "prep_history": 'Interpret the user meaning carefully and choose the correct keyword from the list based on semantic meaning, not exact words from user input. Respond ONLY with one keyword.',
    "partners": 'When the user says a number of partners, map it to one of the ranges: 0-3, 3-6, >7, prefer not to say based on semantic meaning, not exact words from user input. Respond ONLY with one keyword.',
    "condom_use": 'Classify the user answer into always, sometimes, or never based on semantic meaning, not exact words from user input .Respond ONLY with one keyword .',
    "partner_type": 'Interpret the user meaning carefully.Identify partner types mentioned in the answer based on semantic meaning, not exact words from user input . Respond with one or more keywords separated by commas, or "none" if nothing matches depending on semantic meaning between user input and options.',
    "preventive_options": 'Identify preventive options mentioned in the answer based on semantic meaning, not exact words from user input. Respond with one or more keywords separated by commas, or "none" depending on semantic meaning between user input and options.',
    "vaccinations": 'Identify vaccinations mentioned in the answer. Respond with  keywords separated by commas, depending on semantic meaning between user input and options.',
    "prep_followup": 'Interpret the user answer carefully and respond ONLY with "yes" if they clearly indicate having a current PrEP prescription or recent HIV blood test, or "no" if they indicate not having one or are unsure. Do not guess; respond based on the meaning of their answer.',
    "prep_next_step": 'Interpret the user meaning carefully and choose the correct keyword from the list based on semantic meaning, not exact words from user input. Respond ONLY with one keyword from the options.'
}


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

    # Use a question-specific prompt
    prompt_instruction = QUESTION_PROMPTS.get(question_id, "")
    prompt = f'User answer: "{user_input}"\nOptions: {options_map.get(question_id, "any")}\n{prompt_instruction}'

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=20)
    normalized = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

    # Post-processing to ensure output is valid
    if question_id in options_map:
        valid_options = options_map[question_id]
        if question_id in ["prep_history", "partners", "condom_use", "prep_followup", "prep_next_step"]:
            matched = False
            for kw in valid_options:
                if kw.lower() in normalized or normalized in kw.lower():
                    normalized = kw
                    matched = True
                    break
            if not matched:
                normalized = "none"
        else:
            found = [kw for kw in valid_options if kw.lower() in normalized]
            normalized = ", ".join(found) if found else "none"

    return normalized

#     return normalized
# def generate_summary(answers):
#     """
#     Generates a PrEP recommendation with clear justification, using normalized answers.
#     """

#     # Step 1: Extract normalized answers
#     prep_history = answers.get("prep_history_normalized", "no")
#     partners = answers.get("partners_normalized", "none")
#     condom_use = answers.get("condom_use_normalized", "none")
#     partner_type = answers.get("partner_type_normalized", "none")
#     preventive_options = answers.get("preventive_options_normalized", "none")
#     vaccinations = answers.get("vaccinations_normalized", "none")
#     prep_followup = answers.get("prep_followup_normalized", "none")
#     prep_next_step = answers.get("prep_nextstep_normalized", "none")
#     prep_plan = answers.get("prep_plan_normalized", "none")
#     prep_refills = answers.get("prep_refills_normalized", "none")

#     # Step 2: Build a clear prompt
#     if prep_history == "no":
#         summary_prompt = (
#             "The user has never used PrEP before. "
#             "Provide a single, clear recommendation in one paragraph: advise seeing a doctor. "
#             "Write in a supportive tone, using 'You'. "
#             "Do not repeat the question list. Do not give multiple options.\n\n"
#             "Write the recommendation:"
#         )
#     else:
#         summary_prompt = (
#             "The user has previously used PrEP.Also Eligible to check out "
#             "Based on the following normalized answers, provide a single, clear recommendation on whether the user can self-check/self-initiate PrEP or should see a doctor. "
#             # "Write in one paragraph, in a supportive tone, using 'You'. Only mention relevant points that support the recommendation.\n\n"
#             # f"- Number of partners: {partners}\n"
#             # f"- Condom use: {condom_use}\n"
#             # f"- Partner type: {partner_type}\n"
#             # f"- Preventive options: {preventive_options}\n"
#             # f"- Vaccinations: {vaccinations}\n"
#             # f"- PrEP follow-up: {prep_followup}\n"
#             # f"- Next step: {prep_next_step}\n"
#             # f"- Planned PrEP usage: {prep_plan}\n"
#             # f"- Refill preference: {prep_refills}\n\n"
#             # "Write the recommendation:"
#         )


#     # Step 3: Generate
#     inputs = tokenizer(summary_prompt, return_tensors="pt").to(device)
#     with torch.inference_mode():
#         outputs = model.generate(**inputs, max_new_tokens=400)

#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     summary = summary.replace("This user", "You").replace("the user", "you")
#     return summary

def generate_summary(answers):
    """
    Generates a PrEP recommendation with clear justification, using normalized answers.
    Implements risk mapping for new and existing users based on defined conditions.
    """

    # Step 1: Extract normalized answers
    prep_history = answers.get("prep_history_normalized", "no")  # yes = existing, no = new
    partners = answers.get("partners_normalized", "none")        # 0-3, 3-6, >7
    condom_use = answers.get("condom_use_normalized", "none")    # always, sometimes, never
    partner_type = answers.get("partner_type_normalized", "none")# injectable, gay, bisexual, transgender, none
    prep_followup = answers.get("prep_followup_normalized", "none") # yes/no: prescription or recent HIV test

    # Step 2: Determine risk based on scenarios
    high_risk = False  # default

    if prep_history == "no":
        # --------------------------
        # Scenario: New User
        # --------------------------
        # High Risk Conditions for New User:
        # 1) Number of partners > 3 (partners == "3-6" or ">7")
        # 2) Condom use sometimes or never
        # 3) Partner type includes risky types (injectable, gay, bisexual, transgender)
        if partners in ["3-6", ">7"] or condom_use in ["sometimes", "never"] or partner_type != "none":
            high_risk = True  # HIGH RISK NEW USER
        else:
            high_risk = False # LOW RISK NEW USER

    else:
        # --------------------------
        # Scenario: Existing User
        # --------------------------
        # High Risk Conditions for Existing User:
        # 1) No recent prescription or HIV test (prep_followup == "no")
        # 2) Number of partners > 6 (partners == ">7")
        # 3) Condom use sometimes/never
        # 4) Partner type includes risky types (injectable, gay, bisexual, transgender)
        if prep_followup == "no" or partners == ">7" or condom_use in ["sometimes", "never"] or partner_type != "none":
            high_risk = True  # HIGH RISK EXISTING USER
        else:
            high_risk = False # LOW RISK EXISTING USER

    # Step 3: Build recommendation text based on four scenarios
    if prep_history == "no" and high_risk:
        # High risk new user
        recommendation_text = (
            "You are a new PrEP user with higher risk factors. "
            "It is strongly recommended that you see a doctor before starting PrEP. "
            "A doctor can provide proper evaluation, screening, and guidance for safe initiation."
        )
    elif prep_history == "no" and not high_risk:
        # Low risk new user
        recommendation_text = (
            "You are a new PrEP user at lower risk. "
            "You may self-check or self-initiate PrEP safely, while keeping regular monitoring and follow-up as advised."
        )
    elif prep_history == "yes" and high_risk:
        # High risk existing user
        recommendation_text = (
            "You are an existing PrEP user but have higher risk factors. "
            "Please see a doctor before continuing PrEP to ensure your ongoing eligibility and safety."
        )
    elif prep_history == "yes" and not high_risk:
        # Low risk existing user
        recommendation_text = (
            "You are an existing PrEP user at lower risk. "
            "You can safely self-check or continue PrEP on your own, while maintaining regular monitoring and follow-up."
        )

    # Step 4: Optional LLM step for supportive phrasing
    summary_prompt = (
        f"Rewrite the following PrEP recommendation in a supportive, friendly tone, using 'You':\n\n"
        f"{recommendation_text}\n\nWrite the recommendation:"
    )

    inputs = tokenizer(summary_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200)

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
