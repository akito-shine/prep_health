# chatbot_flow.py

# --- Conversation questions ---



# --- Decision rules ---
# def classify_review(answers):
#     """
#     Returns True if doctor review is needed, False otherwise.
#     """
#     # First-time user
#     if answers.get("prep_history_normalized") == "no":
#         return True, "First-time user — needs doctor education and baseline check."
    
#     # High-risk sexual behavior
#     if answers.get("partners_normalized") == ">7" or answers.get("condom_use_normalized") == "never":
#         return True, "High-risk behavior — doctor review suggested."
    
#     # Partner type risk
#     risky_partners = ["injectable", "gay", "bisexual", "transgender"]
#     partner_risk = answers.get("partner_type_normalized", "")
#     if any(rp in partner_risk for rp in risky_partners):
#         return True, "Sexual partners with higher risk — doctor review recommended."
    
#     # Default: no review needed
#     return False, "No doctor review required — you can proceed."

# chatbot_flow.py
QUESTIONS = [
    {"id": "prep_history", "text": "Have you used PrEP before?"},
    {"id": "partners", "text": "How many sexual encounters have you had in the past 3 months?"},
    {"id": "condom_use", "text": "How often do you use a condom?"},
    {"id": "partner_type", "text": "Have you had sex with anyone who is uses injectable drugs,gay/bisexual Mmn,transgender or Nnne"},
    {"id": "preventive_options", "text": "Would you consider using these sexual wellness preventive options? (Doxycycline, At-home HIV test kit, Post anal sex care, Antivirals for HSV)"},
    {"id": "vaccinations", "text": "Which of these vaccinations have you had? (HPV, Hepatitis A, Hepatitis B, MPox, Gonorrhea (Beserox))"},
    {"id": "whatsapp", "text": "What is your WhatsApp number?"}
]
def classify_review(answers):
    """
    Analyze user's answers for PrEP pre-screening.
    - If user has used PrEP before, ask about prescription or recent HIV blood test.
    - Return the follow-up analysis.
    """
    # Check if user has used PrEP before
    prep_history = answers.get("prep_history_normalized")
    if prep_history == "yes":
        # Analyze follow-up answer
        follow_up = answers.get("prep_followup_normalized")  # expects "yes" or "no"
        if follow_up == "yes":
            return False, "User has used PrEP before and has a prescription/recent HIV test."
        else:
            return True, "User has used PrEP before but no prescription or recent HIV test — follow-up recommended."
    
    # If user hasn't used PrEP, no follow-up needed
    return None, "User has not used PrEP before — no follow-up needed."
