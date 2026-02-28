import pandas as pd
from transformers import pipeline

# 1. Load the Modern NLP Pipelines (Downloads models on first run)
# Using a standard robust sentiment model
print("Loading NLP models...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Using a BART model for Zero-Shot Classification
# This allows us to define our own categories on the fly without any training!
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2. Define the target categories for our agency routing
agency_departments = ["Technical Support", "Billing & Invoicing", "New Project Inquiry", "General Feedback"]

# 3. The Scenario Data: A batch of incoming client emails to 'NexusPoint'
incoming_tickets = [
    "My website has been down for the last two hours! I'm losing customers, please fix this immediately!",
    "Hi team, I loved the recent design updates. Can we schedule a call to discuss adding a new e-commerce module to the site?",
    "I was looking at my recent invoice and I think I was double-charged for the hosting fee. Can someone review this?",
    "The new API integration works flawlessly. Great job on getting that deployed ahead of schedule."
]

# 4. Process the tickets
processed_data = []

for ticket in incoming_tickets:
    # Get Sentiment
    sentiment_result = sentiment_analyzer(ticket)[0]

    # Get Category (Zero-Shot)
    category_result = zero_shot_classifier(ticket, candidate_labels=agency_departments)
    top_category = category_result['labels'][0]
    confidence_score = category_result['scores'][0]

    # Structure the results
    processed_data.append({
        "Original Message": ticket,
        "Assigned Department": top_category,
        "Routing Confidence": f"{confidence_score:.2%}",
        "Client Sentiment": sentiment_result['label'],
        "Requires Urgent Escalation": "Yes" if sentiment_result['label'] == "NEGATIVE" else "No"
    })

# 5. Output the results using Pandas for clean data manipulation
df = pd.DataFrame(processed_data)

print("\n--- Automated Ticket Triage Results ---\n")
print(df.to_string(index=False))