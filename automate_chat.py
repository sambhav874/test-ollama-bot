# main.py
import random
import time
from datetime import datetime
from config import MAX_SESSION_TOKENS
from utils import *
from questions import questions_list

# Main chatbot function to interact with user
def qa_bot():
    prompt_template = set_custom_prompt()
    tokens_count = 0
    non_sense_count=0
    duration=60*time_minutes
    start_time = time.time()

    while (tokens_count <= MAX_SESSION_TOKENS) and (non_sense_count<100) and (time.time() - start_time < duration)  :

        query=random.choice(questions_list)
        print(f"YOU: {query}")
        ques_tok = count_tokens(query)
        tokens_count += ques_tok
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Handle greetings
        if is_greeting(query):
            greet_response = "ADA: Hello! How can I assist you with insurance-related questions today?"
            print(f"{greet_response} [Timestamp: {timestamp}]")
            print("-------------------------------------------------------")
            tokens_count += count_tokens(greet_response)
            continue

        # Handle appointment booking
        if is_booking_intent(query):
            appointment_booking()
            print("-------------------------------------------------------")
            continue

        # Check if query is insurance-related
        if not is_insurance_related(query):
            print("ADA: I'm here to help with insurance-related questions only.")
            print("-------------------------------------------------------")

            non_sense_count+=1
            continue

        # Generate and validate response
        validated_answer = generate_and_validate_response(query, db, prompt_template)

        # Token counts for the response
        ans_tok = count_tokens(validated_answer)
        tokens_count += ans_tok
        print(f"ADA: {validated_answer} [Timestamp: {timestamp}]")
        print(f"Tokens used: Question - {ques_tok}, Answer - {ans_tok}, Total - {tokens_count}")
        print("-------------------------------------------------------")

        # Define an expected response for scoring purposes
        context = get_relevant_context(query, db)
        expected_response = context  # Replace with your expected response for actual use

        # Calculate BLEU, ROUGE, Cosine Similarity, METEOR, F1, and Sentiment scores
        bleu_score = calculate_bleu(expected_response, validated_answer)
        rouge_score = calculate_rouge(expected_response, validated_answer)
        cosine_score = calculate_cosine_similarity(expected_response, validated_answer)
        meteor_score = calculate_meteor(expected_response, validated_answer)
        f1 = calculate_f1(expected_response, validated_answer)
        # sentiment_score = calculate_sentiment(validated_answer)

        # Save interaction with all scores
        save_interaction(
            timestamp, query, ques_tok, validated_answer, ans_tok, tokens_count,
            bleu_score, rouge_score, cosine_score, meteor_score, f1
        )

    # End session
    print("ADA: Thank you for chatting! The conversation has ended.")
    print("ADA: You have reached the token limit! Please contact support.")

if __name__ == "__main__":
    qa_bot()
