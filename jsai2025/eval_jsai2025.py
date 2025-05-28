import csv
import re
import argparse
from openai import OpenAI

# ---------------------------
# Initialization and Prompts
# ---------------------------

def init_openai_client(api_key):
    """Initialize and return the OpenAI client using the provided API key."""
    return OpenAI(api_key=api_key)

# Common system prompt shared by all evaluations.
system_prompt_common = ("""
You are an expert evaluator assessing public comments and administrative responses regarding the Basic Environmental Plan in Japan.
Follow the instructions, output format, and evaluation criteria below for your work.

-----------------------
INSTRUCTIONS
-----------------------
1. Read each comment and reply carefully.
2. Assign a score for {criteria}.
3. Provide a brief justification (one or two sentences) after the score.
4. Output your evaluation in the specified line-based format.
5. Please remain objective and evaluate strictly based on the given criteria.

-----------------------
OUTPUT FORMAT
-----------------------
For each comment–reply pair, output:

ID: {row_id}
{criteria} Score: [score]
{criteria} Justification: [justification]

-----------------------
EVALUATION CRITERIA
-----------------------
""")

# Evaluation prompts for each criterion.
comment_attribute_system_prompt = (
    system_prompt_common + """

[Comment Attribute Evaluation]

Please evaluate the comment based on its relevance to the Basic Environmental Plan. Determine to what extent the comment should be reflected in the plan. Consider the following:

   *Scoring:*
   - 3: The comment clearly relates to the Basic Environmental Plan and addresses issues that should be directly reflected in the plan.
   - 2: The comment is partly relevant. It touches on aspects related to the Basic Environmental Plan, but the suggestions are either somewhat vague or the primary response might be more appropriately handled by other legislation or specific implementation plans.
   - 1: The comment is unrelated to the Basic Environmental Plan, or it addresses issues that fall completely outside its intended scope.

  Supplementary Guidance:
   - Comments that reference specific pages, chapters, or sections of the plan indicate strong relevance (favoring a Score 3).
   - Direct mentions of “Basic Environmental Plan” or requirement-related keywords (such as “revision” or “deletion”) typically signal clear relevance.
   - Even if the comment focuses on a linguistic or terminological revision, if it pertains specifically to ensuring the proper articulation of the plan’s content, it should be scored as 3.
   - If the comment lacks pertinent keywords, context, or clearly addresses issues that are not central to the plan, a lower score (2 or 1) should be assigned accordingly.
"""
)

request_presence_system_prompt = (
    system_prompt_common + """

[Request Presence Evaluation]

   *Scoring:*
   - 3: The comment contains a clear, specific request. For example, it explicitly states which part should be revised or added—such as "Revise section X by doing Y" or "Add detailed information about Z"—indicating exactly what change is needed.
   - 2: The comment includes a request, but it is vague or not clearly defined. For instance, the comment may indicate that the final vision or goal should be more explicit without specifying which aspects should be changed or how to change them.
   - 1: The comment does not contain any request. It may offer criticism or express an opinion, but it does not suggest any specific action or revision.
"""
)

request_fulfillment_system_prompt = (
    system_prompt_common + """

[Request Fulfillment Evaluation]

   *Scoring:*
   - 4: Completely fulfilled (e.g., the requested revision or addition is made exactly as asked).
   - 3: Partially fulfilled (some aspects of the request are met, or a different solution partially addresses it).
   - 2: The proposal was not negated but no change was made on their side. If they insist the request should be already fulfilled, this score is applied.
   - 1: The proposal was rejected and the request is not fulfilled.

   *Note:* If the administration states the request is already met but does not match the commenter’s specific expectation, it may still be rated 2 or 1 at your discretion.
"""
)

meaningful_reply_system_prompt = (
    system_prompt_common + """

[Meaningful Reply Evaluation]

   Evaluate whether the reply is “meaningful,” i.e., whether it demonstrates relevance, completeness, and clarity in addressing the comment.

   *Scoring:*  
   - 5: The reply is fully relevant, addresses the comment completely, and is clearly explained.
   - 4: The reply addresses the main points of the comment.
   - 3: The reply addresses part of the comment but lacks some clarity or completeness.
   - 2: The reply scarcely addresses the comment (insufficient or off-topic).
   - 1: The reply does not address the comment at all.
"""
)

# ---------------------------
# Core Functions
# ---------------------------

def evaluate_criterion(client, system_prompt, text_input, expected_format):
    """
    Evaluate a single criterion using the provided OpenAI client.
    
    Args:
        client: OpenAI client instance.
        system_prompt: The system prompt containing evaluation instructions.
        text_input: The input text (full comment and/or reply).
        expected_format: The desired output format including the row ID and criterion label.
    
    Returns:
        The raw evaluation result as a string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{text_input}\n\n{expected_format}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Adjust model name as needed
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            top_p=1
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        result = f"Error: {e}"
    return result

def load_csv_file(file_path):
    """Load a CSV file and return the list of rows and fieldnames."""
    with open(file_path, 'r', encoding="utf8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames

def save_csv_file(file_path, rows, fieldnames):
    """Save rows to a CSV file using the given fieldnames."""
    with open(file_path, 'w', newline='', encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def save_raw_logs(file_path, log_lines):
    """Save raw API response logs to a text file."""
    with open(file_path, 'w', encoding="utf8") as log_file:
        log_file.write("\n".join(log_lines))

def evaluate_rows(client, rows):
    """
    For each row, evaluate the four criteria using the OpenAI client.
    
    Returns:
        The updated rows and a list of raw log lines.
    """
    raw_log_lines = []
    for idx, row in enumerate(rows):
        comment = row.get("comment", "")
        reply = row.get("reply", "")
        
        # Use full text for comment and reply.
        full_comment = comment
        full_reply = reply
        
        row_id = idx + 1  # Using row number as ID.
        # Define expected formats for each evaluation including the row ID.
        comment_attribute_format = f"ID: {row_id}\nComment Attribute Score: [score]\nComment Attribute Justification: [justification]"
        request_presence_format = f"ID: {row_id}\nRequest Presence Score: [score]\nRequest Presence Justification: [justification]"
        request_fulfillment_format = f"ID: {row_id}\nRequest Fulfillment Score: [score]\nRequest Fulfillment Justification: [justification]"
        meaningful_reply_format = f"ID: {row_id}\nMeaningful Reply Score: [score]\nMeaningful Reply Justification: [justification]"
        
        # Evaluate each criterion and log the raw responses.
        ca_result = evaluate_criterion(client, comment_attribute_system_prompt,
                                       f"Comment: {full_comment}",
                                       comment_attribute_format)
        raw_log_lines.append(f"Row {row_id} - Comment Attribute Evaluation:\n{ca_result}\n{'-'*40}")
        
        rp_result = evaluate_criterion(client, request_presence_system_prompt,
                                       f"Comment: {full_comment}",
                                       request_presence_format)
        raw_log_lines.append(f"Row {row_id} - Request Presence Evaluation:\n{rp_result}\n{'-'*40}")
        
        rf_result = evaluate_criterion(client, request_fulfillment_system_prompt,
                                       f"Comment: {full_comment}\nReply: {full_reply}",
                                       request_fulfillment_format)
        raw_log_lines.append(f"Row {row_id} - Request Fulfillment Evaluation:\n{rf_result}\n{'-'*40}")
        
        mr_result = evaluate_criterion(client, meaningful_reply_system_prompt,
                                       f"Comment: {full_comment}\nReply: {full_reply}",
                                       meaningful_reply_format)
        raw_log_lines.append(f"Row {row_id} - Meaningful Reply Evaluation:\n{mr_result}\n{'-'*40}")
        
        # Extract scores and justifications using regex.
        row["comment_attribute_score"], row["comment_attribute_justification"] = extract_score_and_justification(ca_result)
        row["request_presence_score"], row["request_presence_justification"] = extract_score_and_justification(rp_result)
        row["request_fulfillment_score"], row["request_fulfillment_justification"] = extract_score_and_justification(rf_result)
        row["meaningful_reply_score"], row["meaningful_reply_justification"] = extract_score_and_justification(mr_result)
    return rows, raw_log_lines

def extract_score_and_justification(response_text):
    """
    Extract the score and justification from the response text using regex.
    
    Returns:
        A tuple (score, justification). If not found, empty strings are returned.
    """
    score = ""
    justification = ""
    for line in response_text.split('\n'):
        score_match = re.search(r"score:\s*(.+)", line, re.IGNORECASE)
        justification_match = re.search(r"justification:\s*(.+)", line, re.IGNORECASE)
        if score_match and not score:
            score = score_match.group(1).strip()
        if justification_match and not justification:
            justification = justification_match.group(1).strip()
    return score, justification

# ---------------------------
# Main Function and Argument Parsing
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate public comments and administrative responses for the Basic Environmental Plan.")
    parser.add_argument("input_csv", help="Input CSV filename (e.g., sample.csv)")
    parser.add_argument("output_csv", help="Output CSV filename (e.g., result.csv)")
    parser.add_argument("--raw_log", default="raw_api_logs.txt", help="Filename for raw API logs (default: raw_api_logs.txt)")
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    args = parser.parse_args()

    # Initialize OpenAI client.
    client = init_openai_client(args.api_key)

    # Load CSV data.
    rows, original_fieldnames = load_csv_file(args.input_csv)
    # Extend fieldnames with evaluation result fields.
    fieldnames = original_fieldnames + [
        "comment_attribute_score", "comment_attribute_justification",
        "request_presence_score", "request_presence_justification",
        "request_fulfillment_score", "request_fulfillment_justification",
        "meaningful_reply_score", "meaningful_reply_justification"
    ]

    # Evaluate each row.
    updated_rows, raw_logs = evaluate_rows(client, rows)

    # Save results.
    save_csv_file(args.output_csv, updated_rows, fieldnames)
    save_raw_logs(args.raw_log, raw_logs)

    print(f"Evaluation results saved to {args.output_csv}")
    print(f"Raw API responses saved to {args.raw_log}")

if __name__ == "__main__":
    main()
