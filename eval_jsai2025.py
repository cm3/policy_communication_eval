import csv
from openai import OpenAI
import re

# Initialize OpenAI client (replace with your actual API key by specifying "api_key = ***" as an argument)
client = OpenAI()

# --- Common System Prompt ---
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

# --- Evaluation Criteria System Prompts ---
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
   - 2: Proposal was not negated but they didn't make any change on their side. If they insist the request should be already fulfilled, this score is to be applied.
   - 1: Proposal was rejected and the request is not fulfilled.

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

def evaluate_criterion(system_prompt, text_input, expected_format):
    """
    system_prompt: The system prompt containing evaluation instructions.
    text_input: The user-provided input text (full comment and/or reply).
    expected_format: A description of the desired output format.
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

# --- File paths ---
input_file = './sample.csv'
output_file = './result.csv'
raw_log_file = './raw_api_logs.txt'

# --- Read CSV file ---
with open(input_file, 'r', encoding="utf8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# --- Set up output CSV fieldnames (original fields plus evaluation results) ---
fieldnames = reader.fieldnames + [
    "comment_attribute_score", "comment_attribute_justification",
    "request_presence_score", "request_presence_justification",
    "request_fulfillment_score", "request_fulfillment_justification",
    "meaningful_reply_score", "meaningful_reply_justification"
]

# --- Prepare raw logs container ---
raw_log_lines = []

# --- Evaluate each row using the OpenAI client ---
for idx, row in enumerate(rows):
    comment = row.get("comment", "")
    reply = row.get("reply", "")
    
    # Use full text for comment and reply.
    full_comment = comment
    full_reply = reply
    
    # Define expected output format for each evaluation including the row ID.
    row_id = idx + 1  # Alternatively, use a CSV field if available.
    comment_attribute_format = f"ID: {row_id}\nComment Attribute Score: [score]\nComment Attribute Justification: [justification]"
    request_presence_format = f"ID: {row_id}\nRequest Presence Score: [score]\nRequest Presence Justification: [justification]"
    request_fulfillment_format = f"ID: {row_id}\nRequest Fulfillment Score: [score]\nRequest Fulfillment Justification: [justification]"
    meaningful_reply_format = f"ID: {row_id}\nMeaningful Reply Score: [score]\nMeaningful Reply Justification: [justification]"
    
    # Execute each evaluation and record raw responses.
    ca_result = evaluate_criterion(
        comment_attribute_system_prompt,
        f"Comment: {full_comment}",
        comment_attribute_format
    )
    raw_log_lines.append(f"Row {row_id} - Comment Attribute Evaluation:\n{ca_result}\n{'-'*40}\n")
    
    rp_result = evaluate_criterion(
        request_presence_system_prompt,
        f"Comment: {full_comment}",
        request_presence_format
    )
    raw_log_lines.append(f"Row {row_id} - Request Presence Evaluation:\n{rp_result}\n{'-'*40}\n")
    
    rf_result = evaluate_criterion(
        request_fulfillment_system_prompt,
        f"Comment: {full_comment}\nReply: {full_reply}",
        request_fulfillment_format
    )
    raw_log_lines.append(f"Row {row_id} - Request Fulfillment Evaluation:\n{rf_result}\n{'-'*40}\n")
    
    mr_result = evaluate_criterion(
        meaningful_reply_system_prompt,
        f"Comment: {full_comment}\nReply: {full_reply}",
        meaningful_reply_format
    )
    raw_log_lines.append(f"Row {row_id} - Meaningful Reply Evaluation:\n{mr_result}\n{'-'*40}\n")
    
    # --- Extract scores and justifications using regex ---
    # Comment Attribute extraction
    ca_score = ""
    ca_justification = ""
    for line in ca_result.split('\n'):
        score_match = re.search(r"score:\s*(.+)", line, re.IGNORECASE)
        justification_match = re.search(r"justification:\s*(.+)", line, re.IGNORECASE)
        if score_match:
            ca_score = score_match.group(1).strip()
        if justification_match:
            ca_justification = justification_match.group(1).strip()
    row["comment_attribute_score"] = ca_score
    row["comment_attribute_justification"] = ca_justification

    # Request Presence extraction
    rp_score = ""
    rp_justification = ""
    for line in rp_result.split('\n'):
        score_match = re.search(r"score:\s*(.+)", line, re.IGNORECASE)
        justification_match = re.search(r"justification:\s*(.+)", line, re.IGNORECASE)
        if score_match:
            rp_score = score_match.group(1).strip()
        if justification_match:
            rp_justification = justification_match.group(1).strip()
    row["request_presence_score"] = rp_score
    row["request_presence_justification"] = rp_justification

    # Request Fulfillment extraction
    rf_score = ""
    rf_justification = ""
    for line in rf_result.split('\n'):
        score_match = re.search(r"score:\s*(.+)", line, re.IGNORECASE)
        justification_match = re.search(r"justification:\s*(.+)", line, re.IGNORECASE)
        if score_match:
            rf_score = score_match.group(1).strip()
        if justification_match:
            rf_justification = justification_match.group(1).strip()
    row["request_fulfillment_score"] = rf_score
    row["request_fulfillment_justification"] = rf_justification

    # Meaningful Reply extraction
    mr_score = ""
    mr_justification = ""
    for line in mr_result.split('\n'):
        score_match = re.search(r"score:\s*(.+)", line, re.IGNORECASE)
        justification_match = re.search(r"justification:\s*(.+)", line, re.IGNORECASE)
        if score_match:
            mr_score = score_match.group(1).strip()
        if justification_match:
            mr_justification = justification_match.group(1).strip()
    row["meaningful_reply_score"] = mr_score
    row["meaningful_reply_justification"] = mr_justification

# --- Save the updated rows to a new CSV file ---
with open(output_file, 'w', newline='', encoding="utf8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Evaluation results saved to {output_file}")

# --- Save raw API responses to a separate log file ---
with open(raw_log_file, 'w', encoding="utf8") as log_file:
    log_file.write("\n".join(raw_log_lines))

print(f"Raw API responses saved to {raw_log_file}")
