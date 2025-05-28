# 環境基本計画のパブコメの評価

石井　康平, 亀田　尭宙, 小風　尚樹, 倉阪　秀史: "生成AIを用いたパブリックコメントの自動評価手法の提案", JSAI2025 で用いたプロンプトを示すために、コードを公開しています。

## Files

- **eval_jsai2025.py**  
  The main Python script that performs evaluations on comment–reply pairs from an input CSV file. The script:
  - Accepts an input CSV filename (see `sample_input.csv`), output CSV filename(`sample_output.csv`), and an API key as command-line arguments.
  - Evaluates each row using the OpenAI API for four criteria:  
    - **Comment Attribute** (How well the comment relates to the plan)
    - **Request Presence** (The clarity and specificity of requests in the comment)
    - **Request Fulfillment** (How well the reply meets the request)
    - **Meaningful Reply** (The overall quality and relevance of the reply)
  - Extracts scores and justifications via regular expressions.
  - Saves the evaluation results in a new CSV file.
  - Logs raw API responses into a separate text file for debugging and transparency.

- **result.csv**  
  The output CSV file generated after evaluation. About four criteria, check the content of `eval_jsai2025.py`.
  - `ver`: version number of 環境基本計画
  - `id`: id of each comment
  - `comment`: content of the comment
  - `reply`: content of the corresponding reply
  - `comment_attribute_score`
  - `comment_attribute_justification`
  - `request_presence_score`
  - `request_presence_justification`
  - `request_fulfillment_score`
  - `request_fulfillment_justification`
  - `meaningful_reply_score`
  - `meaningful_reply_justification`

- kappa
  - **kappa.py**: カッパ値を出すためのスクリプト
  - **CA.csv**, **RP.csv**, **RF.csv**, **MR.csv**: ChatGPT および 3人の評価者（R1〜R3）それぞれの評価結果。

## How to Use

To run the evaluation pipeline, use the following command from your terminal:

```bash
python eval_jsai2025.py sample_input.csv sample_output.csv --api_key YOUR_API_KEY
```
