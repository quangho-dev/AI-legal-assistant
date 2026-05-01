import pandas as pd

df = pd.read_csv("/Users/admin/Documents/Personal projects/AI assistant/evals/experiments/20260501-120624_naiverag.csv")
cols = ['correctness_score', 'faithfulness_score', 'answer_relevancy_score']
# Ensure numeric (optional but safe)
# df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# # Create result with average row
# result = df[cols].copy()
# result.loc['average'] = result.mean()

# print(result)

print(df.head())