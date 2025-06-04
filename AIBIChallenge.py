# Databricks notebook source
spark.range(1, 2).show()

# COMMAND ----------

# MAGIC %pip install numpy==1.26.4

# COMMAND ----------

# Install only what you need, and let Databricks manage core dependencies
%pip install transformers faiss-cpu openai

# COMMAND ----------

# MAGIC %pip install torch

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# 1. Install required libraries (run this in a Databricks notebook cell)
# %pip install transformers faiss-cpu torch

from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
import torch

# COMMAND ----------

# 2. Load your clinical notes data from a Delta table or CSV

df = spark.read.table("clinical_notes").toPandas()  

# COMMAND ----------

print(df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC **Here we had white spaces which are unnecessary for a most frequently used column name "note_text" so we remove those using strip function**

# COMMAND ----------

# Strip whitespace from column names (if any)
df.columns = df.columns.str.strip()

# COMMAND ----------

print(df.columns)

# COMMAND ----------

#Gather all the note_text data of all the patients to a list
notes = df['note_text'].tolist()

# COMMAND ----------

# 3. Load embedding model
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# COMMAND ----------

# 4. Embed clinical notes
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(1).cpu().numpy()

embeddings = np.vstack([embed_text(note) for note in notes])
print(embeddings)

# COMMAND ----------

# 5. Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(index)

# COMMAND ----------

# Importing openai and using the api key 
import openai
openai.api_key = "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"

# COMMAND ----------

# 6. Define RAG Query Function
def rag_query(query, notes, index, k=3):
    query_emb = embed_text(query)
    D, I = index.search(query_emb, k)
    context = [notes[i] for i in I[0]]
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}], 
        max_tokens=256,
        temperature=0.2
    )  
    return response, context

# COMMAND ----------

# 7. Example RAG Query
query = "What are the key considerations for chest pain"
response, context = rag_query(query, notes, index)
print("Context:", context)
print("Response:", response)

# COMMAND ----------

# 8. Logging and Monitoring
def log_interaction(user_id, query, response, demographics):
    log_entry = [{
        'user_id': user_id,
        'query': query,
        'response': response,
        'demographics': demographics,
        'timestamp': datetime.now()
    }]
    log_df = spark.createDataFrame(log_entry)
    log_df.write.mode("append").format("delta").saveAsTable("rag_logs")

# COMMAND ----------

# Example log
log_interaction(
    user_id="clinician_01",
    query=query,
    #response=response,
    response="This is a sample response", #This is taken as sample for testing as there is no API quota to get the response
    demographics="Male, 45"
)

# COMMAND ----------

# 9. Equity Analysis (pseudocode)
def analyze_equity(logs_df):
    # logs_df: DataFrame with columns ['demographics', 'response_quality', ...]
    # Group by demographics, compute metrics, flag disparities
    equity_report = logs_df.groupby('demographics').agg({'response_quality': 'mean'})
    return equity_report


# COMMAND ----------

dbutils.widgets.text("user_query", "What are key considerations for chest pain?", "Ask Genie")
user_query = dbutils.widgets.get("user_query")

# COMMAND ----------

response, context = rag_query(user_query, notes, index)
print("Genie Context:", context)
print("Genie Answer:", response)