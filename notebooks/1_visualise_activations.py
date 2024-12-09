# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

from lm_memory_recall.activation_study import (
    ActivationTracker, 
    compute_activation_statistics,
    analyze_recollection_patterns,
    plot_activation_analysis,
    plot_top_activations
)

model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="mps",  
    torch_dtype="auto",  
    trust_remote_code=True,  
    attn_implementation='eager'
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

inputs = tokenizer('<|user|> What is the capital of France?<|end|><|assistant|>' , return_tensors="pt")

inputs

# +
# with torch.inference_mode():
#     outputs = model(**{k: v.to("mps") for k,v in inputs.items()})

# +
# outputs
# -

messages = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Which city in France holds the centre of power?"}],
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

inputs

tokenizer.decode(inputs[0])

# Run the analysis
statistics_results, activation_patterns = analyze_recollection_patterns(
    model=model, 
    queries=messages,
    tokenizer=tokenizer
)

# +
# # %debug
# -

# Plot different views of the activations
plot_activation_analysis(statistics_results, plot_type='percentile')

# Plot different views of the activations
plot_activation_analysis(statistics_results, plot_type='percentile')

plot_activation_analysis(statistics_results, plot_type='z_score')

plot_activation_analysis(statistics_results, plot_type='magnitude')

# Plot top activations
plot_top_activations(statistics_results, top_k=20)


