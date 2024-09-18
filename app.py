import gradio as gr
import pandas as pd
import requests
import json
import tiktoken
import matplotlib.pyplot as plt

# Constants
USD_TO_INR = 84
PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Fetch and process token costs
try:
    response = requests.get(PRICES_URL)
    if response.status_code == 200:
        TOKEN_COSTS = response.json()
    else:
        raise Exception(f"Failed to fetch token costs, status code: {response.status_code}")
except Exception as e:
    print(f'Failed to update token costs with error: {e}. Using static costs.')
    with open("model_prices.json", "r") as f:
        TOKEN_COSTS = json.load(f)

TOKEN_COSTS = pd.DataFrame.from_dict(TOKEN_COSTS, orient='index').reset_index()
TOKEN_COSTS.columns = ['model'] + list(TOKEN_COSTS.columns[1:])
TOKEN_COSTS = TOKEN_COSTS.loc[
    (~TOKEN_COSTS["model"].str.contains("sample_spec"))
    & (~TOKEN_COSTS["input_cost_per_token"].isnull())
    & (~TOKEN_COSTS["output_cost_per_token"].isnull())
    & (TOKEN_COSTS["input_cost_per_token"] > 0)
    & (TOKEN_COSTS["output_cost_per_token"] > 0)
]
TOKEN_COSTS["supports_vision"] = TOKEN_COSTS["supports_vision"].fillna(False)

# Convert USD costs to INR
TOKEN_COSTS["input_cost_per_token"] *= USD_TO_INR
TOKEN_COSTS["output_cost_per_token"] *= USD_TO_INR

def clean_names(s):
    s = s.replace("_", " ").replace("ai", "AI")
    return s[0].upper() + s[1:]

TOKEN_COSTS["litellm_provider"] = TOKEN_COSTS["litellm_provider"].apply(clean_names)

cmap = plt.get_cmap('RdYlGn_r')  # Red-Yellow-Green colormap, reversed

def count_string_tokens(string: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model.split('/')[-1])
    except:
        if len(model.split('/')) > 1:
            try:
                encoding = tiktoken.encoding_for_model(model.split('/')[-2] + '/' + model.split('/')[-1])
            except KeyError:
                print(f"Model {model} not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            print(f"Model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def calculate_total_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    model_data = TOKEN_COSTS[TOKEN_COSTS['model'] == model].iloc[0]
    prompt_cost = prompt_tokens * model_data['input_cost_per_token']
    
    
    

   
    completion_cost = completion_tokens * model_data['output_cost_per_token']
    
    return prompt_cost, completion_cost

def update_model_list(function_calling, litellm_provider, max_price, supports_vision, supports_max_input_tokens):
    filtered_models = TOKEN_COSTS.loc[TOKEN_COSTS["max_input_tokens"] >= supports_max_input_tokens*1000]

    if litellm_provider != "Any":
        filtered_models = filtered_models[filtered_models['litellm_provider'] == litellm_provider]
    
    if supports_vision:
        filtered_models = filtered_models[filtered_models['supports_vision']]
    
    list_models = filtered_models['model'].tolist()
    return gr.Dropdown(choices=list_models, value=list_models[0] if list_models else "No model found for this combination!")





def compute_all(input_type, prompt_text, completion_text, prompt_tokens, completion_tokens, models):
    results = []
    temp=prompt_tokens
    temp2=completion_tokens
    for model in models:
        if input_type == "Text Input":
            prompt_tokens = count_string_tokens(prompt_text, model)
            completion_tokens = count_string_tokens(completion_text, model)
        else:  # Token Count Input
            
            
            prompt_tokens= int(prompt_tokens * 1000)

            completion_tokens = int(completion_tokens * 1000)

        model_data = TOKEN_COSTS[TOKEN_COSTS['model'] == model].iloc[0]
        prompt_cost, completion_cost = calculate_total_cost(prompt_tokens, completion_tokens, model)
        
        total_cost = prompt_cost + completion_cost
              
        results.append({
            "Model": model,
            "Provider": model_data['litellm_provider'],
            "Input Cost / M tokens": model_data['input_cost_per_token']*1e6,
            "Output Cost / M tokens": model_data['output_cost_per_token']*1e6,
            "Total Cost": round(total_cost, 2),
        })
        prompt_tokens=temp
        completion_tokens=temp2
        
    
    df = pd.DataFrame(results)

    if len(df) > 1:
        norm = plt.Normalize(df['Total Cost'].min(), df['Total Cost'].max())
        
        def get_color(val):
            color = cmap(norm(val))
            return f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.3)'
        
    else:
        def get_color(val):
            return "rgba(0, 0, 0, 0)"
    
    # Create the HTML table with animations
    html_table = '<table class="styled-table">'
    html_table += '<thead><tr>'
    for col in df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    
    for i, row in df.iterrows():
        html_table += f'<tr class="animate-row" style="animation-delay: {i * 0.1}s;">'
        for col in df.columns:
            value = row[col]
            if col == 'Total Cost':
                color = get_color(value)
                html_table += f'<td class="total-cost" style="background-color: {color};">₹{value:.2f}</td>'
            elif col in ["Input Cost / M tokens", "Output Cost / M tokens"]:
                html_table += f'<td>₹{value:.2f}</td>'
            else:
                html_table += f'<td>{value}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    return html_table

def toggle_input_visibility(choice):
    return (
        gr.Group(visible=(choice == "Text Input")),
        gr.Group(visible=(choice == "Token Count Input"))
    )

with gr.Blocks(css="""
    .styled-table {
        border-collapse: separate;
        border-spacing: 0;
        margin: 25px 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        width: 100%;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        overflow: hidden;
        background-color: #f8f9fa;
    }
    .styled-table thead tr {
        background-color: #3a506b;
        color: #ffffff;
        text-align: left;
        font-weight: bold;
    }
    .styled-table th,
    .styled-table td {
        padding: 14px 18px;
        border-bottom: 1px solid #e0e0e0;
    }
    .styled-table tbody tr {
        transition: all 0.3s ease;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f0f4f8;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #3a506b;
    }
    .styled-table tbody tr:hover {
        background-color: #e3e8ef;
        transform: scale(1.02);
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .total-cost {
        font-weight: bold;
        transition: all 0.3s ease;
        color: #2c3e50;
    }
    .total-cost:hover {
        transform: scale(1.1);
        color: #e74c3c;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-row {
        animation: fadeIn 0.5s ease-out forwards;
        opacity: 0;
    }
    .styled-table tbody tr td {
        color: #34495e;
    }
    .styled-table tbody tr:hover td {
        color: #2c3e50;
    }
""", theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.slate)) as demo:
    gr.Markdown("""
    # 💰 Text-to-Rupees: Get the price of your LLM API calls in INR! 💰
    Based on prices data from [BerriAI's litellm](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).
    Prices converted to INR (1 USD = 84 INR).
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Input type:")
            input_type = gr.Radio(["Text Input", "Token Count Input"], label="Input Type", value="Text Input")
            
            with gr.Group() as text_input_group:
                prompt_text = gr.Textbox(label="Prompt", value="Tell me a joke about AI.", lines=3)
                completion_text = gr.Textbox(label="Completion", value="Certainly: Why did the neural network go to therapy? It had too many deep issues!", lines=3)
            
            with gr.Group(visible=False) as token_input_group:
                prompt_tokens_input = gr.Number(label="Prompt Tokens (thousands)", value=1.5)
                completion_tokens_input = gr.Number(label="Completion Tokens (thousands)", value=2)

        with gr.Column():
            gr.Markdown("## Model choice:")
            with gr.Row():
                with gr.Column():
                    function_calling = gr.Checkbox(label="Supports Tool Calling", value=False)
                    supports_vision = gr.Checkbox(label="Supports Vision", value=False)
                with gr.Column():
                    supports_max_input_tokens = gr.Slider(label="Min Supported Input Length (thousands)", minimum=2, maximum=256, step=2, value=2)
                    max_price = gr.Slider(label="Max Price per Input Token", minimum=0, maximum=0.084, step=0.00084, value=0.084, visible=False, interactive=False)
                litellm_provider = gr.Dropdown(label="Inference Provider", choices=["Any"] + TOKEN_COSTS['litellm_provider'].unique().tolist(), value="Any")
        
            model = gr.Dropdown(label="Models (at least 1)", choices=TOKEN_COSTS['model'].tolist(), value=["anyscale/meta-llama/Meta-Llama-3-8B-Instruct", "gpt-4o", "claude-3-sonnet-20240229"], multiselect=True)
        
    gr.Markdown("## Resulting Costs 👇")

    with gr.Row():
        results_table = gr.HTML()

    input_type.change(
        toggle_input_visibility,
        inputs=[input_type],
        outputs=[text_input_group, token_input_group]
    )

    gr.on(
        triggers=[function_calling.change, litellm_provider.change, max_price.change, supports_vision.change, supports_max_input_tokens.change],
        fn=update_model_list,
        inputs=[function_calling, litellm_provider, max_price, supports_vision, supports_max_input_tokens],
        outputs=model,
    )

    gr.on(
        triggers=[
            input_type.change,
            prompt_text.change,
            completion_text.change,
            prompt_tokens_input.change,
            completion_tokens_input.change,
            function_calling.change,
            litellm_provider.change,
            supports_vision.change,
            supports_max_input_tokens.change,
            model.change
        ],
        fn=compute_all,
        inputs=[
            input_type,
            prompt_text,
            completion_text,
            prompt_tokens_input,
            completion_tokens_input,
            model
        ],
        outputs=results_table
    )

    # Load results on page load
    demo.load(
        fn=compute_all,
        inputs=[
            input_type,
            prompt_text,
            completion_text,
            prompt_tokens_input,
            completion_tokens_input,
            model
        ],
        outputs=results_table
    )

if __name__ == "__main__":
    demo.launch()