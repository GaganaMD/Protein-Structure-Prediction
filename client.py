#%%
from gradio_client import Client

#%%
# client = Client("https://huggingface.co/spaces/GaganaMD/Protein-Structure-Prediction")
client = Client("http://localhost:7860")

# %%
result = client.predict(
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",	# str in 'sequence' Textbox component
    api_name="/esm2_embeddings")

# %%
result