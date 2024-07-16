import gradio as gr
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import torch
from logging import getLogger

logger = getLogger(__name__)

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs[0]

def fold_prot_locally(sequence):
    logger.info("Folding: " + sequence)
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()

    with torch.no_grad():
        output = model(tokenized_input)
    pdb = convert_outputs_to_pdb(output)
    return pdb

def get_esm2_embeddings(sequence):
    logger.info("Getting embeddings for: " + sequence)
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()

    with torch.no_grad():
        aa = tokenized_input 
        L = aa.shape[1]
        device = tokenized_input.device
        attention_mask = torch.ones_like(aa, device=device)

        # === ESM ===
        esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)
        esm_s = model.compute_language_model_representations(esmaa)

    return {"res": esm_s.cpu().tolist()}

def get_esmfold_embeddings(sequence):
    logger.info("Getting embeddings for: " + sequence)
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()

    with torch.no_grad():
        output = model(tokenized_input)

    return {"res": output["s_s"].cpu().tolist()}

def suggest(option):
   if option == "Plastic degradation protein":
     suggestion = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
   elif option == "Antifreeze protein":
     suggestion = "QCTGGADCTSCTGACTGCGNCPNAVTCTNSQHCVKANTCTGSTDCNTAQTCTNSKDCFEANTCTDSTNCYKATACTNSSGCPGH"
   elif option == "AI Generated protein":
     suggestion = "MSGMKKLYEYTVTTLDEFLEKLKEFILNTSKDKIYKLTITNPKLIKDIGKAIAKAAEIADVDPKEIEEMIKAVEENELTKLVITIEQTDDKYVIKVELENEDGLVHSFEIYFKNKEEMEKFLELLEKLISKLSGS"
   elif option == "7-bladed propeller fold":
     suggestion = "VKLAGNSSLCPINGWAVYSKDNSIRIGSKGDVFVIREPFISCSHLECRTFFLTQGALLNDKHSNGTVKDRSPHRTLMSCPVGEAPSPYNSRFESVAWSASACHDGTSWLTIGISGPDNGAVAVLKYNGIITDTIKSWRNNILRTQESECACVNGSCFTVMTDGPSNGQASYKIFKMEKGKVVKSVELDAPNYHYEECSCYPNAGEITCVCRDNWHGSNRPWVSFNQNLEYQIGYICSGVFGDNPRPNDGTGSCGPVSSNGAYGVKGFSFKYGNGVWIGRTKSTNSRSGFEMIWDPNGWTETDSSFSVKQDIVAITDWSGYSGSFVQHPELTGLDCIRPCFWVELIRGRPKESTIWTSGSSISFCGVNSDTVGWSWPDGAELPFTIDK"
   else:
     suggestion = ""
   return suggestion


def molecule(mol):
    x = (
        """<!DOCTYPE html>
        <html>
        <head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <style>
    body{
        font-family:sans-serif
    }
    .mol-container {
    width: 100%;
    height: 600px;
    position: relative;
    }
    .mol-container select{
        background-image:None;
    }
    </style>
     <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js" integrity="sha512-STof4xm1wgkfm7heWqFJVn58Hm3EtS31XFaagaa8VMReCXAkQnJZ+jEy8PCC/iT18dFy95WcExNHFTqLyp72eQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>  
    <div id="container" class="mol-container"></div>
  
            <script>
               let pdb = `"""
        + mol
        + """`  
      
             $(document).ready(function () {
                let element = $("#container");
                let config = { backgroundColor: "white" };
                let viewer = $3Dmol.createViewer(element, config);
                viewer.addModel(pdb, "pdb");
                viewer.getModel(0).setStyle({}, { cartoon: { colorscheme:"whiteCarbon" } });
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(0.8, 2000);
              })
        </script>
        </body></html>"""
    )

    return f"""<iframe style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""


sample_code = """
from gradio_client import Client
client = Client("https://wwydmanski-esmfold.hf.space/")
def fold_huggingface(sequence, fname=None):
    result = client.predict(
                    sequence,	# str in 'sequence' Textbox component
                    api_name="/pdb")
    if fname is None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".pdb", prefix="esmfold_") as fp:
            fp.write(result)
            fp.flush()
            return fp.name
    else:
        with open(fname, "w") as fp:
            fp.write(result)
            fp.flush()
        return fname
pdb_fname = fold_huggingface("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")
"""

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).cuda()
model.esm = model.esm.half()
torch.backends.cuda.matmul.allow_tf32 = True

with gr.Blocks() as demo:
    gr.Markdown("# ESMFold")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(lines=1, label="Sequence")
            name = gr.Dropdown(label="Choose a Sample Protein", value="Plastic degradation protein", choices=["Antifreeze protein", "Plastic degradation protein",  "AI Generated protein", "7-bladed propeller fold", "custom"])
            btn = gr.Button("🔬 Predict Structure ")
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Sample code")
            gr.Code(sample_code, label="Sample usage", language="python", interactive=False)

    with gr.Row():
        gr.Markdown("## Output")

    with gr.Row():
        with gr.Column():
            out = gr.Code(label="Output", interactive=False)
        with gr.Column():
            out_mol = gr.HTML(label="3D Structure")

    with gr.Row(visible=False):
       with gr.Column():
           gr.Markdown("## Embeddings")
           embs = gr.JSON(label="Embeddings")

    name.change(fn=suggest, inputs=name, outputs=inp)
    btn.click(fold_prot_locally, inputs=[inp], outputs=[out], api_name="pdb")
    btn.click(get_esmfold_embeddings, inputs=[inp], outputs=[embs], api_name="embeddings")
    btn.click(get_esm2_embeddings, inputs=[inp], outputs=[embs], api_name="esm2_embeddings")
    out.change(fn=molecule, inputs=[out], outputs=[out_mol], api_name="3d_fold")

demo.launch()