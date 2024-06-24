The notebook is taken from this [repository](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions) from Google Research. It inspects information flow as a smaller circuit that performs a 3 step attribute recall process: 

1. Subject Enrichment in MLPs with Vocabulary Projection and tested by Sublayer Knockout  

2. Relation Propagation to the last token via Attention Knockout

3. Attribute Extraction with Attention Heads

The code needs configuration for supporting GPT2-Small but it goes halfway through without changing anything. We tested for the indirect KBICR example with Alice lives in France. Images and statistics are saved in `kbicr_results/` for attention knockout and attribute extraction with MHSA throughout layers.

From the README:

""Our code can be applied as-is for other sizes of GPT2, and can be easily adjusted to other transformer-based models available on Huggingface. To adjust the experiments to other models, the following adaptations are needed:

(1) Code modifications. Different models on Huggingface have different naming schemes (e.g., the first MLP matrix is called c_fc in GPT2 and fc_in in GPT-J). The hooks used to extract intermediate information and intervene on the network computation are applied to modules by their names. Therefore, the part in the code that needs to be adjusted when switching to other models is the hooks, where the modification merely should adjust the names of the hooked modules. One way to inspect the names is to look at the source code in Huggingface (e.g. for GPT2 and GPT-J)

(2) Data generation. Generating a set of input queries where the model's next token prediction matches the correct attribute. This in order to make sure that the analysis focuses on predictions that involve an attribute extraction. ""

In order to adaptat to our indirect KBICR dataset we only need to perform the second step: 

`create_json.py` script to generate a dictionary with subject, attributes, template and prediction fields, following `known_1000.json` used by the authors' code.


