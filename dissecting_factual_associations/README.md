The notebook is taken from this [repository](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions) from Google Research. It inspects information flow as a smaller circuit that performs a 3 step attribute recall process: 

1. Subject Enrichment in MLPs with Vocabulary Projection and tested by Sublayer Knockout  

2. Relation Propagation to the last token via Attention Knockout

3. Attribute Extraction with Attention Heads

The code needs configuration for supporting GPT2-Small but it goes halfway through without changing anything. We tested for the indirect KBICR example with Alice lives in France. Images and statistics are shown for attention knockout and attribute extraction with MHSA throughout layers.

