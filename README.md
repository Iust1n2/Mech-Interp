# Automatic Circuit Discovery for Knowledge-Based In-Context Retrieval

GPT-2 small circuit for our task:

![](assets/img_new_71.png) 

## Using GPT-2

[HuggingFace docs](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#openai-gpt2) 

- With `AutoModelForCausalLM` we need to declare `attention_mask` and `input_ids` as: 

```
encoded_input = tokenizer(prompt, return_tensors="pt")
input_ids = encoded_input.input_ids
print(f"Length of tokens: {len(input_ids[0])}")
attention_mask = encoded_input.attention_mask

gen_tokens = model1.generate(
    input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.1,
    max_length=input_ids.size(1) + 3,
)
```

- With `GPT2Model` and `GPT2LMHeadModel` we can only get the features, it doesn't do generation

- With `pipeline` the generator is not so customizable, but works

Notebooks for inference are inside the `gpt2` subdirectory.

## Prompt Dataset

In order to inspect both the factual recall retrieval with FFNs and in-context retrieval we must create a dual setting that simulates at the same time an abstractive and extractive task. 

Example 1: 
```
Alice is from France. The capital of France is Paris.
Bob is from Germany. The capital of Germany is Berlin.
John is from the USA. The capital of the USA is 
```

Example 2. 
```
Alice is from France, and she lives in Paris.
Bob is from Germany, and he lives in Berlin.
John is from the USA, and he lives in
```

In both examples, 1 appearing easier than 2, because it explicitly mentions the task in the context as being country-capital and 2 being inexplicitly referring to country-capital but rather a place of residence that corresponds to the capital, for the in context extractive part we repeat the task in the prompt for 2 times and for the 3-rd time the LLMs needs to predict the correct capital by following a previous pattern and for the factual recall from memory part the model needs to find the correspondence between country a capital, an apparently trivial task which even GPT-small can manage. However, the hard part is figuring out the order, if there is one, of this subtle subprocesses.

After laying out the prompt setting, we need to methodically create a dataset of the same type of prompts, a collection of prompts that requires of the model, hopefully, to repeat the same process. 

| Task                      |  Correct prompt                | Expected answer | Corrupted prompt |  Dataset | 
|---------------------------------|----------------------------|-----------------|----------------|----------|
| (Indirect) Knowledge-Based In-Context Retrieval | Prompt 1: Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in USA, Washington - & Prompt 2: Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Spain, Madrid - | Peter, Bob | first corruption is different country outside the prompt: Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in Spain, Washington - Peter & second corruption is repeated capital from the prompt: Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Spain, Ankara - Bob | Dataset 1 |
| (Direct) Knowledge-Based In-Context Retrieval | Alice lives in France, Alice - Paris, John lives in Germany, John - Berlin, Peter lives in USA, Peter - | Washington | different name outside the prompt: Alice lives in France, Alice - Paris, John lives in Germany, John - Berlin, Peter lives in USA, Michael - Washington | Dataset 1direct |
| (Indirect) Knowledge Retrieval | Rome - Italy, Madrid - Spain, Ottawa - Canada, Berlin -  | Germany |  we corrupt the middle city and we repeat the city which has a corruption to see if the model repeats the incorrect country or breaks: Rome - Italy, Bucharest - Spain, Ottawa - Canada, Bucharest -  |  Dataset 2indirect |
| (Direct) Join | Alice lives in France, Alice - France, Bob lives in Germany, Bob - Germany, John lives in USA, John - | USA | the first corruption is person outside the prompt: "Lucy lives in Italy, Lucy - Italy, Tom lives in Spain, Tom - Spain, Sara lives in Canada, Michael -  & the second corruption is person repeated from the prompt: Alice lives in France, Alice - France, Bob lives in Germany, Bob - Germany, John lives in USA, Alice - | Dataset 3direct | 
| (Direct) Knowledge Retrieval | Italy - Rome, Spain - Madrid, Canada - Ottawa, Germany -  | Berlin | the corruption is in the middle country and we repeat the country which has a corruption to see if the model repeats the incorrect city or breaks: Italy - Rome, Spain - Bucharest, Canada - Ottawa, Spain -  | Dataset 2 | 
| (Indirect) Join | Prompt 1: Alice lives in France, France - Alice, Bob lives in Germany, Germany - Bob, John lives in USA, USA - & Prompt 2: Lucy lives in Italy, Italy - Lucy, Tom lives in Spain, Spain - Tom, Sara lives in Canada, Canada - | John, Sara | first corruption is city outside the prompt: Alice lives in France, France - Alice, Bob lives in Germany, Germany - Bob, John lives in USA, Peru - John & second corruption is city repeated from the prompt: Lucy lives in Italy, Italy - Lucy, Tom lives in Spain, Spain - Tom, Sara lives in Canada, Italy - Sara | Dataset 3 | 


The scope of this experiment is to find a the circuit of components inside the model that points towards a typical behavior that happens when prompted the same type of prompt. A method that automates circuit finding in LLMs is the Automatic Circuit Discovery [ACDC](https://arxiv.org/pdf/2304.14997). They propose a three step workflow for this:

### 1. Observe a behavior 

or task that a neural network displays, create a dataset that reproduces the behavior in question, and choose a metric to measure the extent to which the model performs the task. We have chosen KL divergence for our metric.

### 2. Define the scope of the circuit interpretation, 

i.e. decide to what level of granularity (e.g. attention heads and MLP layers, individual neurons, whether these are split by token position) at which one wants to analyze the network. This results in a computational graph of interconnected model units that perform the given task.

### 3. Perform an extensive and iterative series of patching experiments, 

with the goal of removing as many unnecessary edge connections and nodes from the model as possible. 
 

## ACDC for Knowledge-Based In-Context Retrieval

Create and compare circuits for the following: 

1. indirect KBICR with indirect K + J 
2. direct KBICR with direct K + J

To run the ACDC algorithm on a task (from Prompting), three steps are required:

First, in `acdc/hybridretrieval/utils.py` in line 20 import the dataset: 
```
from acdc.hybridretrieval.hybrid_retrieval_dataset3direct import HybridRetrievalDataset
```

Then, in `acdc/main.py` in line 390 use a save path like so:
```
save_path = "acdc/hybridretrieval/acdc_results/ims_hybridretrieval_indirect_0.15"
```

And finally run the following command in the terminal: 
```
python main.py --task hybrid-retrieval --zero-ablation --threshold 0.15 --indices-mode reverse --first-cache-cpu False --second-cache-cpu False --max-num-epochs 100000 > log_kbicr_direct_0.15.txt 2>&1
```

! **Note**: Every ACDC run was done with a KL divergence threshold of 0.15. As we experimented with a smaller threshold, as the authors did in the IOI experiment, we found out that the value of the threshold is important in determining the outcome of the circuit. We ran ACDC with a KL divergence ranging from 0.7 to 0.1. The former was penalizing the model too much and the latter was not excluding as many edges as we would find it useful for post-hoc interpretation.

### Circuit recovery

In order to verify the performance of the task circuit, because our task is composed of two subtasks: Knowledge Retrieval and Join, by running ACDC again for the two subtasks we can verify if the two resulting circuits use most of the same components. 

This come as an additional phase in our experiment. We want to see if components for either or from the two smaller circuits are recovered in the bigger circuits. Algorithmically, we created a setting in which nodes fall into 7 categories and for simpliciy we labelled them as J (Join), K (Knowledge) and KJ (Join + Knowledge). So each node can be of the following: 

1. J
2. K
3. KJ
4. J & K
5. J & KJ
6. K & KJ
7. J & K & KJ

This second phase of our experiment follows this intuition:

First, we convert .gv files for each of the task and subtasks to TGF files (Trivial Graph Forma which can be read by most interactive graph softwares). Script is in `acdc/hybridretrieval/convert_gv_to_tgf.py`.

Second, we need to verify if node components inside of a .tgf is found in another file or in multiple. To do so we need to manually label them according to the previous notations. Script is in `acdc/hbyridretrieval/graph_overlaps_kj_labels.py`. We generate an equivalent TGF which instead assigns colors as labels for visualization purposes. After we generate a `combined_graph.tgf` file we use the yEd software for visualizations, which thankfully supports multiple graph layouts. That way we can generate a recovered circuit with color coded nodes that correspond to helper circuits.   