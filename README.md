# Automatic Circuit Discovery for Knowledge-Based In-Context Retrieval

GPT-2 small circuit for our task:

![](Automatic-Circuit-Discovery/acdc/ims_hybridretrieval_kl/img_new_100.png)

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

## Prompting

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

| Task                      |  Correct prompt                | Expected answer | Corrupted prompt | 
|---------------------------------|----------------------------|-----------------|----------------|
| Knowledge-Based In-Context Retrieval | Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in USA, Washington - | Peter | Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in Italy, Paris - Sara |  


| batch | type      | question                                                                                             | answer | complete                                                                                             |
|-------|-----------|------------------------------------------------------------------------------------------------------|--------|------------------------------------------------------------------------------------------------------|
| 1     | correct   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - John" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Paris - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Paris - John" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Berlin - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Berlin - John" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Madrid - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Madrid - John" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Chicago - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Chicago - John" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Istanbul - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Istanbul - John" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - " | Alice  | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - Alice" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - " | Bob    | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - Bob" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - " | Peter  | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - Peter" |
| 1     | corrupted | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in Spain, Washington - " | John   | "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in Spain, Washington - John" |
| 2     | correct   | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - Bob" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Ankara - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Ankara - Bob" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Rome - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Rome - Bob" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Berlin - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Berlin - Bob" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Marseille - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Marseille - Bob" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Toronto - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Toronto - Bob" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - " | Peter  | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - Peter" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - " | Alice  | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - Alice" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - " | John   | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - John" |
| 2     | corrupted | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in Portugal, Paris - " | Bob    | "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in Portugal, Paris - Bob" |


Prompt structure: 

[P1] [Verb] [C1], [K1] - [P1]... until [P3]

The query: [K3] - & Answer: [P3]

The scope of this experiment is to find a the circuit of components inside the model that points towards a typical behavior that happens when prompted the same type of prompt that we described earlier. A method that automates circuit finding in LLMs is the Automatic Circuit Discovery [ACDC](https://arxiv.org/pdf/2304.14997). They propose a three step workflow for this:

### 1. Observe a behavior 

or task that a neural network displays, create a dataset that reproduces
the behavior in question, and choose a metric to measure the extent to which the model
performs the task. 

![Examples of prompts and task used in the paper](image.png)

### 2. Define the scope of the circuit interpretation, 

i.e. decide to what level of granularity (e.g. attention
heads and MLP layers, individual neurons, whether these are split by token position) at which
one wants to analyze the network. This results in a computational graph of interconnected
model units.

### 3. Perform an extensive and iterative series of patching experiments, 

with the goal of removing as many unnecessary components and connections from the model as possible.




