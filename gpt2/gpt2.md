# Using GPT-2

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

| Task                      |  Example prompt                | Expected answer | 
|---------------------------------|----------------------------|-----------------|
| country-capital-person corespondence | Alice is from France, and she lives in Paris. Bob is from Germany, and he lives in Berlin. John is from the USA, and he lives in | Washington | 
<!-- | author-book corespondence | Mark Twain wrote The Adventures of Tom Sawyer. Jane Austen wrote Pride and Prejudice. George Orwell wrote 1984. Herman Melville wrote Moby-Dick. J.K. Rowling wrote Harry Potter and the Philosopher's Stone. Mary Shelley wrote | Frankenstien | 
| inventor-invention corespondence | Thomas Edison invented the light bulb. Alexander Graham Bell invented the telephone. Wright brothers invented the airplane. Nikola Tesla invented the alternating current. Marie Curie discovered radium. Alan Turing invented the | Turing Machine | 
| actor-movie corespondence | Tom Hanks acted in Forrest Gump. Leonardo DiCaprio acted in Titanic.Meryl Streep acted in The Devil Wears Prada. Brad Pitt acted in Fight Club. Angelina Jolie acted in Mr. & Mrs. Smith. Johnny Depp acted in Pirates of the Caribbean. Jennifer Lawrence acted in | The Hunger Games  |
| CEO-company corespondence | Tim Cook is the CEO of Apple. Satya Nadella is the CEO of Microsoft. Sundar Pichai is the CEO of Google. Mark Zuckerberg is the CEO of Facebook. Elon Musk is the CEO of Tesla. Jeff Bezos is the CEO of | Amazon |    
| planet-position relative to the sun corespondence | Jupiter is the 5th planet from the Sun. Saturn is the 6th planet from the Sun. Uranus is the 7th planet from the Sun. Neptune is the 8th planet from the Sun.Mercury is the 1st planet from the Sun. Venus is | the 2nd planet from the Sun | -->

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

3. Perform an extensive and iterative series of patching experiments with the goal of removing
as many unnecessary components and connections from the model as possible.




