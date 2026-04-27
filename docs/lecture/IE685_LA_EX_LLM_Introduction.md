# Exercise 1: Introduction to Large Language Models (FSS 2026)

**Course:** IE 685 - Large Language Models and Agents  
**University of Mannheim** | [Course Website](https://www.uni-mannheim.de/dws/teaching/course-details/courses-for-master-candidates/ie-685-large-language-models-and-agents/)

---

**Goal:** This exercise provides a hands-on introduction to the building blocks of Large Language Models (LLMs). We will explore:

1. **Tokenization** — How LLMs see text, comparing different tokenizers
2. **Token Cost Estimation** — Predicting API costs before running large batch jobs
3. **Embeddings** — Visualizing how tokens become meaningful vectors
4. **Inference** — Sending prompts to a small local LLM for sentiment analysis
5. **Fine-Tuning with LoRA** — Improving a model's performance on a specific task

**Environment:** This notebook is designed to run on **Google Colab** with a free **T4 GPU** runtime.

---

## 0. Setup & Installation

Let's start by installing all necessary packages. Make sure your Colab runtime is set to **T4 GPU**:
`Runtime → Change runtime type → T4 GPU`

### Hugging Face Authentication (Required)

The Gemma 3 model is a **gated model** on Hugging Face. Before running this notebook, you need to:

1. **Create a Hugging Face account** (if you don't have one): [huggingface.co/join](https://huggingface.co/join)
2. **Accept the Gemma 3 license**: Visit the model page at [huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it) and click **"Agree and access repository"**. This is required by Google for all Gemma models.
3. **Create an access token**: Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new token (the free `read` scope is sufficient)

You will be prompted to enter your token below.

### Install Required Packages

```python
# Install packages
# We use 'uv' for faster, more reliable dependency resolution (recommended by Unsloth)

!pip install --upgrade pip && pip install uv

# Unsloth — handles torch, transformers, peft, trl, and all training dependencies
!uv pip install unsloth --system

# Tokenization (OpenAI's tiktoken)
!uv pip install tiktoken --system

# Embeddings visualization
!uv pip install scikit-learn matplotlib sentence-transformers --system

# Dataset loading
!uv pip install datasets --system

!uv pip install huggingface_hub
```

### Log in to Hugging Face

```python
# Log in to Hugging Face
# This will prompt you to enter your access token.
# Paste the token you created at https://huggingface.co/settings/tokens

from huggingface_hub import login
login()
```

---

# Part 1: Tokenization — How LLMs See Text

Before an LLM can process text, it must be converted into **tokens** — numerical representations that the model understands. Different models use different tokenizers, which means the same text can be split into different tokens depending on the model.

### Why does this matter?

- Tokenization affects **model performance** (how well the model understands your input)
- It determines **cost** (API pricing is per-token)
- It defines **context limits** (max tokens, not max characters)
- Different languages and scripts are tokenized very differently

### 1.1 Tokenizer Comparison: GPT-2 vs. GPT-4o vs. Gemma

We'll compare three tokenizers across different eras:
- **tiktoken** (`r50k_base`): Used by OpenAI's GPT-2 (2019) — only ~50k vocabulary
- **tiktoken** (`o200k_base`): Used by OpenAI's GPT-4o family — ~200k vocabulary
- **Hugging Face Tokenizer** (`google/gemma-3-270m-it`): Used by Google's Gemma 3 model — ~262k vocabulary

All use variants of **Byte Pair Encoding (BPE)**, but their vocabularies and merge rules differ significantly. Older tokenizers like GPT-2's were trained mostly on English text with a small vocabulary, which means non-English languages get split into many more tokens.

```python
import tiktoken
from transformers import AutoTokenizer

# Load tokenizers
enc_gpt2  = tiktoken.encoding_for_model("gpt-2")        # OpenAI's old GPT-2 tokenizer (~2019)
enc_gpt4o = tiktoken.encoding_for_model("gpt-4o")       # OpenAI's latest tokenizer
enc_gemma = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")  # Google's tokenizer

print(f"GPT-2 tokenizer vocabulary size:   {enc_gpt2.n_vocab:,}")
print(f"GPT-4o tokenizer vocabulary size:  {enc_gpt4o.n_vocab:,}")
print(f"Gemma 3 tokenizer vocabulary size: {enc_gemma.vocab_size:,}")
```

```python
# Let's tokenize some example sentences and compare
import unicodedata

def display_width(s):
    """Calculate display width, accounting for full-width CJK/emoji characters."""
    return sum(2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1 for ch in s)

def ljust_display(s, width):
    """Left-justify a string based on its actual display width."""
    return s + ' ' * max(0, width - display_width(s))

examples = [
    "The quick brown fox jumps over the lazy dog.",
    "Large Language Models are revolutionizing artificial intelligence.",
    "Die Universität Mannheim ist eine hervorragende Universität.",
    "東京は日本の首都です。",  # "Tokyo is the capital of Japan" in Japanese
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "🎓📚🤖💡",  # Emojis
]

col_width = 75
print(f"{'Text':<{col_width}} {'GPT-2':>8} {'GPT-4o':>8} {'Gemma':>8}")
print("-" * (col_width + 27))

for text in examples:
    gpt2_tokens  = enc_gpt2.encode(text)
    gpt4o_tokens = enc_gpt4o.encode(text)
    gemma_tokens = enc_gemma.encode(text)
    display_text = text[:72] + "..." if len(text) > 72 else text
    print(f"{ljust_display(display_text, col_width)} {len(gpt2_tokens):>8} {len(gpt4o_tokens):>8} {len(gemma_tokens):>8}")
```

**Observation:** Notice how GPT-2 (the oldest tokenizer with ~50k vocab) consistently uses the most tokens, especially for non-English text like Japanese and emoji. The newer tokenizers (GPT-4o and Gemma) have much larger vocabularies and handle multilingual content far more efficiently. This is a direct consequence of vocabulary size and training data diversity.

```python
# Let's look at the actual token splits for a specific example
text = "Large Language Models are revolutionizing artificial intelligence."

# GPT-4o tokens
gpt4o_token_ids = enc_gpt4o.encode(text)
gpt4o_token_strings = [enc_gpt4o.decode([t]) for t in gpt4o_token_ids]

# Gemma tokens
gemma_token_ids = enc_gemma.encode(text)
gemma_token_strings = enc_gemma.convert_ids_to_tokens(gemma_token_ids)

print("GPT-4o tokenization:")
print(f"  Token IDs:     {gpt4o_token_ids}")
print(f"  Token strings: {gpt4o_token_strings}")
print(f"  Num tokens:    {len(gpt4o_token_ids)}")
print()
print("Gemma 3 tokenization:")
print(f"  Token IDs:     {gemma_token_ids}")
print(f"  Token strings: {gemma_token_strings}")
print(f"  Num tokens:    {len(gemma_token_ids)}")
```

### 1.2 Interactive Tokenizer Visualization

To get a more intuitive feel for tokenization, explore these interactive tools:

- **OpenAI Tokenizer:** [platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) — Visualizes how GPT models tokenize text, with color-coded tokens
- **Tiktokenizer:** [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app) — Compare tokenization across different model encodings side by side

**Try it out!** Paste the German and Japanese examples from above into the OpenAI tokenizer and observe how non-English text gets split into more (and sometimes surprising) tokens.

### 1.3 The Impact of Tokenization on Different Languages

Tokenization has real-world implications: the same semantic content costs more tokens (and therefore more money and context window) in some languages than others. Let's quantify this.

```python
# The same sentence in multiple languages
translations = {
    "English":    "The weather is beautiful today.",
    "German":     "Das Wetter ist heute wunderschön.",
    "French":     "Le temps est magnifique aujourd'hui.",
    "Spanish":    "El tiempo es hermoso hoy.",
    "Chinese":    "今天天气很好。",
    "Japanese":   "今日はいい天気です。",
    "Arabic":     "الطقس جميل اليوم.",
    "Korean":     "오늘 날씨가 아름답습니다.",
}

print(f"{'Language':<12} {'Chars':>5}   {'GPT-2':>6} {'GPT-4o':>7} {'Gemma':>6}   {'Ch/Tok GPT-2':>14} {'Ch/Tok GPT-4o':>15}")
print("-" * 75)

for lang, text in translations.items():
    n_chars = len(text)
    n_gpt2  = len(enc_gpt2.encode(text))
    n_gpt4o = len(enc_gpt4o.encode(text))
    n_gemma = len(enc_gemma.encode(text))
    ratio_gpt2  = n_chars / n_gpt2  if n_gpt2  > 0 else 0
    ratio_gpt4o = n_chars / n_gpt4o if n_gpt4o > 0 else 0
    print(f"{lang:<12} {n_chars:>5}   {n_gpt2:>6} {n_gpt4o:>7} {n_gemma:>6}   {ratio_gpt2:>14.1f} {ratio_gpt4o:>15.1f}")
```

**Discussion:** Compare the Chars/Token ratios between GPT-2 and GPT-4o. English is fairly efficient for both, but look at CJK languages (Chinese, Japanese, Korean) and Arabic — GPT-2 needs far more tokens for the same content. This is because GPT-2's small vocabulary (~50k) was trained primarily on English web text, so it has to fall back to byte-level encoding for unfamiliar scripts. Modern tokenizers with larger vocabularies (200k+) include dedicated tokens for many languages, making them much more efficient and cost-effective for multilingual use.

---

# Part 2: Token Cost Estimation

When working with LLM APIs at scale (e.g., processing thousands of documents), it's crucial to estimate costs **before** submitting jobs. The `tiktoken` library lets us count tokens locally without making any API calls.

### Example: Estimating the cost of classifying 10,000 product reviews

```python
import tiktoken

# Initialize the encoder once (reuse across all calls for efficiency)
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

# Simulated batch of product reviews
sample_reviews = [
    "This product is amazing! Best purchase I've made this year. Highly recommend to anyone looking for quality.",
    "Terrible quality. Broke after two days of use. Complete waste of money. Would not recommend.",
    "It's okay, nothing special. Does what it says but the build quality could be better for the price.",
    "Absolutely love it! The design is sleek, performance is top-notch, and customer service was excellent.",
    "Not worth the money. The product looks nothing like the pictures. Very disappointed with my purchase.",
]

# System prompt for classification (sent with every request)
system_prompt = """You are a sentiment classifier. Classify the following product review as 
'positive', 'negative', or 'neutral'. Respond with only the label."""

# Count tokens for the system prompt (sent once per request)
system_tokens = len(encoder.encode(system_prompt))
print(f"System prompt tokens: {system_tokens}")

# Count tokens for each review
review_tokens = [len(encoder.encode(review)) for review in sample_reviews]
avg_review_tokens = sum(review_tokens) / len(review_tokens)
print(f"Average review tokens: {avg_review_tokens:.0f}")

# Estimate output tokens (classification label ≈ 1-2 tokens)
avg_output_tokens = 2
```

```python
# Scale up to 10,000 reviews
n_reviews = 10_000

total_input_tokens = n_reviews * (system_tokens + avg_review_tokens)
total_output_tokens = n_reviews * avg_output_tokens

# Pricing as of early 2026 (check https://openai.com/api/pricing/ for current rates)
# GPT-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
price_input_per_1m = 0.15
price_output_per_1m = 0.60

cost_input = (total_input_tokens / 1_000_000) * price_input_per_1m
cost_output = (total_output_tokens / 1_000_000) * price_output_per_1m
total_cost = cost_input + cost_output

print(f"=== Cost Estimation for {n_reviews:,} Reviews ===")
print(f"Total input tokens:  {total_input_tokens:>12,.0f}")
print(f"Total output tokens: {total_output_tokens:>12,.0f}")
print(f"")
print(f"Input cost:          ${cost_input:>10.4f}")
print(f"Output cost:         ${cost_output:>10.4f}")
print(f"Total estimated cost: ${total_cost:>9.4f}")
print(f"")
print(f"💡 Tip: At ~${total_cost:.2f} for {n_reviews:,} classifications, GPT-4o-mini")
print(f"   is very cost-effective for simple classification tasks.")
print(f"   For comparison, GPT-4o would cost ~{total_cost * (2.5/0.15):.2f}$ (input: $2.50/1M tokens).")
```

**Key Takeaway:** Always estimate costs before running large batch jobs. The difference between models can be 10-50x in cost. For simple tasks like classification, smaller and cheaper models often work just as well.

---

# Part 3: From Tokens to Vectors — Embeddings Visualization

Tokens are just numbers — they don't carry meaning by themselves. Inside an LLM, each token is mapped to a high-dimensional **embedding vector**. These vectors are learned during training and capture semantic relationships.

To visualize this clearly, we'll use a **sentence-transformer** model ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). Unlike raw LLM hidden states, sentence-transformers are explicitly trained with a contrastive objective so that semantically similar texts end up close together in the embedding space — making them ideal for visualizing how meaning is encoded as vectors.

### 3.1 Computing Word Embeddings

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load a sentence-transformer model (small and fast, ~80MB)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Words grouped by semantic category
word_groups = {
    "Animals":    ["cat", "dog", "fish", "bird", "horse", "elephant"],
    "Colors":     ["red", "blue", "green", "yellow", "purple", "orange"],
    "Numbers":    ["one", "two", "three", "four", "five", "six"],
    "Emotions":   ["happy", "sad", "angry", "excited", "scared", "calm"],
    "Countries":  ["Germany", "France", "Japan", "Brazil", "Canada", "Italy"],
    "Food":       ["pizza", "sushi", "burger", "pasta", "salad", "cake"],
}

# Compute embeddings for all words
all_words = []
all_categories = []
for category, words in word_groups.items():
    all_words.extend(words)
    all_categories.extend([category] * len(words))

all_embeddings = embedder.encode(all_words)
print(f"Computed {len(all_words)} embeddings of dimension {all_embeddings.shape[1]}")
```

### 3.2 Visualizing Embeddings with PCA

```python
# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(all_embeddings)

# Plot with distinct colors per category
fig, ax = plt.subplots(figsize=(14, 9))
colors = {
    "Animals": "#e74c3c", "Colors": "#3498db", "Numbers": "#2ecc71",
    "Emotions": "#9b59b6", "Countries": "#e67e22", "Food": "#1abc9c",
}

for category in word_groups:
    mask = [c == category for c in all_categories]
    points = embeddings_2d[mask]
    ax.scatter(points[:, 0], points[:, 1], c=colors[category], label=category,
               s=120, alpha=0.8, edgecolors="white", linewidth=0.8)

# Add word labels with slight offset to avoid overlap
for i, word in enumerate(all_words):
    ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=9, ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points')

ax.set_title("Word Embeddings (all-MiniLM-L6-v2, PCA projection)", fontsize=14)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.legend(fontsize=11, loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Discussion:** You should see clear, well-separated clusters for each semantic category. This is because sentence-transformers are trained with a **contrastive learning** objective — the model learns to pull similar meanings together and push different meanings apart in the embedding space. This is also the core mechanism behind **semantic search** and **Retrieval-Augmented Generation (RAG)**: by comparing embedding vectors, we can find the most semantically relevant passages for a given query.

Note: Inside a generative LLM like Gemma, the internal representations also capture semantic structure, but they are optimized for *next-token prediction* rather than *similarity*, so they don't cluster as cleanly when projected to 2D.

### 3.3 Embedding Arithmetic — Doing Math with Meaning

One of the most fascinating properties of embedding spaces is that **semantic relationships are encoded as directions**. This means we can do vector arithmetic with words and get meaningful results!

The classic example: `king - man + woman ≈ queen`

The intuition: subtracting `man` from `king` removes the "maleness" direction, leaving behind the concept of "royalty". Adding `woman` then gives us the female version of royalty — `queen`. Let's try some fun ones with our embedder.

```python
from sklearn.metrics.pairwise import cosine_similarity

def embed(word):
    """Get the embedding for a single word."""
    return embedder.encode([word])[0]

def analogy(a, b, c, candidates, top_k=5):
    """Solve: a - b + c ≈ ?
    Excludes the input words (a, b, c) from the candidate pool to avoid trivial matches."""
    result_vector = embed(a) - embed(b) + embed(c)
    
    # Filter out the input words so they can't be returned as answers
    filtered = [w for w in candidates if w.lower() not in {a.lower(), b.lower(), c.lower()}]
    
    # Embed filtered candidates and compute similarity
    candidate_embeddings = embedder.encode(filtered)
    similarities = cosine_similarity([result_vector], candidate_embeddings)[0]
    
    # Sort by similarity (descending)
    ranked = sorted(zip(filtered, similarities), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# Build a large pool of candidate words the model can "pick" from
candidate_pool = [
    # Countries & capitals
    "Germany", "France", "Japan", "Brazil", "Italy", "Canada", "USA", "China", "India", "Spain",
    "Berlin", "Paris", "Tokyo", "Rome", "Madrid", "London", "Beijing", "Ottawa", "Brasilia", "Delhi",
    # Food & drink
    "pizza", "sushi", "burger", "pasta", "croissant", "bratwurst", "taco", "curry", "kimchi", "paella",
    "beer", "wine", "sake", "tea", "coffee", "espresso", "champagne",
    # Animals
    "cat", "dog", "kitten", "puppy", "fish", "bird", "horse", "elephant",
    # People & royalty
    "king", "queen", "man", "woman", "boy", "girl", "prince", "princess",
    # Concepts
    "hot", "cold", "fast", "slow", "big", "small", "happy", "sad",
]

# === Fun analogies ===
print("=" * 65)
print("  EMBEDDING ARITHMETIC: a - b + c ≈ ?")
print("=" * 65)

analogies = [
    ("Tokyo",      "Japan",   "France",   "Japan is to Tokyo as France is to..."),
    ("sushi",      "Japan",   "Italy",    "Japan is to sushi as Italy is to..."),
    ("king",       "man",     "woman",    "man is to king as woman is to..."),
    ("kitten",     "cat",     "dog",      "cat is to kitten as dog is to..."),
    ("Berlin",     "Germany", "Spain",    "Germany is to Berlin as Spain is to..."),
]

for a, b, c, description in analogies:
    results = analogy(a, b, c, candidate_pool, top_k=3)
    top_word, top_score = results[0]
    runners_up = ", ".join([f"{w} ({s:.2f})" for w, s in results[1:]])
    
    print(f"\n  {description}")
    print(f"  {a} - {b} + {c} = {top_word}  (score: {top_score:.2f})")
    print(f"  Runner-ups: {runners_up}")
```

**Takeaway:** The fact that simple vector subtraction and addition can capture relationships like "country → capital" or "animal → baby animal" shows that embedding spaces aren't just bags of numbers — they encode **structured knowledge** as geometric directions. This is why embeddings are so powerful for search, recommendation, and as the foundation layer inside every LLM.

Not every analogy will work perfectly (try a few of your own!), but the fact that it works at all from just reading text is remarkable.

---

# Part 4: Sentiment Analysis with a Small LLM (Zero-Shot)

Now let's use a real (small) LLM for a practical task: **sentiment analysis**. We'll use Google's **Gemma 3 270M** model — a model with only 270 million parameters that can run easily on a free Colab T4 GPU.

First, we'll see how the model performs **without any fine-tuning** (zero-shot), then we'll improve it with LoRA fine-tuning.

### 4.1 Load the Model

```python
from unsloth import FastModel
import torch

# Load the Gemma 3 270M instruction-tuned model
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=2048,
    load_in_4bit=False,     # Model is small enough to load in full precision
    load_in_8bit=False,
    full_finetuning=False,
)

print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```python
# IMPORTANT: Enable Unsloth's optimized inference mode
# Without this, generation uses the default HuggingFace path which is ~2x slower
FastModel.for_inference(model)
```

### 4.2 Define Generation Function

```python
def generate_response(model, tokenizer, prompt, max_new_tokens=32):
    """Generate a response from the model given a prompt."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
        )
    
    # Decode only the generated tokens (not the prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()
```

### 4.3 Zero-Shot Evaluation

```python
# Test reviews with known sentiment
test_reviews = [
    ("This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.", "positive"),
    ("Terrible film. Waste of time and money. The script was awful and the acting was wooden.", "negative"),
    ("A masterpiece of modern cinema. Every scene was beautifully crafted.", "positive"),
    ("I couldn't even finish watching it. Boring, predictable, and poorly directed.", "negative"),
    ("One of the best films I've seen this year. Highly recommend!", "positive"),
    ("The worst movie I have ever seen. Absolutely dreadful in every way.", "negative"),
    ("Brilliant performances by the entire cast. A truly moving experience.", "positive"),
    ("Save your money. This movie is an insult to the audience's intelligence.", "negative"),
]

print("=== Zero-Shot Sentiment Analysis (Before Fine-Tuning) ===")
print()

correct = 0
for review, true_label in test_reviews:
    prompt = f"""Classify the sentiment of the following movie review as 'positive' or 'negative'. 
Respond with only one word: positive or negative.

Review: {review}

Sentiment:"""
    
    response = generate_response(model, tokenizer, prompt, max_new_tokens=10)
    
    # Check if the response contains the correct label
    predicted = "positive" if "positive" in response.lower() else "negative" if "negative" in response.lower() else "unclear"
    is_correct = predicted == true_label
    correct += int(is_correct)
    
    status = "✓" if is_correct else "✗"
    print(f"  {status} True: {true_label:<8} | Predicted: {predicted:<8} | Raw output: '{response[:60]}'")
    print(f"    Review: {review[:80]}...")
    print()

print(f"Zero-shot accuracy: {correct}/{len(test_reviews)} ({correct/len(test_reviews):.0%})")
```

**Observation:** The 270M parameter model is very small and likely struggles with this task in a zero-shot setting. It may not follow instructions well, produce inconsistent labels, or simply not understand the task. This is exactly what we'll fix with fine-tuning!

---

# Part 5: Fine-Tuning with LoRA

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique. Instead of updating all model weights (which would require enormous compute), LoRA:

1. **Freezes** the original model weights
2. **Adds** small trainable matrices (adapters) to key layers
3. **Trains** only these adapters (typically <1% of total parameters)

This makes fine-tuning feasible even on a free Colab T4 GPU!

### 5.1 Prepare the LoRA Adapters

```python
# Add LoRA adapters to the model
model = FastModel.get_peft_model(
    model,
    r=16,                   # LoRA rank — higher = more capacity, but slower
    lora_alpha=16,          # Scaling factor (typically set equal to r)
    lora_dropout=0,         # No dropout for small models
    bias="none",
    random_state=42,
    use_gradient_checkpointing="unsloth",  # Memory optimization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Show how few parameters we're actually training
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
```

### 5.2 Prepare the Training Data (IMDB Dataset)

We'll use the classic IMDB movie review dataset — 50,000 reviews labeled as positive or negative. For efficiency, we'll use a subset for training.

```python
from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("imdb")

print(f"Training samples: {len(dataset['train']):,}")
print(f"Test samples:     {len(dataset['test']):,}")
print(f"Labels: 0 = negative, 1 = positive")
print(f"\nExample review (first 200 chars):")
print(f"  Text:  '{dataset['train'][0]['text'][:200]}...'")
print(f"  Label: {dataset['train'][0]['label']} ({'positive' if dataset['train'][0]['label'] == 1 else 'negative'})")
```

```python
# Format the dataset for instruction fine-tuning
# We convert each review into a chat-format conversation

label_map = {0: "negative", 1: "positive"}

def format_for_training(example):
    """Convert an IMDB example into a chat-format training example."""
    # Truncate very long reviews to keep training efficient
    review_text = example["text"][:512]
    sentiment = label_map[example["label"]]
    
    messages = [
        {"role": "user", "content": f"Classify the sentiment of this movie review as 'positive' or 'negative'.\n\nReview: {review_text}"},
        {"role": "assistant", "content": sentiment},
    ]
    
    # Apply the chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# Use a subset for faster training (2,000 samples is plenty for this small model)
train_subset = dataset["train"].shuffle(seed=42).select(range(2000))
train_formatted = train_subset.map(format_for_training)

print(f"Training examples prepared: {len(train_formatted)}")
print(f"\nExample formatted text (first 300 chars):")
print(train_formatted[0]["text"][:300])
```

### 5.3 Train the Model

We use the `SFTTrainer` (Supervised Fine-Tuning Trainer) from the TRL library, which handles the training loop, logging, and optimization for us.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_formatted,
    args=SFTConfig(
        output_dir="./outputs",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=20,
        logging_steps=25,
        save_strategy="no",       # Don't save checkpoints to save disk space
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",       # Memory-efficient optimizer
        seed=42,
        max_seq_length=512,
        dataset_text_field="text",
        packing=True,             # Pack multiple short examples into one sequence
    ),
)
```

```python
# Track GPU memory usage
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_mem / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}")
print(f"Max memory: {max_memory} GB")
print(f"Memory reserved before training: {start_gpu_memory} GB")
print(f"\nStarting training...")

# Train!
trainer_stats = trainer.train()

# Report memory and time
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"\n=== Training Complete ===")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"Peak GPU memory: {used_memory} GB / {max_memory} GB ({used_memory/max_memory:.0%})")
```

### 5.4 Evaluate Before and After Fine-Tuning

```python
# Re-enable optimized inference after training
FastModel.for_inference(model)

# Evaluate both the stock (zero-shot) model and the fine-tuned model on the IMDB test set
test_subset = dataset["test"].shuffle(seed=42).select(range(10))

def evaluate_model(model, tokenizer, test_data, label_map):
    """Evaluate sentiment classification accuracy on a test set."""
    correct = 0
    results = []
    
    for i, example in enumerate(test_data):
        review = example["text"][:512]
        true_label = label_map[example["label"]]
        
        prompt = f"Classify the sentiment of this movie review as 'positive' or 'negative'.\n\nReview: {review}"
        response = generate_response(model, tokenizer, prompt, max_new_tokens=10)
        
        predicted = "positive" if "positive" in response.lower() else "negative" if "negative" in response.lower() else "unclear"
        is_correct = predicted == true_label
        correct += int(is_correct)
        results.append({"true": true_label, "predicted": predicted, "correct": is_correct})
        
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(test_data)} examples... (running accuracy: {correct/(i+1):.1%})")
    
    accuracy = correct / len(test_data)
    return accuracy, results

# --- Evaluate zero-shot (base model without LoRA) ---
print("=== Evaluating ZERO-SHOT model (LoRA disabled) ===")
model.disable_adapter_layers()
zeroshot_accuracy, zeroshot_results = evaluate_model(model, tokenizer, test_subset, label_map)
print(f"Zero-shot accuracy: {zeroshot_accuracy:.1%}\n")

# --- Evaluate fine-tuned (with LoRA) ---
print("=== Evaluating FINE-TUNED model (LoRA enabled) ===")
model.enable_adapter_layers()
finetuned_accuracy, finetuned_results = evaluate_model(model, tokenizer, test_subset, label_map)
print(f"Fine-tuned accuracy: {finetuned_accuracy:.1%}\n")

# --- Summary ---
print("=" * 50)
print(f"  Zero-shot accuracy:  {zeroshot_accuracy:.1%}")
print(f"  Fine-tuned accuracy: {finetuned_accuracy:.1%}")
print(f"  Improvement:         {finetuned_accuracy - zeroshot_accuracy:+.1%}")
print("=" * 50)
```

```python
# Let's also re-run our original hand-picked examples to see the qualitative difference
print("=== After Fine-Tuning: Same Test Reviews ===")
print()

correct = 0
for review, true_label in test_reviews:
    prompt = f"Classify the sentiment of this movie review as 'positive' or 'negative'.\n\nReview: {review}"
    response = generate_response(model, tokenizer, prompt, max_new_tokens=10)
    
    predicted = "positive" if "positive" in response.lower() else "negative" if "negative" in response.lower() else "unclear"
    is_correct = predicted == true_label
    correct += int(is_correct)
    
    status = "✓" if is_correct else "✗"
    print(f"  {status} True: {true_label:<8} | Predicted: {predicted:<8} | Raw output: '{response[:60]}'")
    print(f"    Review: {review[:80]}...")
    print()

print(f"Fine-tuned accuracy on hand-picked examples: {correct}/{len(test_reviews)} ({correct/len(test_reviews):.0%})")
```

### 5.5 Results Summary

Let's visualize the improvement.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    ["Zero-Shot\n(Before Fine-Tuning)", "LoRA Fine-Tuned\n(After Training)"],
    [zeroshot_accuracy * 100, finetuned_accuracy * 100],
    color=["#e74c3c", "#2ecc71"],
    width=0.5,
    edgecolor="black",
    linewidth=1.2,
)

# Add value labels on bars
for bar, acc in zip(bars, [zeroshot_accuracy, finetuned_accuracy]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{acc:.0%}", ha='center', va='bottom', fontsize=16, fontweight='bold')

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Sentiment Analysis: Before vs. After LoRA Fine-Tuning\n(Gemma 3 270M on IMDB — 10 test examples)", fontsize=14)
ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

# Summary & Key Takeaways

In this exercise, we covered the core building blocks of working with LLMs:

1. **Tokenization** — Different models use different tokenizers, leading to different token counts and costs. Non-English languages are often less efficiently tokenized.

2. **Cost Estimation** — Libraries like `tiktoken` let you estimate API costs locally before running expensive batch jobs.

3. **Embeddings** — Tokens are mapped to high-dimensional vectors that capture semantic meaning. Similar words cluster together in embedding space.

4. **Inference** — Even very small models (270M parameters) can be used for tasks like sentiment analysis, though their zero-shot performance may be limited.

5. **LoRA Fine-Tuning** — By training only ~1% of parameters with LoRA, we can dramatically improve a model's performance on a specific task, all on a free Colab GPU in just a few minutes.

### Further Reading

- Karpathy's "Let's build the GPT Tokenizer" — [YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) | [minbpe repo](https://github.com/karpathy/minbpe)
- Hugging Face Tokenizer Summary — [Documentation](https://huggingface.co/docs/transformers/tokenizer_summary)
- LoRA Paper — [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Unsloth — [unsloth.ai](https://unsloth.ai/) | [Documentation](https://unsloth.ai/docs/models/gemma-3-how-to-run-and-fine-tune)
