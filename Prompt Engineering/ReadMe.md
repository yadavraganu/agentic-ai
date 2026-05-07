## What is Zero-Shot Prompting?

**Zero-Shot Prompting** is a technique where you ask an AI to perform a task without providing any prior examples or specific training data within the prompt. You are essentially relying on the model's pre-existing knowledge and "reasoning" capabilities to understand the instructions and generate the correct output immediately.

In a zero-shot scenario, the prompt consists only of the instruction and the input data.

### When to Use It
Zero-shot is the "default" way most people interact with AI. It is most effective in the following scenarios:

*   **Common Knowledge Tasks:** For standard questions, creative writing, or summarizing well-known topics where the model already has vast internal data.
*   **Simple Sentiment Analysis:** When you need to classify text as "Positive" or "Negative" without needing a specific brand-toned nuance.
*   **Standard Translations:** Translating common phrases between major languages.
*   **Rapid Prototyping:** When you want to see if a model can handle a task "out of the box" before spending time crafting complex examples.
*   **Generic Formatting:** Asking for a list, a table, or a brief summary of a provided text.

### When to Avoid It
Zero-shot prompting often hits a ceiling when the task requires high precision or a very specific "vibe." Avoid it when:

*   **Niche or Proprietary Domains:** If you are working with specialized industry jargon, internal company logic, or highly technical data structures that aren't common in public datasets.
*   **Strict Output Requirements:** When the output *must* follow a very specific schema, tone, or length that is difficult to describe with instructions alone.
*   **Complex Reasoning:** If the task requires multiple logical steps. In these cases, **Chain-of-Thought** or **Few-Shot Prompting** (providing 3-5 examples) is significantly more reliable.
*   **High-Stakes Accuracy:** When "hallucination" is a risk. Providing examples (Few-Shot) acts as a guardrail, showing the model the boundaries of a valid answer.

## What is Few-Shot Prompting?

**Few-Shot Prompting** is a technique where you provide the model with a few high-quality examples (the "shots") of the task being performed before asking it to complete a new instance. 

By providing a pattern of **Input $\rightarrow$ Output**, you are effectively "tuning" the model’s focus within that specific conversation. This allows the AI to pick up on nuances like formatting, tone, and logic that are difficult to explain through instructions alone.

### When to Use It
Few-shot is the gold standard for achieving consistency and precision. Use it when:

*   **Consistency in Formatting:** If you need the output to strictly follow a specific JSON structure, a specific list style, or a unique shorthand.
*   **Defining a Specific Tone:** When you need the AI to mimic a brand’s voice, a specific persona, or a technical writing style that isn't the "standard" AI default.
*   **Complex Classification:** If you are categorizing data into buckets that aren't obvious (e.g., classifying support tickets based on internal company priorities rather than just "happy/sad").
*   **Handling Edge Cases:** By including examples of tricky or unusual scenarios, you teach the model how to handle exceptions correctly.
*   **Technical Transformations:** When converting one technical format to another where logic is specific, such as mapping legacy data fields to a modern schema.

### When to Avoid It
While powerful, few-shot prompting isn't always the best tool for the job. Avoid it when:

*   **Simple, Common Tasks:** If you just need a summary or a standard email, zero-shot is faster and consumes fewer "tokens" (processing power/cost).
*   **Risk of Overfitting (Bias):** If your examples are too similar, the model might start copying the *content* of the examples rather than the *pattern*. 
    * *Example:* If all your "Positive" sentiment examples mention "Great food," the model might incorrectly associate "food" with "positivity" even in non-food contexts.
*   **Token Limits:** Every example you provide uses up space in the model's "context window." For extremely long prompts or very large documents, too many examples can crowd out the actual data you need to process.
*   **Highly Creative/Diverse Tasks:** If you want the AI to be truly original or "brainstorm" outside the box, examples can actually restrict its creativity by forcing it into a specific pattern.

## What is Chain-of-Thought (CoT) Prompting?

**Chain-of-Thought (CoT) Prompting** is a technique that encourages an AI to break down a complex problem into intermediate steps before providing a final answer. Instead of jumping straight to the conclusion, the model "thinks out loud" by generating a series of logical reasoning steps.

This is often triggered by simple phrases like **"Let’s think step-by-step,"** or by providing examples (few-shot) where the reasoning process is explicitly written out.

### When to Use It
CoT is a powerhouse for tasks that require logic rather than just pattern matching. Use it for:

*   **Complex Math and Logic:** Problems where the answer depends on a sequence of calculations or "if-then" conditions.
*   **Symbolic Reasoning:** Tasks involving coin flips, seating arrangements, or navigating complex rules.
*   **Multi-Step Data Transformations:** When you need to extract data, clean it, and then perform a calculation or categorization based on the results.
*   **Debugging Code:** When you need the model to trace the execution of a script to find a logical error rather than a syntax error.
*   **Strategic Planning:** Drafting a project roadmap where Phase B depends strictly on the outcome of Phase A.

### When to Avoid It
While powerful, CoT isn't always necessary and can sometimes be counterproductive:

*   **Simple Retrieval:** If you just need a fact (e.g., "Who wrote *The Great Gatsby*?"), CoT adds unnecessary latency and cost.
*   **Creative Writing:** Asking for a poem or a story "step-by-step" can stifle the creative flow and result in a more mechanical, less "human" output.
*   **High-Volume, Low-Latency Tasks:** CoT significantly increases the number of tokens generated, which makes responses slower and more expensive in an API environment.
*   **Basic Classification:** If you are just tagging an email as "Spam" or "Not Spam," the model doesn't need to explain the history of the internet to get it right.
*   **When the Model is Too Small:** Smaller, "lighter" models often struggle with CoT; they might hallucinate a logical path that sounds right but leads to a wrong conclusion.
