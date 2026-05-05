# Zero‑Shot Prompting

### 1. Definition

*   **Zero‑shot prompting** is a technique where an LLM performs a task **using only instructions, without any examples**.
*   The model relies entirely on:
    *   Pre‑training knowledge
    *   Instruction‑following capability
*   No input–output demonstrations are provided.

### 2. Key Idea

*   You **tell** the model what to do
*   You **do NOT show** how it should be done
*   The model infers the task from its training

### 3. How Zero‑Shot Prompting Works

1.  **Instruction given**
    *   e.g., “Summarise”, “Translate”, “Classify”
2.  **Task inference**
    *   Model recognises task type from training
3.  **Response generation**
    *   Uses learned patterns and language understanding

### 4. When to Use Zero‑Shot Prompting

Use zero‑shot when the task is **common, generic, or exploratory**.

#### Best‑fit Use Cases

*   **Summarisation**
    ```text
    Summarise this document in 5 bullet points.
    ```
*   **Translation**
    ```text
    Translate this paragraph to French.
    ```
*   **Simple classification**
    ```text
    Is this email Spam or Not Spam?
    ```
*   **General Q\&A**
    ```text
    Explain what data normalisation is.
    ```
*   **Brainstorming / ideation**
    ```text
    Suggest improvement ideas for data pipeline reliability.
    ```
### 5. Why Zero‑Shot Works Well

*   Models are trained on **massive, diverse datasets**
*   They already know:
    *   Common task patterns
    *   Standard output formats
*   Fast to write and iterate
*   Scales easily across tasks
Most chat‑based AI usage is zero‑shot by default

### 6. When to Avoid Zero‑Shot Prompting

Avoid zero‑shot when **accuracy, consistency, or structure matter**.

### Poor Use Cases

*   **Strict output formats**
    *   JSON schemas
    *   SQL templates
    *   Fixed column ordering

*   **Domain‑specific logic**
    *   Internal business rules
    *   Company‑specific definitions

*   **Ambiguous quality criteria**
    *   “Good”, “Optimised”, “Clean” (without examples)

*   **Repeated inconsistency**
    *   Same prompt → different outputs

*   **Complex multi‑step reasoning**
    *   Edge‑case heavy logic
    *   Conditional workflows

### 7. Zero‑Shot Failure Signals

Switch to few‑shot if you observe:

*   Output format drifting
*   Correct idea but wrong structure
*   Different results across runs
*   Excessive prompt tweaking with little improvement
