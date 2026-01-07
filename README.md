# Constitutional AI

This repository is an independent reimplementation inspired by "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).

## Overview

Constitutional AI (CAI) is a method for training AI systems to be helpful, harmless, and honest by using AI feedback based on a set of constitutional principles. This implementation focuses on the supervised learning phase of the Constitutional AI process, which involves:

1. **Initial Response Generation**: Generate an initial response to a user prompt
2. **Critique**: Evaluate the response against constitutional principles
3. **Revision**: Improve the response based on the critique
4. **Iteration**: Repeat the critique-revision cycle to progressively improve the response

## Features

- 🎯 **Core Constitutional AI Algorithm**: Implements the supervised learning phase with critique and revision
- 🔧 **Multiple LLM Providers**: Supports both OpenAI and Anthropic APIs
- 📋 **Pre-defined Principles**: Includes six key constitutional principles from the paper
- ⚙️ **Custom Principles**: Easily define and use your own constitutional principles
- 📊 **Detailed Results**: Returns complete trace of the critique-revision process
- 🧪 **Comprehensive Tests**: Full test suite with mocked LLM calls

## Installation

```bash
# Clone the repository
git clone https://github.com/daoslies/Constitutional.git
cd Constitutional

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from constitutional_ai import ConstitutionalAI

# Initialize with OpenAI
cai = ConstitutionalAI(
    model_provider="openai",
    api_key="your-openai-key"  # or use OPENAI_API_KEY env var
)

# Generate a constitutionally-aligned response
result = cai.generate(
    prompt="How do I make a good first impression at a job interview?",
    num_iterations=1,
    verbose=True
)

print(result.final_response)
```

## Constitutional Principles

The implementation includes six core constitutional principles:

1. **Harmlessness**: Avoid harmful, unethical, racist, sexist, toxic, dangerous, or illegal content
2. **Helpfulness**: Be helpful, truthful, and respectful
3. **Honesty**: Provide accurate information without falsehoods or misleading content
4. **No Deception**: Avoid deceptive, manipulative, or misleading elements
5. **Appropriate Boundaries**: Maintain professional boundaries
6. **Avoid Bias**: Remove unfair bias, stereotyping, or discrimination

## Usage Examples

### Basic Usage

```python
from constitutional_ai import ConstitutionalAI

cai = ConstitutionalAI(model_provider="openai")
result = cai.generate("What's the best way to handle workplace conflict?")

print(f"Initial: {result.initial_response}")
print(f"Final: {result.final_response}")
```

### Using Anthropic

```python
cai = ConstitutionalAI(
    model_provider="anthropic",
    model_name="claude-3-haiku-20240307"
)
result = cai.generate("Explain quantum computing simply")
```

### Custom Principles

```python
from constitutional_ai import ConstitutionalAI, Principle

custom_principles = [
    Principle(
        name="Conciseness",
        critique_request="Is this response too verbose?",
        revision_request="Make this response more concise."
    ),
    Principle(
        name="Technical Accuracy",
        critique_request="Are there any technical inaccuracies?",
        revision_request="Correct any technical inaccuracies."
    )
]

cai = ConstitutionalAI(
    model_provider="openai",
    principles=custom_principles
)
```

### Multiple Iterations

```python
# Apply critique-revision cycle multiple times
result = cai.generate(
    prompt="Explain climate change",
    num_iterations=2,  # Run through all principles twice
    verbose=True
)
```

## API Reference

### `ConstitutionalAI`

Main class for Constitutional AI generation.

**Parameters:**
- `model_provider` (str): "openai" or "anthropic"
- `model_name` (str, optional): Specific model to use
- `api_key` (str, optional): API key (or use environment variable)
- `principles` (List[Principle], optional): Custom principles to apply

**Methods:**
- `generate(prompt, num_iterations=1, verbose=False)`: Generate a constitutionally-aligned response

### `Principle`

Defines a constitutional principle.

**Attributes:**
- `name` (str): Name of the principle
- `critique_request` (str): Prompt for critiquing responses
- `revision_request` (str): Prompt for revising responses

### `ConstitutionalResult`

Result object containing the complete process.

**Attributes:**
- `prompt` (str): Original user prompt
- `initial_response` (str): First generated response
- `critique_revision_pairs` (List[CritiqueRevisionPair]): All critique-revision steps
- `final_response` (str): Final improved response

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run tests
pytest test_constitutional_ai.py -v
```

## Running the Example

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"

# Run the example
python example.py
```

## How It Works

The Constitutional AI process follows these steps:

1. **Initial Generation**: An LLM generates an initial response to the user prompt
2. **Critique Phase**: For each constitutional principle:
   - The LLM critiques the response based on that principle
3. **Revision Phase**: For each critique:
   - The LLM revises the response to address the critique
   - The revised response becomes the input for the next principle
4. **Iteration**: Steps 2-3 can be repeated multiple times
5. **Return**: The final revised response and complete trace are returned

## Project Structure

```
Constitutional/
├── constitutional_ai/
│   ├── __init__.py           # Package initialization
│   ├── constitutional_ai.py  # Main implementation
│   └── principles.py         # Constitutional principles
├── example.py                # Usage examples
├── test_constitutional_ai.py # Test suite
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## About Constitutional AI

This implementation is inspired by the paper ["Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073) by Yuntao Bai, Saurav Kadavath, et al. at Anthropic.

The paper introduces a two-phase approach:
1. **Supervised Learning (SL)**: Critique and revision based on principles (implemented here)
2. **Reinforcement Learning (RL)**: Train a preference model and use RL from AI feedback

This repository implements the supervised learning phase, demonstrating the core concept of using constitutional principles to guide AI behavior through self-critique and revision.

## Contributing

This is a portfolio project, but suggestions and improvements are welcome! Feel free to open issues or pull requests.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Anthropic for the Constitutional AI paper and research
- The open-source AI community for tools and inspiration
