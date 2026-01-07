"""
Example usage of the Constitutional AI implementation.

This demonstrates how to use Constitutional AI to generate responses that
are critiqued and improved based on constitutional principles.
"""

import os
from constitutional_ai import ConstitutionalAI, CONSTITUTIONAL_PRINCIPLES


def main():
    """Run example Constitutional AI generations."""
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize Constitutional AI
    # Use OpenAI by default, or Anthropic if ANTHROPIC_API_KEY is set and OPENAI_API_KEY is not
    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
    
    print(f"Using {provider} as the model provider")
    print("=" * 80)
    
    cai = ConstitutionalAI(model_provider=provider)
    
    # Example 1: Simple helpful request
    print("\n\nExample 1: Helpful Request")
    print("-" * 80)
    result = cai.generate(
        prompt="How do I make a good first impression at a job interview?",
        num_iterations=1,
        verbose=True,
    )
    print("\n" + str(result))
    
    # Example 2: Request that might need ethical guidance
    print("\n\nExample 2: Request Requiring Ethical Guidance")
    print("-" * 80)
    result = cai.generate(
        prompt="What's the best way to win an argument with someone?",
        num_iterations=1,
        verbose=True,
    )
    print("\n" + str(result))
    
    # Example 3: Custom principles
    from constitutional_ai import Principle
    
    custom_principles = [
        Principle(
            name="Conciseness",
            critique_request="Is this response too verbose or repetitive?",
            revision_request="Please make this response more concise while keeping the essential information.",
        )
    ]
    
    print("\n\nExample 3: Custom Principle (Conciseness)")
    print("-" * 80)
    cai_custom = ConstitutionalAI(model_provider=provider, principles=custom_principles)
    result = cai_custom.generate(
        prompt="Explain what machine learning is.",
        num_iterations=1,
        verbose=True,
    )
    print("\n" + str(result))


if __name__ == "__main__":
    main()
