#!/usr/bin/env python3
"""
Simple Demo Script - No API Key Required

This script demonstrates the Constitutional AI structure and flow
without requiring actual API calls. It uses mock responses to show
how the system works.
"""

from constitutional_ai import Principle, CONSTITUTIONAL_PRINCIPLES
from constitutional_ai.constitutional_ai import CritiqueRevisionPair, ConstitutionalResult


def demonstrate_structure():
    """Show the structure and flow of Constitutional AI."""
    
    print("=" * 80)
    print("CONSTITUTIONAL AI DEMONSTRATION (No API Key Required)")
    print("=" * 80)
    
    print("\n1. CONSTITUTIONAL PRINCIPLES")
    print("-" * 80)
    print(f"The system includes {len(CONSTITUTIONAL_PRINCIPLES)} core principles:\n")
    for i, principle in enumerate(CONSTITUTIONAL_PRINCIPLES, 1):
        print(f"{i}. {principle.name}")
        print(f"   Critique: {principle.critique_request[:70]}...")
        print(f"   Revision: {principle.revision_request[:70]}...")
        print()
    
    print("\n2. CUSTOM PRINCIPLES")
    print("-" * 80)
    print("You can also define custom principles:\n")
    custom = Principle(
        name="Conciseness",
        critique_request="Is this response too verbose or repetitive?",
        revision_request="Make this response more concise.",
    )
    print(f"Name: {custom.name}")
    print(f"Critique: {custom.critique_request}")
    print(f"Revision: {custom.revision_request}")
    
    print("\n\n3. CONSTITUTIONAL AI FLOW")
    print("-" * 80)
    print("The Constitutional AI process works as follows:\n")
    
    # Simulate a result
    prompt = "How do I handle workplace conflict?"
    initial = "You should always confront people directly and tell them exactly what you think."
    
    pairs = [
        CritiqueRevisionPair(
            principle_name="Helpfulness",
            original_response=initial,
            critique="The response is too aggressive and doesn't consider different conflict resolution styles.",
            revised_response="Consider different approaches based on the situation: direct communication when appropriate, mediation for serious conflicts, and focusing on specific behaviors rather than personal attacks.",
        ),
        CritiqueRevisionPair(
            principle_name="Appropriate Boundaries",
            original_response="Consider different approaches based on the situation...",
            critique="The response could better emphasize professional boundaries and HR involvement when needed.",
            revised_response="Address workplace conflicts professionally: 1) Try direct, respectful communication first, 2) Focus on specific behaviors and their impact, 3) Involve HR or management for serious issues, 4) Document important conversations, 5) Maintain professional boundaries throughout.",
        ),
    ]
    
    result = ConstitutionalResult(
        prompt=prompt,
        initial_response=initial,
        critique_revision_pairs=pairs,
        final_response=pairs[-1].revised_response,
    )
    
    print("Step 1: User Prompt")
    print(f"  → {result.prompt}")
    
    print("\nStep 2: Initial Response (before Constitutional AI)")
    print(f"  → {result.initial_response}")
    
    print("\nStep 3: Critique & Revision Cycles")
    for i, pair in enumerate(result.critique_revision_pairs, 1):
        print(f"\n  Cycle {i} - Principle: {pair.principle_name}")
        print(f"    Critique: {pair.critique}")
        print(f"    Revised:  {pair.revised_response[:100]}...")
    
    print("\nStep 4: Final Response (after Constitutional AI)")
    print(f"  → {result.final_response}")
    
    print("\n\n4. USAGE WITH REAL API")
    print("-" * 80)
    print("""
To use with a real API, set your API key and run:

    export OPENAI_API_KEY="your-key-here"
    # or
    export ANTHROPIC_API_KEY="your-key-here"
    
    python example.py

The example.py script shows several use cases including:
- Simple helpful requests
- Requests requiring ethical guidance
- Custom principles
- Multiple iteration cycles
    """)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThis implementation demonstrates the core concept from:")
    print("'Constitutional AI: Harmlessness from AI Feedback' (Bai et al., 2022)")
    print("\nKey Insight: AI systems can improve their own responses by")
    print("critiquing them against constitutional principles and revising iteratively.")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_structure()
