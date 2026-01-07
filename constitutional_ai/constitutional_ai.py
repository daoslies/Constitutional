"""
Constitutional AI Main Implementation

Implements the core Constitutional AI algorithm from Anthropic's paper.
This includes the critique and revision loop that improves responses based on
constitutional principles.
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .principles import Principle, CONSTITUTIONAL_PRINCIPLES


@dataclass
class CritiqueRevisionPair:
    """A critique and its corresponding revision."""
    
    principle_name: str
    original_response: str
    critique: str
    revised_response: str


@dataclass
class ConstitutionalResult:
    """Result of the constitutional AI process."""
    
    prompt: str
    initial_response: str
    critique_revision_pairs: List[CritiqueRevisionPair]
    final_response: str
    
    def __str__(self):
        output = [
            "=" * 80,
            "Constitutional AI Result",
            "=" * 80,
            f"\nPrompt: {self.prompt}\n",
            f"Initial Response:\n{self.initial_response}\n",
        ]
        
        for i, pair in enumerate(self.critique_revision_pairs, 1):
            output.append(f"\n--- Revision {i} (Principle: {pair.principle_name}) ---")
            output.append(f"Critique: {pair.critique}")
            output.append(f"Revised Response:\n{pair.revised_response}\n")
        
        output.append(f"\nFinal Response:\n{self.final_response}")
        output.append("=" * 80)
        
        return "\n".join(output)


class ConstitutionalAI:
    """
    Constitutional AI implementation.
    
    This class implements the supervised learning phase of Constitutional AI,
    which involves generating an initial response, critiquing it based on
    constitutional principles, and revising it iteratively.
    """
    
    def __init__(
        self,
        model_provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        principles: Optional[List[Principle]] = None,
    ):
        """
        Initialize the Constitutional AI system.
        
        Args:
            model_provider: The LLM provider to use ("openai" or "anthropic")
            model_name: The specific model to use (defaults based on provider)
            api_key: API key for the provider (or use environment variable)
            principles: List of constitutional principles to apply
        """
        self.model_provider = model_provider.lower()
        self.principles = principles or CONSTITUTIONAL_PRINCIPLES
        
        # Set up the model client
        if self.model_provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name or "gpt-3.5-turbo"
        elif self.model_provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = model_name or "claude-3-haiku-20240307"
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
    
    def _generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            The generated response
        """
        if self.model_provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        
        elif self.model_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.content[0].text
    
    def _critique_response(self, prompt: str, response: str, principle: Principle) -> str:
        """
        Generate a critique of a response based on a constitutional principle.
        
        Args:
            prompt: The original user prompt
            response: The response to critique
            principle: The constitutional principle to apply
            
        Returns:
            The critique
        """
        critique_prompt = f"""Original Human Request: {prompt}

Assistant Response: {response}

{principle.critique_request}

Please provide a detailed critique based on this constitutional principle."""
        
        return self._generate_response(critique_prompt)
    
    def _revise_response(
        self, prompt: str, response: str, critique: str, principle: Principle
    ) -> str:
        """
        Revise a response based on critique and constitutional principle.
        
        Args:
            prompt: The original user prompt
            response: The original response
            critique: The critique of the response
            principle: The constitutional principle to apply
            
        Returns:
            The revised response
        """
        revision_prompt = f"""Original Human Request: {prompt}

Original Assistant Response: {response}

Critique: {critique}

{principle.revision_request}

Please provide an improved response that addresses the critique."""
        
        return self._generate_response(revision_prompt)
    
    def generate(
        self,
        prompt: str,
        num_iterations: int = 1,
        verbose: bool = False,
    ) -> ConstitutionalResult:
        """
        Generate a response using Constitutional AI.
        
        This implements the core Constitutional AI algorithm:
        1. Generate an initial response
        2. For each constitutional principle:
           a. Critique the response
           b. Revise based on the critique
        3. Return the final revised response
        
        Args:
            prompt: The user prompt
            num_iterations: Number of critique-revision cycles (default: 1)
            verbose: Print progress information
            
        Returns:
            ConstitutionalResult containing the complete process
        """
        if verbose:
            print(f"Generating initial response for: {prompt[:100]}...")
        
        # Step 1: Generate initial response
        current_response = self._generate_response(prompt)
        initial_response = current_response
        
        if verbose:
            print(f"Initial response generated: {current_response[:100]}...")
        
        # Step 2: Apply constitutional principles
        critique_revision_pairs = []
        
        for iteration in range(num_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            for principle in self.principles:
                if verbose:
                    print(f"  Applying principle: {principle.name}")
                
                # Generate critique
                critique = self._critique_response(prompt, current_response, principle)
                
                if verbose:
                    print(f"    Critique: {critique[:100]}...")
                
                # Generate revision
                revised_response = self._revise_response(
                    prompt, current_response, critique, principle
                )
                
                if verbose:
                    print(f"    Revision: {revised_response[:100]}...")
                
                # Store the critique-revision pair
                critique_revision_pairs.append(
                    CritiqueRevisionPair(
                        principle_name=principle.name,
                        original_response=current_response,
                        critique=critique,
                        revised_response=revised_response,
                    )
                )
                
                # Update current response for next iteration
                current_response = revised_response
        
        if verbose:
            print("\nConstitutional AI process complete!")
        
        return ConstitutionalResult(
            prompt=prompt,
            initial_response=initial_response,
            critique_revision_pairs=critique_revision_pairs,
            final_response=current_response,
        )