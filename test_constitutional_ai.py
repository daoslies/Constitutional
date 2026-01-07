"""
Tests for Constitutional AI implementation.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from constitutional_ai import Principle, CONSTITUTIONAL_PRINCIPLES
from constitutional_ai.constitutional_ai import CritiqueRevisionPair, ConstitutionalResult


def test_principle_creation():
    """Test that principles can be created correctly."""
    principle = Principle(
        name="Test Principle",
        critique_request="Is this a test?",
        revision_request="Please revise this test.",
    )
    assert principle.name == "Test Principle"
    assert "test" in principle.critique_request.lower()
    assert "revise" in principle.revision_request.lower()


def test_constitutional_principles_exist():
    """Test that default constitutional principles are defined."""
    assert len(CONSTITUTIONAL_PRINCIPLES) > 0
    assert all(isinstance(p, Principle) for p in CONSTITUTIONAL_PRINCIPLES)
    
    # Check that key principles exist
    principle_names = [p.name for p in CONSTITUTIONAL_PRINCIPLES]
    assert "Harmlessness" in principle_names
    assert "Helpfulness" in principle_names


def test_constitutional_ai_initialization_openai():
    """Test ConstitutionalAI initialization with OpenAI."""
    # Mock the openai module before importing
    sys.modules['openai'] = MagicMock()
    
    from constitutional_ai import ConstitutionalAI
    
    with patch('openai.OpenAI') as mock_openai:
        cai = ConstitutionalAI(model_provider="openai", api_key="test-key")
        assert cai.model_provider == "openai"
        assert cai.model_name == "gpt-3.5-turbo"
        mock_openai.assert_called_once()


def test_constitutional_ai_initialization_anthropic():
    """Test ConstitutionalAI initialization with Anthropic."""
    # Mock the anthropic module before importing
    sys.modules['anthropic'] = MagicMock()
    
    from constitutional_ai import ConstitutionalAI
    
    with patch('anthropic.Anthropic') as mock_anthropic:
        cai = ConstitutionalAI(model_provider="anthropic", api_key="test-key")
        assert cai.model_provider == "anthropic"
        assert cai.model_name == "claude-3-haiku-20240307"
        mock_anthropic.assert_called_once()


def test_constitutional_ai_invalid_provider():
    """Test that invalid provider raises ValueError."""
    sys.modules['openai'] = MagicMock()
    from constitutional_ai import ConstitutionalAI
    
    with pytest.raises(ValueError):
        ConstitutionalAI(model_provider="invalid_provider")


def test_critique_revision_pair():
    """Test CritiqueRevisionPair data class."""
    pair = CritiqueRevisionPair(
        principle_name="Test",
        original_response="Original",
        critique="This is a critique",
        revised_response="Revised",
    )
    assert pair.principle_name == "Test"
    assert pair.original_response == "Original"
    assert pair.critique == "This is a critique"
    assert pair.revised_response == "Revised"


def test_constitutional_result():
    """Test ConstitutionalResult data class."""
    pair = CritiqueRevisionPair(
        principle_name="Test",
        original_response="Original",
        critique="Critique",
        revised_response="Revised",
    )
    result = ConstitutionalResult(
        prompt="Test prompt",
        initial_response="Initial",
        critique_revision_pairs=[pair],
        final_response="Final",
    )
    assert result.prompt == "Test prompt"
    assert result.initial_response == "Initial"
    assert result.final_response == "Final"
    assert len(result.critique_revision_pairs) == 1
    
    # Test string representation
    result_str = str(result)
    assert "Test prompt" in result_str
    assert "Initial" in result_str
    assert "Final" in result_str


def test_generate_response_openai():
    """Test response generation with OpenAI."""
    # Mock the openai module
    sys.modules['openai'] = MagicMock()
    
    from constitutional_ai import ConstitutionalAI
    
    with patch('openai.OpenAI') as mock_openai_class:
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test
        cai = ConstitutionalAI(model_provider="openai", api_key="test-key")
        response = cai._generate_response("Test prompt")
        
        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()


def test_generate_response_anthropic():
    """Test response generation with Anthropic."""
    # Mock the anthropic module
    sys.modules['anthropic'] = MagicMock()
    
    from constitutional_ai import ConstitutionalAI
    
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_client.messages.create.return_value = mock_response
        
        # Test
        cai = ConstitutionalAI(model_provider="anthropic", api_key="test-key")
        response = cai._generate_response("Test prompt")
        
        assert response == "Test response"
        mock_client.messages.create.assert_called_once()


def test_full_generation_flow():
    """Test the full Constitutional AI generation flow."""
    # Mock the openai module
    sys.modules['openai'] = MagicMock()
    
    from constitutional_ai import ConstitutionalAI
    
    with patch('openai.OpenAI') as mock_openai_class:
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock different responses for different calls
        responses = [
            "Initial response",
            "Critique 1",
            "Revision 1",
            "Critique 2",
            "Revision 2",
        ]
        
        def create_mock_response(content):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = content
            return mock_response
        
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(r) for r in responses
        ]
        
        # Test with 2 principles
        custom_principles = [
            Principle("P1", "critique 1", "revise 1"),
            Principle("P2", "critique 2", "revise 2"),
        ]
        
        cai = ConstitutionalAI(
            model_provider="openai",
            api_key="test-key",
            principles=custom_principles,
        )
        
        result = cai.generate("Test prompt", num_iterations=1, verbose=False)
        
        assert result.prompt == "Test prompt"
        assert result.initial_response == "Initial response"
        assert result.final_response == "Revision 2"
        assert len(result.critique_revision_pairs) == 2
        assert result.critique_revision_pairs[0].principle_name == "P1"
        assert result.critique_revision_pairs[1].principle_name == "P2"


def test_custom_principles():
    """Test using custom principles."""
    # Mock the openai module
    sys.modules['openai'] = MagicMock()
    
    from constitutional_ai import ConstitutionalAI
    
    custom_principles = [
        Principle(
            name="Custom",
            critique_request="Is this custom?",
            revision_request="Make it custom.",
        )
    ]
    
    with patch('openai.OpenAI'):
        cai = ConstitutionalAI(
            model_provider="openai",
            api_key="test-key",
            principles=custom_principles,
        )
        assert len(cai.principles) == 1
        assert cai.principles[0].name == "Custom"

