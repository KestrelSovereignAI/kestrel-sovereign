"""
Unit tests for docstring parameter parsing in features/base.py.

Tests the parse_docstring_params function and tool decorator integration.
"""

import pytest
from kestrel_sovereign.features.base import parse_docstring_params, tool, Feature
from kestrel_sovereign.tools.base import ToolCategory


class TestParseDocstringParams:
    """Tests for the parse_docstring_params function."""
    
    def test_google_style_docstring(self):
        """Test parsing Google-style docstrings."""
        docstring = '''
        Do something useful.
        
        Args:
            file_path: The path to the file to process
            count: Number of items to process
            verbose: Whether to output verbose logs
        
        Returns:
            The processed result
        '''
        
        result = parse_docstring_params(docstring)
        
        assert result['file_path'] == 'The path to the file to process'
        assert result['count'] == 'Number of items to process'
        assert result['verbose'] == 'Whether to output verbose logs'
    
    def test_google_style_with_types(self):
        """Test Google-style with type annotations in docstring."""
        docstring = '''
        Process data.
        
        Args:
            name (str): The name of the item
            count (int): How many items to process
            options (dict): Configuration options
        '''
        
        result = parse_docstring_params(docstring)
        
        assert result['name'] == 'The name of the item'
        assert result['count'] == 'How many items to process'
        assert result['options'] == 'Configuration options'
    
    def test_sphinx_style_docstring(self):
        """Test parsing Sphinx-style docstrings."""
        docstring = '''
        Do something useful.
        
        :param file_path: The path to the file to process
        :param count: Number of items to process
        :returns: The processed result
        '''
        
        result = parse_docstring_params(docstring)
        
        assert result['file_path'] == 'The path to the file to process'
        assert result['count'] == 'Number of items to process'
    
    def test_multiline_descriptions(self):
        """Test that multi-line descriptions are collapsed."""
        docstring = '''
        Do something.
        
        Args:
            config: The configuration dictionary containing
                all the settings needed for processing
                including nested options
            name: Simple name param
        '''
        
        result = parse_docstring_params(docstring)
        
        # Multi-line should be collapsed to single line
        assert 'configuration dictionary' in result['config']
        assert result['name'] == 'Simple name param'
    
    def test_empty_docstring(self):
        """Test handling of empty docstring."""
        assert parse_docstring_params(None) == {}
        assert parse_docstring_params('') == {}
    
    def test_docstring_without_params(self):
        """Test docstring with no parameter section."""
        docstring = '''
        This function does something.
        
        Returns:
            Nothing useful.
        '''
        
        result = parse_docstring_params(docstring)
        assert result == {}
    
    def test_parameters_section_variant(self):
        """Test 'Parameters:' section header (NumPy style)."""
        docstring = '''
        Do something.
        
        Parameters:
            data: The input data to process
            output_path: Where to write results
        '''
        
        result = parse_docstring_params(docstring)
        
        assert result['data'] == 'The input data to process'
        assert result['output_path'] == 'Where to write results'
    
    def test_arguments_section_variant(self):
        """Test 'Arguments:' section header."""
        docstring = '''
        Do something.
        
        Arguments:
            source: The source file
            destination: The destination path
        '''
        
        result = parse_docstring_params(docstring)
        
        assert result['source'] == 'The source file'
        assert result['destination'] == 'The destination path'


class TestToolDecorator:
    """Tests for the @tool decorator with docstring parsing."""
    
    def test_tool_extracts_param_descriptions(self):
        """Test that @tool decorator extracts parameter descriptions."""
        
        @tool("test_tool", "A test tool")
        async def my_tool(self, file_path: str, count: int = 10):
            '''
            Process a file.
            
            Args:
                file_path: The path to the file to process
                count: Number of items (default: 10)
            '''
            pass
        
        schema = my_tool._tool_schema
        params = {p.name: p for p in schema['parameters']}
        
        assert params['file_path'].description == 'The path to the file to process'
        assert params['count'].description == 'Number of items (default: 10)'
    
    def test_tool_fallback_description(self):
        """Test fallback when docstring doesn't have param info."""
        
        @tool("test_tool", "A test tool")
        async def my_tool(self, some_value: str):
            '''Just do something.'''
            pass
        
        schema = my_tool._tool_schema
        params = {p.name: p for p in schema['parameters']}
        
        # Should have human-readable fallback
        assert params['some_value'].description == 'The some value parameter'
    
    def test_tool_no_docstring(self):
        """Test when function has no docstring."""
        
        @tool("test_tool", "A test tool")
        async def my_tool(self, input_data: str):
            pass
        
        schema = my_tool._tool_schema
        params = {p.name: p for p in schema['parameters']}
        
        # Should have human-readable fallback
        assert params['input_data'].description == 'The input data parameter'
    
    def test_tool_preserves_types(self):
        """Test that parameter types are correctly inferred."""
        
        @tool("test_tool", "A test tool")
        async def my_tool(self, name: str, count: int, enabled: bool, ratio: float):
            '''
            Test function.
            
            Args:
                name: The name
                count: The count
                enabled: Is enabled
                ratio: The ratio
            '''
            pass
        
        schema = my_tool._tool_schema
        params = {p.name: p for p in schema['parameters']}
        
        assert params['name'].type == 'string'
        assert params['count'].type == 'integer'
        assert params['enabled'].type == 'boolean'
        assert params['ratio'].type == 'number'
    
    def test_tool_required_vs_optional(self):
        """Test that required vs optional parameters are detected."""
        
        @tool("test_tool", "A test tool")
        async def my_tool(self, required_param: str, optional_param: str = "default"):
            '''
            Args:
                required_param: This is required
                optional_param: This is optional
            '''
            pass
        
        schema = my_tool._tool_schema
        params = {p.name: p for p in schema['parameters']}
        
        assert params['required_param'].required is True
        assert params['optional_param'].required is False
        assert params['optional_param'].default == "default"
    
    def test_tool_sphinx_style_docstring(self):
        """Test @tool with Sphinx-style docstring."""
        
        @tool("test_tool", "A test tool")
        async def my_tool(self, cid: str, destination: str):
            '''
            Download something.
            
            :param cid: The content identifier
            :param destination: Where to save the file
            :returns: The download result
            '''
            pass
        
        schema = my_tool._tool_schema
        params = {p.name: p for p in schema['parameters']}
        
        assert params['cid'].description == 'The content identifier'
        assert params['destination'].description == 'Where to save the file'


class TestToolDecoratorIntegration:
    """Integration tests with Feature class."""
    
    def test_feature_get_tools_has_descriptions(self):
        """Test that Feature.get_tools() returns tools with proper descriptions."""
        
        class TestFeature(Feature):
            @property
            def tool_description(self) -> str:
                return "A test feature"

            def initialize(self):
                pass

            @tool("test_action", "Perform a test action", ToolCategory.SYSTEM)
            @pytest.mark.asyncio
            async def test_action(self, target: str, force: bool = False):
                '''
                Perform an action on a target.
                
                Args:
                    target: The target to act upon
                    force: Whether to force the action
                '''
                return {"success": True}
        
        # Create feature with mock agent
        feature = TestFeature(agent=None)
        tools = feature.get_tools()
        
        assert len(tools) == 1
        tool_instance = tools[0]
        
        # Check schema has correct descriptions
        schema = tool_instance.schema
        params = {p.name: p for p in schema.parameters}
        
        assert params['target'].description == 'The target to act upon'
        assert params['force'].description == 'Whether to force the action'
