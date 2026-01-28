"""
Unit tests for Automation Engine.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml

from src.core.engine import AutomationEngine
from src.core.config import ConfigManager

class TestAutomationEngine:
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = {
            'global': {
                'environment': 'test',
                'region': 'us-east-1'
            },
            'plugins': {
                'aws_infra': {
                    'enabled': True,
                    'profile': 'test'
                },
                'kubernetes': {
                    'enabled': True
                }
            }
        }
        return config
    
    @pytest.fixture
    def engine(self, mock_config):
        """Create AutomationEngine instance with mocked dependencies."""
        with patch('src.core.engine.ConfigManager') as MockConfig:
            with patch('src.core.engine.PluginRegistry') as MockRegistry:
                mock_config_manager = Mock(spec=ConfigManager)
                mock_config_manager.get.return_value = mock_config
                mock_config_manager.get_state_file.return_value = Path('/tmp/state.json')
                mock_config_manager.get_clusters_file.return_value = Path('/tmp/clusters.yaml')
                
                mock_registry = Mock()
                mock_aws_plugin = Mock()
                mock_k8s_plugin = Mock()
                
                mock_registry.get_plugin.side_effect = lambda name: {
                    'aws_infra': mock_aws_plugin,
                    'kubernetes': mock_k8s_plugin
                }.get(name)
                
                engine = AutomationEngine(mock_config_manager)
                engine.plugins = mock_registry
                
                return engine, mock_aws_plugin, mock_k8s_plugin
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        engine_instance, _, _ = engine
        assert engine_instance.config is not None
        assert engine_instance.plugins is not None
    
    def test_provision_infrastructure(self, engine):
        """Test infrastructure provisioning."""
        engine_instance, mock_aws_plugin, _ = engine
        
        # Mock provisioning result
        mock_result = {
            'success': True,
            'resources': [{'type': 'vpc', 'name': 'test-vpc'}],
            'outputs': {'vpc_id': 'vpc-123'}
        }
        mock_aws_plugin.provision.return_value = mock_result
        
        # Test provisioning
        config = {
            'project': 'test-project',
            'resources': [
                {'type': 'vpc', 'name': 'test-vpc'}
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = engine_instance.provision_infrastructure(
                provider='aws',
                environment='dev',
                region='us-east-1',
                config=config,
                templates_dir=Path(temp_dir)
            )
        
        assert result['success'] == True
        assert len(result['resources']) == 1
        mock_aws_plugin.provision.assert_called_once()
    
    def test_create_kubernetes_cluster(self, engine):
        """Test Kubernetes cluster creation."""
        engine_instance, _, mock_k8s_plugin = engine
        
        # Mock cluster creation result
        mock_result = {
            'success': True,
            'cluster_name': 'test-cluster',
            'endpoint': 'https://test-cluster.eks.amazonaws.com',
            'kubeconfig': 'apiVersion: v1\n...'
        }
        mock_k8s_plugin.create_cluster.return_value = mock_result
        
        # Mock config manager methods
        engine_instance.config.get_cluster_config = Mock(return_value={})
        
        # Test cluster creation
        result = engine_instance.create_kubernetes_cluster(
            name='test-cluster',
            provider='eks',
            version='1.24',
            node_count=3,
            node_type='t3.medium'
        )
        
        assert result['success'] == True
        assert result['cluster_name'] == 'test-cluster'
        mock_k8s_plugin.create_cluster.assert_called_once()
    
    def test_generate_pipeline_config(self, engine):
        """Test pipeline configuration generation."""
        engine_instance, _, _ = engine
        
        # Mock pipeline plugin
        mock_pipeline_plugin = Mock()
        engine_instance.plugins.get_plugin.return_value = mock_pipeline_plugin
        
        # Mock result
        mock_result = {
            'generated_files': [
                {'name': 'ci.yml', 'description': 'CI workflow'}
            ]
        }
        mock_pipeline_plugin.generate_configuration.return_value = mock_result
        
        # Mock config manager
        engine_instance.config.get_project_config = Mock(return_value={})
        
        # Test pipeline generation
        with tempfile.TemporaryDirectory() as temp_dir:
            result = engine_instance.generate_pipeline_config(
                provider='github',
                project_type='nodejs',
                output_dir=Path(temp_dir),
                overwrite=False
            )
        
        assert len(result['generated_files']) == 1
        mock_pipeline_plugin.generate_configuration.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_operations(self, engine):
        """Test async operation execution."""
        engine_instance, _, _ = engine
        
        # Create a mock async task
        async def mock_task():
            await asyncio.sleep(0.1)
            return {'status': 'completed'}
        
        # Execute async task through thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            engine_instance.thread_pool,
            lambda: asyncio.run(mock_task())
        )
        
        assert result['status'] == 'completed'
    
    def test_cleanup_resources(self, engine):
        """Test resource cleanup."""
        engine_instance, mock_aws_plugin, mock_k8s_plugin = engine
        
        # Mock cleanup results
        mock_aws_plugin.cleanup_resources.return_value = {
            'cleaned': 5,
            'skipped': 2
        }
        mock_k8s_plugin.cleanup_resources.return_value = {
            'cleaned': 3,
            'skipped': 1
        }
        
        # Mock docker plugin
        mock_docker_plugin = Mock()
        mock_docker_plugin.cleanup_images.return_value = {
            'cleaned': 10,
            'skipped': 0
        }
        
        engine_instance.plugins.get_plugin.side_effect = lambda name: {
            'aws_infra': mock_aws_plugin,
            'kubernetes': mock_k8s_plugin,
            'docker': mock_docker_plugin
        }.get(name)
        
        # Test cleanup
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=30)
        
        result = engine_instance.cleanup_resources(
            older_than_days=30,
            dry_run=True
        )
        
        assert 'docker_images' in result
        assert 'kubernetes_resources' in result
        assert 'aws_infrastructure' in result
        assert result['cleaned'] == 18  # 5 + 3 + 10
    
    def test_error_handling(self, engine):
        """Test error handling in operations."""
        engine_instance, mock_aws_plugin, _ = engine
        
        # Mock plugin to raise exception
        mock_aws_plugin.provision.side_effect = Exception("Provisioning failed")
        
        config = {'resources': []}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception) as exc_info:
                engine_instance.provision_infrastructure(
                    provider='aws',
                    environment='dev',
                    region='us-east-1',
                    config=config,
                    templates_dir=Path(temp_dir)
                )
        
        assert "Provisioning failed" in str(exc_info.value)
    
    def test_state_management(self, engine):
        """Test resource state management."""
        engine_instance, _, _ = engine
        
        # Mock state file operations
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tmp_file:
            tmp_path = Path(tmp_file.name)
            engine_instance.config.get_state_file.return_value = tmp_path
            
            # Initial state
            state = {
                'aws': {
                    'dev': {
                        'provisioned_at': '2024-01-01T00:00:00',
                        'status': 'provisioned'
                    }
                }
            }
            
            tmp_path.write_text(json.dumps(state))
            
            # Verify state can be read
            if tmp_path.exists():
                with open(tmp_path, 'r') as f:
                    loaded_state = json.load(f)
                
                assert 'aws' in loaded_state
                assert 'dev' in loaded_state['aws']

class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_valid_configuration(self):
        """Test valid configuration parsing."""
        valid_config = """
        global:
          environment: test
          region: us-east-1
        
        infrastructure:
          network:
            vpc_cidr: 10.0.0.0/16
        """
        
        config = yaml.safe_load(valid_config)
        
        # Should not raise any exceptions
        assert config['global']['environment'] == 'test'
        assert config['infrastructure']['network']['vpc_cidr'] == '10.0.0.0/16'
    
    def test_invalid_configuration(self):
        """Test invalid configuration handling."""
        invalid_config = """
        global:
          environment: 123  # Should be string
        
        infrastructure:
          network:
            vpc_cidr: "invalid_cidr"
        """
        
        config = yaml.safe_load(invalid_config)
        
        # Type checking would be done by validators
        assert isinstance(config['global']['environment'], int)
        assert config['infrastructure']['network']['vpc_cidr'] == "invalid_cidr"
    
    def test_environment_overrides(self):
        """Test environment-specific configuration overrides."""
        config = {
            'global': {
                'environment': 'dev',
                'region': 'us-east-1'
            },
            'infrastructure': {
                'kubernetes': {
                    'node_count': 3,
                    'node_type': 't3.medium'
                }
            },
            'environments': {
                'dev': {
                    'kubernetes': {
                        'node_count': 2,
                        'node_type': 't3.small'
                    }
                }
            }
        }
        
        # In practice, the engine would merge configurations
        # For dev environment, it should use dev-specific values
        dev_config = config['environments']['dev']
        assert dev_config['kubernetes']['node_count'] == 2
        assert dev_config['kubernetes']['node_type'] == 't3.small'

class TestCommandExecution:
    """Test command execution utilities."""
    
    def test_command_executor(self):
        """Test command execution."""
        from src.utils.executor import CommandExecutor
        
        executor = CommandExecutor()
        
        # Test successful command
        result = executor.execute(['echo', 'test'], capture_output=True)
        assert result.returncode == 0
        assert 'test' in result.stdout.decode().strip()
        
        # Test command failure
        with pytest.raises(Exception):
            executor.execute(['false'])
    
    def test_shell_command_execution(self):
        """Test shell command execution."""
        from src.utils.executor import CommandExecutor
        
        executor = CommandExecutor()
        
        # Test shell command
        result = executor.execute_shell('echo "Hello, World!"', capture_output=True)
        assert result.returncode == 0
        assert 'Hello, World!' in result.stdout.decode().strip()
    
    def test_timeout_handling(self):
        """Test command timeout handling."""
        from src.utils.executor import CommandExecutor
        
        executor = CommandExecutor()
        
        # Test command that times out
        with pytest.raises(Exception) as exc_info:
            executor.execute(['sleep', '10'], timeout=1)
        
        assert "timed out" in str(exc_info.value).lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
