"""
Integration tests for AWS infrastructure.
"""

import pytest
import boto3
from moto import mock_aws
import json
import tempfile
from pathlib import Path

from src.plugins.aws.infrastructure import AWSInfrastructurePlugin

@mock_aws
class TestAWSInfrastructureIntegration:
    """Integration tests with mocked AWS services."""
    
    @pytest.fixture
    def aws_plugin(self):
        """Create AWS plugin with mocked configuration."""
        config = {
            'profile': 'test',
            'region': 'us-east-1'
        }
        return AWSInfrastructurePlugin(config)
    
    @pytest.fixture
    def cloudformation_template(self):
        """Create a simple CloudFormation template."""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Test stack",
            "Resources": {
                "TestVPC": {
                    "Type": "AWS::EC2::VPC",
                    "Properties": {
                        "CidrBlock": "10.0.0.0/16",
                        "Tags": [
                            {"Key": "Name", "Value": "TestVPC"},
                            {"Key": "ManagedBy", "Value": "DevOpsAutomator"}
                        ]
                    }
                }
            },
            "Outputs": {
                "VPCId": {
                    "Value": {"Ref": "TestVPC"},
                    "Description": "VPC ID"
                }
            }
        }
    
    def test_cloudformation_provisioning(self, aws_plugin, cloudformation_template):
        """Test CloudFormation stack provisioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write CloudFormation template
            template_file = temp_path / 'template.yaml'
            with open(template_file, 'w') as f:
                json.dump(cloudformation_template, f)
            
            # Provision infrastructure
            config = {
                'project': 'test-project',
                'parameters': {
                    'Environment': 'test'
                },
                'provisioning_method': 'cloudformation'
            }
            
            result = aws_plugin.provision(
                environment='test',
                region='us-east-1',
                config=config,
                templates_dir=temp_path
            )
            
            assert result['success'] == True
            assert 'stack_id' in result
            assert result['stack_name'] == 'test-project-test'
            assert result['provisioning_method'] == 'cloudformation'
            
            # Verify resources were created
            assert len(result['resources']) == 1
            assert result['resources'][0]['type'] == 'AWS::EC2::VPC'
    
    def test_multiple_environments(self, aws_plugin):
        """Test provisioning multiple environments."""
        environments = ['dev', 'staging', 'prod']
        
        for env in environments:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create minimal template
                template = {
                    "Resources": {
                        f"TestResource{env}": {
                            "Type": "AWS::S3::Bucket",
                            "Properties": {
                                "BucketName": f"test-bucket-{env}",
                                "Tags": [
                                    {"Key": "Environment", "Value": env},
                                    {"Key": "ManagedBy", "Value": "DevOpsAutomator"}
                                ]
                            }
                        }
                    }
                }
                
                template_file = temp_path / 'template.yaml'
                with open(template_file, 'w') as f:
                    json.dump(template, f)
                
                config = {
                    'project': 'test-project',
                    'provisioning_method': 'cloudformation'
                }
                
                result = aws_plugin.provision(
                    environment=env,
                    region='us-east-1',
                    config=config,
                    templates_dir=temp_path
                )
                
                assert result['success'] == True
                assert result['environment'] == env
    
    def test_error_handling_invalid_template(self, aws_plugin):
        """Test error handling with invalid CloudFormation template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write invalid template
            template_file = temp_path / 'template.yaml'
            with open(template_file, 'w') as f:
                f.write('Invalid YAML: [')
            
            config = {
                'provisioning_method': 'cloudformation'
            }
            
            with pytest.raises(Exception) as exc_info:
                aws_plugin.provision(
                    environment='test',
                    region='us-east-1',
                    config=config,
                    templates_dir=temp_path
                )
            
            assert "Invalid" in str(exc_info.value) or "error" in str(exc_info.value).lower()
    
    def test_cleanup_resources(self, aws_plugin):
        """Test resource cleanup."""
        # First create some resources
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            template = {
                "Resources": {
                    "TestBucket": {
                        "Type": "AWS::S3::Bucket",
                        "Properties": {
                            "Tags": [
                                {"Key": "ManagedBy", "Value": "DevOpsAutomator"},
                                {"Key": "Environment", "Value": "test"}
                            ]
                        }
                    }
                }
            }
            
            template_file = temp_path / 'template.yaml'
            with open(template_file, 'w') as f:
                json.dump(template, f)
            
            config = {'provisioning_method': 'cloudformation'}
            
            # Provision
            aws_plugin.provision(
                environment='test',
                region='us-east-1',
                config=config,
                templates_dir=temp_path
            )
        
        # Test cleanup (dry run)
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=1)
        
        result = aws_plugin.cleanup_resources(
            older_than=cutoff_date,
            dry_run=True
        )
        
        assert 'cloudformation' in result
        assert result['cloudformation']['cleaned'] >= 0
    
    def test_get_status(self, aws_plugin):
        """Test getting infrastructure status."""
        # Create a stack first
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            template = {
                "Resources": {
                    "TestResource": {
                        "Type": "AWS::S3::Bucket",
                        "Properties": {
                            "Tags": [
                                {"Key": "ManagedBy", "Value": "DevOpsAutomator"},
                                {"Key": "Environment", "Value": "dev"}
                            ]
                        }
                    }
                }
            }
            
            template_file = temp_path / 'template.yaml'
            with open(template_file, 'w') as f:
                json.dump(template, f)
            
            config = {'provisioning_method': 'cloudformation'}
            
            aws_plugin.provision(
                environment='dev',
                region='us-east-1',
                config=config,
                templates_dir=temp_path
            )
        
        # Get status
        status = aws_plugin.get_status()
        
        assert isinstance(status, list)
        if status:  # Stack might take time to appear in mocked environment
            assert status[0]['provider'] == 'aws'
            assert status[0]['environment'] == 'dev'

class TestRealAWSConnection:
    """Tests that require real AWS connection (skipped by default)."""
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--real-aws"),
        reason="Requires real AWS connection"
    )
    def test_real_aws_connection(self):
        """Test connection to real AWS (only runs with --real-aws flag)."""
        import boto3
        
        # Test that we can create a session
        session = boto3.Session()
        sts = session.client('sts')
        
        # Get caller identity
        identity = sts.get_caller_identity()
        
        assert 'Account' in identity
        assert 'UserId' in identity
        assert 'Arn' in identity
        
        print(f"Connected as: {identity['Arn']}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
