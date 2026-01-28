"""
AWS infrastructure automation plugin.
"""

import boto3
import botocore
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import subprocess
import tempfile
import time

from src.plugins.base import BasePlugin
from src.utils.logger import get_logger
from src.utils.executor import CommandExecutor

logger = get_logger(__name__)

class AWSInfrastructurePlugin(BasePlugin):
    """AWS infrastructure automation plugin."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize AWS clients
        self.session = boto3.Session(
            profile_name=config.get('profile', 'default'),
            region_name=config.get('region', 'us-east-1')
        )
        
        self.clients = {}
        self._init_clients()
        
        # Terraform executor
        self.executor = CommandExecutor()
    
    def _init_clients(self):
        """Initialize AWS service clients."""
        services = [
            'ec2', 'rds', 's3', 'iam', 'cloudformation',
            'eks', 'ecr', 'elbv2', 'autoscaling', 'cloudwatch'
        ]
        
        for service in services:
            try:
                self.clients[service] = self.session.client(service)
            except Exception as e:
                logger.warning(f"Failed to initialize {service} client: {e}")
    
    def provision(self, environment: str, region: str,
                 config: Dict[str, Any], templates_dir: Path) -> Dict[str, Any]:
        """
        Provision AWS infrastructure.
        
        Args:
            environment: Environment name
            region: AWS region
            config: Infrastructure configuration
            templates_dir: Directory containing IaC templates
        
        Returns:
            Dictionary with provisioning results
        """
        logger.info(f"Provisioning AWS infrastructure in {region} for {environment}")
        
        # Determine provisioning method
        method = config.get('provisioning_method', 'terraform')
        
        if method == 'terraform':
            return self._provision_with_terraform(environment, region, config, templates_dir)
        elif method == 'cloudformation':
            return self._provision_with_cloudformation(environment, region, config, templates_dir)
        elif method == 'cdk':
            return self._provision_with_cdk(environment, region, config, templates_dir)
        else:
            raise ValueError(f"Unsupported provisioning method: {method}")
    
    def _provision_with_terraform(self, environment: str, region: str,
                                config: Dict[str, Any], templates_dir: Path) -> Dict[str, Any]:
        """Provision using Terraform."""
        try:
            # Initialize Terraform
            init_cmd = ['terraform', 'init', '-upgrade']
            self.executor.execute(init_cmd, cwd=templates_dir)
            
            # Plan
            plan_file = templates_dir / 'terraform.plan'
            plan_cmd = [
                'terraform', 'plan',
                '-var', f'environment={environment}',
                '-var', f'region={region}',
                '-out', str(plan_file)
            ]
            
            # Add additional variables from config
            for key, value in config.get('variables', {}).items():
                plan_cmd.extend(['-var', f'{key}={value}'])
            
            self.executor.execute(plan_cmd, cwd=templates_dir)
            
            # Apply
            apply_cmd = ['terraform', 'apply', str(plan_file)]
            self.executor.execute(apply_cmd, cwd=templates_dir)
            
            # Get outputs
            output_cmd = ['terraform', 'output', '-json']
            output_result = self.executor.execute(output_cmd, cwd=templates_dir, capture_output=True)
            outputs = json.loads(output_result.stdout)
            
            # Get created resources
            state_cmd = ['terraform', 'show', '-json']
            state_result = self.executor.execute(state_cmd, cwd=templates_dir, capture_output=True)
            state = json.loads(state_result.stdout)
            
            resources = self._extract_resources_from_state(state)
            
            return {
                'success': True,
                'environment': environment,
                'region': region,
                'resources': resources,
                'outputs': outputs,
                'provisioning_method': 'terraform'
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Terraform provisioning failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Terraform provisioning failed: {e}")
            raise
    
    def _provision_with_cloudformation(self, environment: str, region: str,
                                     config: Dict[str, Any], templates_dir: Path) -> Dict[str, Any]:
        """Provision using AWS CloudFormation."""
        cf_client = self.clients['cloudformation']
        stack_name = f"{config.get('project', 'devops')}-{environment}"
        
        try:
            # Find CloudFormation template
            template_files = list(templates_dir.glob('*.yaml')) + list(templates_dir.glob('*.yml'))
            if not template_files:
                template_files = list(templates_dir.glob('*.json'))
            
            if not template_files:
                raise ValueError("No CloudFormation template found")
            
            template_path = template_files[0]
            with open(template_path, 'r') as f:
                template_body = f.read()
            
            # Check if stack exists
            try:
                cf_client.describe_stacks(StackName=stack_name)
                stack_exists = True
            except cf_client.exceptions.ClientError:
                stack_exists = False
            
            # Prepare parameters
            parameters = []
            for key, value in config.get('parameters', {}).items():
                parameters.append({
                    'ParameterKey': key,
                    'ParameterValue': str(value)
                })
            
            # Add environment parameter
            parameters.append({
                'ParameterKey': 'Environment',
                'ParameterValue': environment
            })
            
            # Create or update stack
            if stack_exists:
                logger.info(f"Updating CloudFormation stack: {stack_name}")
                response = cf_client.update_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
                )
            else:
                logger.info(f"Creating CloudFormation stack: {stack_name}")
                response = cf_client.create_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
                )
            
            stack_id = response['StackId']
            
            # Wait for completion
            waiter = cf_client.get_waiter('stack_create_complete' if not stack_exists else 'stack_update_complete')
            waiter.wait(StackName=stack_name)
            
            # Get stack outputs
            stack_info = cf_client.describe_stacks(StackName=stack_name)
            stack = stack_info['Stacks'][0]
            
            outputs = {}
            for output in stack.get('Outputs', []):
                outputs[output['OutputKey']] = output['OutputValue']
            
            # Get created resources
            resources_response = cf_client.describe_stack_resources(StackName=stack_name)
            resources = [
                {
                    'type': r['ResourceType'],
                    'name': r['LogicalResourceId'],
                    'physical_id': r.get('PhysicalResourceId'),
                    'status': r['ResourceStatus']
                }
                for r in resources_response['StackResources']
            ]
            
            return {
                'success': True,
                'stack_id': stack_id,
                'stack_name': stack_name,
                'environment': environment,
                'resources': resources,
                'outputs': outputs,
                'provisioning_method': 'cloudformation'
            }
            
        except Exception as e:
            logger.error(f"CloudFormation provisioning failed: {e}")
            raise
    
    def _provision_with_cdk(self, environment: str, region: str,
                          config: Dict[str, Any], templates_dir: Path) -> Dict[str, Any]:
        """Provision using AWS CDK."""
        # CDK implementation would go here
        pass
    
    def create_eks_cluster(self, name: str, version: str, node_count: int,
                         node_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create EKS cluster.
        
        Args:
            name: Cluster name
            version: Kubernetes version
            node_count: Number of worker nodes
            node_type: Node instance type
            config: Additional configuration
        
        Returns:
            Dictionary with cluster creation results
        """
        eks_client = self.clients['eks']
        ec2_client = self.clients['ec2']
        
        try:
            # Create EKS cluster
            logger.info(f"Creating EKS cluster: {name}")
            
            cluster_response = eks_client.create_cluster(
                name=name,
                version=version,
                roleArn=config.get('role_arn'),
                resourcesVpcConfig={
                    'subnetIds': config.get('subnet_ids', []),
                    'securityGroupIds': config.get('security_group_ids', []),
                    'endpointPublicAccess': config.get('endpoint_public_access', True),
                    'endpointPrivateAccess': config.get('endpoint_private_access', False)
                },
                logging={
                    'clusterLogging': [
                        {
                            'types': ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                            'enabled': True
                        }
                    ]
                }
            )
            
            cluster_arn = cluster_response['cluster']['arn']
            
            # Wait for cluster to be active
            waiter = eks_client.get_waiter('cluster_active')
            waiter.wait(name=name)
            
            # Create node group
            nodegroup_name = f"{name}-nodegroup"
            
            nodegroup_response = eks_client.create_nodegroup(
                clusterName=name,
                nodegroupName=nodegroup_name,
                scalingConfig={
                    'minSize': node_count,
                    'maxSize': node_count * 2,
                    'desiredSize': node_count
                },
                diskSize=config.get('disk_size', 20),
                subnets=config.get('subnet_ids', []),
                instanceTypes=[node_type],
                amiType='AL2_x86_64',
                nodeRole=config.get('node_role_arn')
            )
            
            # Wait for node group to be active
            nodegroup_waiter = eks_client.get_waiter('nodegroup_active')
            nodegroup_waiter.wait(clusterName=name, nodegroupName=nodegroup_name)
            
            # Get cluster endpoint and certificate
            cluster_info = eks_client.describe_cluster(name=name)
            cluster = cluster_info['cluster']
            
            # Generate kubeconfig
            kubeconfig = self._generate_kubeconfig(cluster)
            
            return {
                'success': True,
                'cluster_name': name,
                'cluster_arn': cluster_arn,
                'endpoint': cluster['endpoint'],
                'version': cluster['version'],
                'status': cluster['status'],
                'node_count': node_count,
                'kubeconfig': kubeconfig
            }
            
        except Exception as e:
            logger.error(f"EKS cluster creation failed: {e}")
            raise
    
    def _generate_kubeconfig(self, cluster_info: Dict[str, Any]) -> str:
        """Generate kubeconfig for EKS cluster."""
        cluster = cluster_info['cluster']
        
        kubeconfig = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'clusters': [
                {
                    'name': cluster['name'],
                    'cluster': {
                        'server': cluster['endpoint'],
                        'certificate-authority-data': cluster['certificateAuthority']['data']
                    }
                }
            ],
            'contexts': [
                {
                    'name': cluster['name'],
                    'context': {
                        'cluster': cluster['name'],
                        'user': f"{cluster['name']}-user"
                    }
                }
            ],
            'current-context': cluster['name'],
            'users': [
                {
                    'name': f"{cluster['name']}-user",
                    'user': {
                        'exec': {
                            'apiVersion': 'client.authentication.k8s.io/v1beta1',
                            'command': 'aws',
                            'args': [
                                'eks',
                                'get-token',
                                '--cluster-name',
                                cluster['name']
                            ]
                        }
                    }
                }
            ]
        }
        
        return yaml.dump(kubeconfig, default_flow_style=False)
    
    def create_rds_instance(self, name: str, engine: str, size: str,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create RDS database instance.
        
        Args:
            name: Database instance name
            engine: Database engine (mysql, postgres, etc.)
            size: Instance size
            config: Additional configuration
        
        Returns:
            Dictionary with RDS creation results
        """
        rds_client = self.clients['rds']
        
        try:
            logger.info(f"Creating RDS instance: {name}")
            
            # Map engine to full engine name
            engine_map = {
                'mysql': 'mysql',
                'postgres': 'postgres',
                'aurora': 'aurora',
                'aurora-mysql': 'aurora-mysql',
                'aurora-postgresql': 'aurora-postgresql'
            }
            
            full_engine = engine_map.get(engine, engine)
            
            # Create DB instance
            response = rds_client.create_db_instance(
                DBInstanceIdentifier=name,
                DBInstanceClass=size,
                Engine=full_engine,
                MasterUsername=config.get('username', 'admin'),
                MasterUserPassword=config.get('password'),
                AllocatedStorage=config.get('storage', 20),
                VpcSecurityGroupIds=config.get('security_group_ids', []),
                DBSubnetGroupName=config.get('subnet_group'),
                MultiAZ=config.get('multi_az', False),
                PubliclyAccessible=config.get('publicly_accessible', False),
                StorageType=config.get('storage_type', 'gp2'),
                BackupRetentionPeriod=config.get('backup_retention', 7),
                Tags=[
                    {'Key': 'Environment', 'Value': config.get('environment', 'dev')},
                    {'Key': 'ManagedBy', 'Value': 'DevOpsAutomator'}
                ]
            )
            
            db_instance = response['DBInstance']
            
            # Wait for instance to be available
            waiter = rds_client.get_waiter('db_instance_available')
            waiter.wait(DBInstanceIdentifier=name)
            
            # Get endpoint
            instance_info = rds_client.describe_db_instances(DBInstanceIdentifier=name)
            instance = instance_info['DBInstances'][0]
            
            return {
                'success': True,
                'instance_id': name,
                'engine': instance['Engine'],
                'endpoint': instance['Endpoint']['Address'],
                'port': instance['Endpoint']['Port'],
                'status': instance['DBInstanceStatus'],
                'arn': instance['DBInstanceArn']
            }
            
        except Exception as e:
            logger.error(f"RDS instance creation failed: {e}")
            raise
    
    def cleanup_resources(self, older_than, dry_run: bool = False) -> Dict[str, Any]:
        """Cleanup old AWS resources."""
        cleanup_results = {
            'cleaned': 0,
            'skipped': 0,
            'errors': []
        }
        
        try:
            # Cleanup old EC2 instances
            cleanup_results['ec2'] = self._cleanup_ec2_instances(older_than, dry_run)
            
            # Cleanup old EBS volumes
            cleanup_results['ebs'] = self._cleanup_ebs_volumes(older_than, dry_run)
            
            # Cleanup old snapshots
            cleanup_results['snapshots'] = self._cleanup_snapshots(older_than, dry_run)
            
            # Cleanup old CloudFormation stacks
            cleanup_results['cloudformation'] = self._cleanup_cloudformation_stacks(older_than, dry_run)
            
            # Sum up results
            for resource_type, result in cleanup_results.items():
                if isinstance(result, dict):
                    cleanup_results['cleaned'] += result.get('cleaned', 0)
                    cleanup_results['skipped'] += result.get('skipped', 0)
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"AWS cleanup failed: {e}")
            cleanup_results['errors'].append(str(e))
            return cleanup_results
    
    def _cleanup_ec2_instances(self, older_than, dry_run: bool) -> Dict[str, Any]:
        """Cleanup old EC2 instances."""
        ec2_client = self.clients['ec2']
        result = {'cleaned': 0, 'skipped': 0}
        
        try:
            # Get all instances
            response = ec2_client.describe_instances(
                Filters=[
                    {'Name': 'tag:ManagedBy', 'Values': ['DevOpsAutomator']},
                    {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}
                ]
            )
            
            instances_to_cleanup = []
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    launch_time = instance['LaunchTime']
                    
                    # Check if instance is older than cutoff
                    if launch_time < older_than:
                        # Check if it's a production instance
                        is_production = False
                        for tag in instance.get('Tags', []):
                            if tag['Key'] == 'Environment' and tag['Value'] == 'prod':
                                is_production = True
                                break
                        
                        if not is_production:
                            instances_to_cleanup.append(instance['InstanceId'])
            
            # Cleanup instances
            for instance_id in instances_to_cleanup:
                if dry_run:
                    logger.info(f"[DRY RUN] Would terminate EC2 instance: {instance_id}")
                    result['cleaned'] += 1
                else:
                    try:
                        ec2_client.terminate_instances(InstanceIds=[instance_id])
                        logger.info(f"Terminated EC2 instance: {instance_id}")
                        result['cleaned'] += 1
                    except Exception as e:
                        logger.error(f"Failed to terminate instance {instance_id}: {e}")
                        result['skipped'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"EC2 cleanup failed: {e}")
            result['skipped'] = len(instances_to_cleanup) if 'instances_to_cleanup' in locals() else 0
            return result
    
    def _extract_resources_from_state(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract resources from Terraform state."""
        resources = []
        
        if 'values' in state and 'root_module' in state['values']:
            root_module = state['values']['root_module']
            
            # Extract resources from root module
            for resource in root_module.get('resources', []):
                resources.append({
                    'type': resource['type'],
                    'name': resource['name'],
                    'address': resource['address'],
                    'values': resource.get('values', {})
                })
            
            # Recursively check child modules
            for child_module in root_module.get('child_modules', []):
                for resource in child_module.get('resources', []):
                    resources.append({
                        'type': resource['type'],
                        'name': resource['name'],
                        'address': resource['address'],
                        'values': resource.get('values', {})
                    })
        
        return resources
    
    def get_status(self) -> List[Dict[str, Any]]:
        """Get status of AWS infrastructure."""
        status_list = []
        
        try:
            # Get CloudFormation stacks status
            cf_client = self.clients['cloudformation']
            stacks = cf_client.describe_stacks()
            
            for stack in stacks['Stacks']:
                tags = {tag['Key']: tag['Value'] for tag in stack.get('Tags', [])}
                
                if tags.get('ManagedBy') == 'DevOpsAutomator':
                    status_list.append({
                        'provider': 'aws',
                        'environment': tags.get('Environment', 'unknown'),
                        'resource_count': len(stack.get('Outputs', [])),
                        'status': stack['StackStatus'],
                        'name': stack['StackName'],
                        'created': stack['CreationTime'].isoformat()
                    })
            
            return status_list
            
        except Exception as e:
            logger.error(f"Failed to get AWS status: {e}")
            return status_list
