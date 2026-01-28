"""
Core automation engine orchestrating all DevOps operations.
"""

import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml
import tempfile
import shutil

from src.utils.logger import get_logger
from src.core.config import ConfigManager
from src.plugins.registry import PluginRegistry
from src.utils.templating import TemplateRenderer
from src.utils.executor import CommandExecutor
from src.utils.validators import ResourceValidator

logger = get_logger(__name__)

class AutomationEngine:
    """
    Main automation engine coordinating all DevOps operations.
    Thread-safe and supports concurrent operations.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.plugins = PluginRegistry()
        self.template_renderer = TemplateRenderer()
        self.executor = CommandExecutor()
        self.validator = ResourceValidator()
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Task queue for async operations
        self.task_queue = queue.Queue()
        self.running_tasks = {}
        
        # Initialize plugins
        self._initialize_plugins()
        
        # Event bus for plugin communication
        self.event_handlers = {}
    
    def _initialize_plugins(self):
        """Initialize all registered plugins."""
        plugin_configs = self.config.get('plugins', {})
        
        for plugin_name, plugin_config in plugin_configs.items():
            try:
                self.plugins.register_plugin(plugin_name, plugin_config)
                logger.info(f"Plugin {plugin_name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
    
    def provision_infrastructure(self, provider: str, environment: str,
                               region: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provision cloud infrastructure using Infrastructure as Code.
        
        Args:
            provider: Cloud provider (aws, azure, gcp)
            environment: Environment (dev, staging, prod)
            region: Region to deploy to
            config: Infrastructure configuration
        
        Returns:
            Dictionary with provisioning results
        """
        logger.info(f"Provisioning {provider} infrastructure in {region} for {environment}")
        
        try:
            # Get provider plugin
            provider_plugin = self.plugins.get_plugin(f"{provider}_infra")
            if not provider_plugin:
                raise ValueError(f"Provider {provider} not supported")
            
            # Validate configuration
            self.validator.validate_infrastructure_config(config)
            
            # Generate IaC templates
            templates = self._generate_iac_templates(provider, environment, region, config)
            
            # Create temporary directory for Terraform/CloudFormation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write templates to temp directory
                self._write_templates(templates, temp_path)
                
                # Execute provisioning
                result = provider_plugin.provision(
                    environment=environment,
                    region=region,
                    config=config,
                    templates_dir=temp_path
                )
                
                # Store provisioned resources in state
                self._update_resource_state(provider, environment, result)
                
                logger.info(f"Infrastructure provisioned successfully: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Infrastructure provisioning failed: {e}", exc_info=True)
            raise
    
    def _generate_iac_templates(self, provider: str, environment: str,
                              region: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate Infrastructure as Code templates."""
        templates = {}
        
        # Common template variables
        template_vars = {
            'environment': environment,
            'region': region,
            'project': config.get('project', 'devops-automator'),
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        
        # Generate Terraform templates if provider supports it
        if provider in ['aws', 'azure', 'gcp']:
            tf_main = self.template_renderer.render(
                'terraform/main.tf.j2',
                {**template_vars, 'provider': provider}
            )
            tf_vars = self.template_renderer.render(
                'terraform/variables.tf.j2',
                template_vars
            )
            tf_outputs = self.template_renderer.render(
                'terraform/outputs.tf.j2',
                template_vars
            )
            
            templates.update({
                'main.tf': tf_main,
                'variables.tf': tf_vars,
                'outputs.tf': tf_outputs
            })
        
        # Generate Kubernetes manifests if needed
        if config.get('kubernetes'):
            k8s_manifests = self._generate_k8s_manifests(config['kubernetes'])
            templates.update(k8s_manifests)
        
        return templates
    
    def _write_templates(self, templates: Dict[str, str], output_dir: Path):
        """Write templates to files."""
        for filename, content in templates.items():
            file_path = output_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
    
    def _update_resource_state(self, provider: str, environment: str,
                              result: Dict[str, Any]):
        """Update resource state in persistent storage."""
        state_file = self.config.get_state_file()
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
        else:
            state = {}
        
        # Update state
        if provider not in state:
            state[provider] = {}
        
        state[provider][environment] = {
            'provisioned_at': datetime.now().isoformat(),
            'resources': result.get('resources', []),
            'outputs': result.get('outputs', {}),
            'status': 'provisioned'
        }
        
        # Write updated state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def create_kubernetes_cluster(self, name: str, provider: str,
                                version: str, node_count: int,
                                node_type: str) -> Dict[str, Any]:
        """
        Create Kubernetes cluster on specified provider.
        
        Args:
            name: Cluster name
            provider: Cluster provider (eks, aks, gke, kind)
            version: Kubernetes version
            node_count: Number of worker nodes
            node_type: Node instance type
        
        Returns:
            Dictionary with cluster creation results
        """
        logger.info(f"Creating {provider} cluster '{name}'")
        
        try:
            # Get Kubernetes plugin
            k8s_plugin = self.plugins.get_plugin('kubernetes')
            if not k8s_plugin:
                raise ValueError("Kubernetes plugin not available")
            
            # Provider-specific configuration
            provider_config = self.config.get_cluster_config(provider)
            
            # Create cluster
            result = k8s_plugin.create_cluster(
                name=name,
                provider=provider,
                version=version,
                node_count=node_count,
                node_type=node_type,
                config=provider_config
            )
            
            # Store cluster configuration
            self._store_cluster_config(name, provider, result)
            
            # Generate kubeconfig
            if result.get('kubeconfig'):
                kubeconfig_path = self._save_kubeconfig(name, result['kubeconfig'])
                result['kubeconfig_path'] = str(kubeconfig_path)
            
            logger.info(f"Cluster '{name}' created successfully")
            return result
            
        except Exception as e:
            logger.error(f"Cluster creation failed: {e}", exc_info=True)
            raise
    
    def _store_cluster_config(self, name: str, provider: str,
                            result: Dict[str, Any]):
        """Store cluster configuration."""
        clusters_file = self.config.get_clusters_file()
        
        if clusters_file.exists():
            with open(clusters_file, 'r') as f:
                clusters = yaml.safe_load(f) or {}
        else:
            clusters = {}
        
        clusters[name] = {
            'provider': provider,
            'created_at': datetime.now().isoformat(),
            'endpoint': result.get('endpoint'),
            'version': result.get('version'),
            'node_count': result.get('node_count'),
            'status': result.get('status', 'creating')
        }
        
        with open(clusters_file, 'w') as f:
            yaml.dump(clusters, f, default_flow_style=False)
    
    def _save_kubeconfig(self, name: str, kubeconfig: str) -> Path:
        """Save kubeconfig to file."""
        kubeconfig_dir = Path.home() / '.kube' / 'devops-automator'
        kubeconfig_dir.mkdir(parents=True, exist_ok=True)
        
        kubeconfig_path = kubeconfig_dir / f'config-{name}'
        kubeconfig_path.write_text(kubeconfig)
        
        # Set appropriate permissions
        kubeconfig_path.chmod(0o600)
        
        return kubeconfig_path
    
    def deploy_kubernetes_manifests(self, cluster_name: str, namespace: str,
                                  manifest_path: Path) -> Dict[str, Any]:
        """
        Deploy Kubernetes manifests to cluster.
        
        Args:
            cluster_name: Target cluster name
            namespace: Namespace to deploy to
            manifest_path: Path to manifest files
        
        Returns:
            Dictionary with deployment results
        """
        logger.info(f"Deploying manifests to cluster '{cluster_name}' in namespace '{namespace}'")
        
        try:
            # Get cluster configuration
            cluster_config = self._get_cluster_config(cluster_name)
            if not cluster_config:
                raise ValueError(f"Cluster '{cluster_name}' not found")
            
            # Get Kubernetes plugin
            k8s_plugin = self.plugins.get_plugin('kubernetes')
            
            # Deploy manifests
            result = k8s_plugin.deploy_manifests(
                cluster_name=cluster_name,
                namespace=namespace,
                manifest_path=manifest_path,
                kubeconfig=self._get_cluster_kubeconfig(cluster_name)
            )
            
            # Track deployment
            self._track_deployment(cluster_name, namespace, manifest_path, result)
            
            logger.info(f"Deployment to '{cluster_name}' completed")
            return result
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            raise
    
    def deploy_helm_chart(self, cluster_name: str, namespace: str,
                         chart: str, values_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Deploy Helm chart to Kubernetes cluster.
        
        Args:
            cluster_name: Target cluster name
            namespace: Namespace to deploy to
            chart: Helm chart name or path
            values_file: Optional values file
        
        Returns:
            Dictionary with deployment results
        """
        logger.info(f"Deploying Helm chart '{chart}' to '{cluster_name}'")
        
        try:
            k8s_plugin = self.plugins.get_plugin('kubernetes')
            
            result = k8s_plugin.deploy_helm_chart(
                cluster_name=cluster_name,
                namespace=namespace,
                chart=chart,
                values_file=values_file,
                kubeconfig=self._get_cluster_kubeconfig(cluster_name)
            )
            
            # Track Helm release
            self._track_helm_release(cluster_name, namespace, chart, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Helm deployment failed: {e}", exc_info=True)
            raise
    
    def generate_pipeline_config(self, provider: str, project_type: str,
                               output_dir: Path, overwrite: bool = False) -> Dict[str, Any]:
        """
        Generate CI/CD pipeline configuration.
        
        Args:
            provider: CI/CD provider (github, gitlab, jenkins)
            project_type: Project type (nodejs, python, java, etc.)
            output_dir: Output directory for pipeline files
            overwrite: Overwrite existing files
        
        Returns:
            Dictionary with generation results
        """
        logger.info(f"Generating {provider} pipeline for {project_type} project")
        
        try:
            # Get pipeline plugin
            pipeline_plugin = self.plugins.get_plugin('pipeline')
            if not pipeline_plugin:
                raise ValueError("Pipeline plugin not available")
            
            # Get project configuration
            project_config = self.config.get_project_config(project_type)
            
            # Generate pipeline files
            result = pipeline_plugin.generate_configuration(
                provider=provider,
                project_type=project_type,
                config=project_config,
                output_dir=output_dir,
                overwrite=overwrite
            )
            
            logger.info(f"Pipeline configuration generated: {len(result.get('generated_files', []))} files")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline generation failed: {e}", exc_info=True)
            raise
    
    def execute_pipeline_deployment(self, project_path: Path,
                                  target_environment: str,
                                  image_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute deployment through CI/CD pipeline.
        
        Args:
            project_path: Path to project
            target_environment: Target environment
            image_tag: Docker image tag
        
        Returns:
            Dictionary with pipeline execution results
        """
        logger.info(f"Executing pipeline deployment for {project_path} to {target_environment}")
        
        try:
            # Detect project type
            project_type = self._detect_project_type(project_path)
            
            # Get pipeline configuration
            pipeline_config = self._get_pipeline_config(project_path, project_type)
            
            # Build Docker image if needed
            if pipeline_config.get('docker'):
                image_name = self._build_docker_image(
                    project_path,
                    pipeline_config['docker'],
                    image_tag
                )
                pipeline_config['docker']['image'] = image_name
            
            # Trigger pipeline execution
            pipeline_plugin = self.plugins.get_plugin('pipeline')
            result = pipeline_plugin.execute_deployment(
                project_path=project_path,
                environment=target_environment,
                config=pipeline_config
            )
            
            # Monitor deployment
            if result.get('pipeline_id'):
                monitor_result = self._monitor_deployment(
                    result['pipeline_id'],
                    target_environment
                )
                result.update(monitor_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
    
    def setup_monitoring_stack(self, cluster_name: str, namespace: str,
                             deploy_prometheus: bool, deploy_grafana: bool,
                             deploy_alertmanager: bool, deploy_exporters: bool) -> Dict[str, Any]:
        """
        Setup monitoring stack in Kubernetes cluster.
        
        Args:
            cluster_name: Target cluster name
            namespace: Namespace for monitoring stack
            deploy_prometheus: Whether to deploy Prometheus
            deploy_grafana: Whether to deploy Grafana
            deploy_alertmanager: Whether to deploy AlertManager
            deploy_exporters: Whether to deploy node exporters
        
        Returns:
            Dictionary with monitoring setup results
        """
        logger.info(f"Setting up monitoring stack in cluster '{cluster_name}'")
        
        try:
            # Get monitoring plugin
            monitoring_plugin = self.plugins.get_plugin('monitoring')
            if not monitoring_plugin:
                raise ValueError("Monitoring plugin not available")
            
            # Deploy monitoring stack
            result = monitoring_plugin.setup_stack(
                cluster_name=cluster_name,
                namespace=namespace,
                kubeconfig=self._get_cluster_kubeconfig(cluster_name),
                deploy_prometheus=deploy_prometheus,
                deploy_grafana=deploy_grafana,
                deploy_alertmanager=deploy_alertmanager,
                deploy_exporters=deploy_exporters
            )
            
            # Store monitoring configuration
            self._store_monitoring_config(cluster_name, namespace, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}", exc_info=True)
            raise
    
    def cleanup_resources(self, older_than_days: int,
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Cleanup old or unused resources.
        
        Args:
            older_than_days: Cleanup resources older than X days
            dry_run: Show what would be cleaned up without actually doing it
        
        Returns:
            Dictionary with cleanup results
        """
        logger.info(f"Cleaning up resources older than {older_than_days} days")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cleanup_results = {}
            
            # Cleanup unused Docker images
            docker_plugin = self.plugins.get_plugin('docker')
            if docker_plugin:
                docker_result = docker_plugin.cleanup_images(
                    older_than=cutoff_date,
                    dry_run=dry_run
                )
                cleanup_results['docker_images'] = docker_result
            
            # Cleanup old Kubernetes resources
            k8s_plugin = self.plugins.get_plugin('kubernetes')
            if k8s_plugin:
                k8s_result = k8s_plugin.cleanup_resources(
                    older_than=cutoff_date,
                    dry_run=dry_run
                )
                cleanup_results['kubernetes_resources'] = k8s_result
            
            # Cleanup old infrastructure
            for provider in ['aws', 'azure', 'gcp']:
                infra_plugin = self.plugins.get_plugin(f"{provider}_infra")
                if infra_plugin:
                    infra_result = infra_plugin.cleanup_resources(
                        older_than=cutoff_date,
                        dry_run=dry_run
                    )
                    cleanup_results[f"{provider}_infrastructure"] = infra_result
            
            # Calculate estimated savings
            if not dry_run:
                cleanup_results['estimated_savings'] = self._calculate_savings(cleanup_results)
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)
            raise
    
    def _calculate_savings(self, cleanup_results: Dict[str, Any]) -> float:
        """Calculate estimated monthly savings from cleanup."""
        total_savings = 0.0
        
        # Define cost estimates per resource type
        cost_estimates = {
            'docker_images': 0.01,  # per GB-month
            'kubernetes_resources': {
                'pvc': 0.10,  # per GB-month
                'service': 0.05,  # per month
            },
            'aws_infrastructure': {
                'ec2': 0.0116,  # per hour (t3.micro)
                'ebs': 0.10,  # per GB-month
                'rds': 0.017,  # per hour (db.t3.micro)
            }
        }
        
        # Calculate savings (simplified)
        for resource_type, result in cleanup_results.items():
            if isinstance(result, dict) and 'cleaned' in result:
                cleaned_count = result['cleaned']
                if resource_type in cost_estimates:
                    if isinstance(cost_estimates[resource_type], dict):
                        # Sum up different resource costs
                        for sub_type, count in cleaned_count.items():
                            if sub_type in cost_estimates[resource_type]:
                                total_savings += count * cost_estimates[resource_type][sub_type] * 730  # hours in month
                    else:
                        total_savings += cleaned_count * cost_estimates[resource_type] * 730
        
        return round(total_savings, 2)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get status of all managed resources."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'infrastructure': [],
            'kubernetes_clusters': [],
            'deployments': [],
            'pipelines': []
        }
        
        try:
            # Get infrastructure status
            for provider in ['aws', 'azure', 'gcp']:
                infra_plugin = self.plugins.get_plugin(f"{provider}_infra")
                if infra_plugin:
                    infra_status = infra_plugin.get_status()
                    status['infrastructure'].extend(infra_status)
            
            # Get Kubernetes clusters status
            k8s_plugin = self.plugins.get_plugin('kubernetes')
            if k8s_plugin:
                clusters_status = k8s_plugin.get_clusters_status()
                status['kubernetes_clusters'] = clusters_status
            
            # Get deployments status
            if k8s_plugin:
                deployments_status = k8s_plugin.get_deployments_status()
                status['deployments'] = deployments_status
            
            # Get pipeline status
            pipeline_plugin = self.plugins.get_plugin('pipeline')
            if pipeline_plugin:
                pipeline_status = pipeline_plugin.get_status()
                status['pipelines'] = pipeline_status
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get resource status: {e}")
            return status
    
    def destroy_resource(self, resource_type: str, resource_name: str) -> Dict[str, Any]:
        """Destroy specific resource."""
        # Implementation would handle different resource types
        pass
    
    def destroy_all_resources(self, resource_type: str) -> Dict[str, Any]:
        """Destroy all resources of a type."""
        # Implementation would handle bulk destruction
        pass
    
    # Helper methods
    def _get_cluster_config(self, cluster_name: str) -> Optional[Dict[str, Any]]:
        """Get cluster configuration."""
        clusters_file = self.config.get_clusters_file()
        if clusters_file.exists():
            with open(clusters_file, 'r') as f:
                clusters = yaml.safe_load(f) or {}
                return clusters.get(cluster_name)
        return None
    
    def _get_cluster_kubeconfig(self, cluster_name: str) -> str:
        """Get kubeconfig for cluster."""
        kubeconfig_path = Path.home() / '.kube' / 'devops-automator' / f'config-{cluster_name}'
        if kubeconfig_path.exists():
            return kubeconfig_path.read_text()
        raise ValueError(f"Kubeconfig for cluster '{cluster_name}' not found")
    
    def _detect_project_type(self, project_path: Path) -> str:
        """Detect project type from files."""
        # Check for package.json (Node.js)
        if (project_path / 'package.json').exists():
            return 'nodejs'
        # Check for requirements.txt or setup.py (Python)
        elif (project_path / 'requirements.txt').exists() or (project_path / 'setup.py').exists():
            return 'python'
        # Check for pom.xml (Java)
        elif (project_path / 'pom.xml').exists():
            return 'java'
        # Check for go.mod (Go)
        elif (project_path / 'go.mod').exists():
            return 'go'
        else:
            return 'generic'
    
    def _build_docker_image(self, project_path: Path,
                          docker_config: Dict[str, Any],
                          tag: Optional[str] = None) -> str:
        """Build Docker image for project."""
        docker_plugin = self.plugins.get_plugin('docker')
        if not docker_plugin:
            raise ValueError("Docker plugin not available")
        
        # Generate image name
        image_name = docker_config.get('image_name', f"{project_path.name}:latest")
        if tag:
            image_name = f"{image_name.split(':')[0]}:{tag}"
        
        # Build image
        result = docker_plugin.build_image(
            context_path=project_path,
            dockerfile=docker_config.get('dockerfile', 'Dockerfile'),
            image_name=image_name,
            build_args=docker_config.get('build_args', {})
        )
        
        return image_name
    
    def _track_deployment(self, cluster_name: str, namespace: str,
                         manifest_path: Path, result: Dict[str, Any]):
        """Track deployment in state."""
        # Implementation would store deployment metadata
        pass
    
    def _track_helm_release(self, cluster_name: str, namespace: str,
                           chart: str, result: Dict[str, Any]):
        """Track Helm release in state."""
        # Implementation would store Helm release metadata
        pass
    
    def _get_pipeline_config(self, project_path: Path,
                            project_type: str) -> Dict[str, Any]:
        """Get pipeline configuration for project."""
        # Load project-specific configuration
        config_file = project_path / 'devops-config.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Fall back to default configuration
        return self.config.get_project_config(project_type)
    
    def _monitor_deployment(self, pipeline_id: str,
                           environment: str) -> Dict[str, Any]:
        """Monitor deployment progress."""
        # Implementation would poll pipeline status and deployment health
        return {
            'monitored': True,
            'status': 'in_progress'
        }
    
    def _store_monitoring_config(self, cluster_name: str, namespace: str,
                               result: Dict[str, Any]):
        """Store monitoring configuration."""
        # Implementation would store monitoring stack configuration
        pass
