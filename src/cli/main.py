"""
DevOps Automator CLI - Main entry point for automation tool.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

from src.core.engine import AutomationEngine
from src.core.config import ConfigManager
from src.utils.logger import setup_logging
from src.plugins.registry import PluginRegistry

# Initialize console for rich output
console = Console()

# Setup logging
logger = setup_logging()

class DevOpsAutomatorCLI:
    """Main CLI application class."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.engine = AutomationEngine(self.config)
        self.plugins = PluginRegistry()
        
    def load_project_config(self, project_path: Path) -> Dict[str, Any]:
        """Load project configuration."""
        config_file = project_path / "devops-config.yaml"
        
        if not config_file.exists():
            console.print(f"[red]Error: Config file not found at {config_file}[/red]")
            sys.exit(1)
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_environment(self) -> bool:
        """Validate that required tools are installed."""
        required_tools = ['docker', 'terraform', 'kubectl', 'aws']
        missing_tools = []
        
        for tool in required_tools:
            if not self._check_tool_installed(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            console.print(f"[yellow]Warning: Missing tools: {', '.join(missing_tools)}[/yellow]")
            console.print("Some features may not work correctly.")
            return False
        
        return True
    
    def _check_tool_installed(self, tool: str) -> bool:
        """Check if a tool is installed and accessible."""
        import shutil
        return shutil.which(tool) is not None

@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.option('--config', '-c', type=click.Path(), help='Path to config file')
@click.pass_context
def cli(ctx, verbose, debug, config):
    """DevOps Automator - Comprehensive DevOps automation toolkit."""
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug
    ctx.obj['config'] = config
    
    # Set log level
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logger.setLevel(log_level)
    
    # Initialize CLI
    ctx.obj['cli'] = DevOpsAutomatorCLI()
    
    # Validate environment
    ctx.obj['cli'].validate_environment()

@cli.group()
def infra():
    """Infrastructure management commands."""
    pass

@infra.command()
@click.option('--provider', '-p', required=True, 
              type=click.Choice(['aws', 'azure', 'gcp', 'multi']),
              help='Cloud provider')
@click.option('--environment', '-e', default='dev',
              type=click.Choice(['dev', 'staging', 'prod']),
              help='Environment to provision')
@click.option('--region', '-r', default='us-east-1',
              help='Region to deploy resources')
@click.option('--dry-run', is_flag=True, help='Show what would be created')
@click.pass_context
def provision(ctx, provider, environment, region, dry_run):
    """Provision cloud infrastructure."""
    cli_app = ctx.obj['cli']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Provisioning infrastructure...", total=None)
        
        try:
            # Load infrastructure configuration
            infra_config = cli_app.config.get_infrastructure_config(
                provider, environment, region
            )
            
            if dry_run:
                console.print("\n[yellow]DRY RUN - No resources will be created[/yellow]")
                console.print(f"Provider: {provider}")
                console.print(f"Environment: {environment}")
                console.print(f"Region: {region}")
                console.print(f"\nResources to be created:")
                
                # Show planned resources
                table = Table(title="Planned Infrastructure")
                table.add_column("Resource Type", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Configuration", style="yellow")
                
                for resource in infra_config.get('resources', []):
                    table.add_row(
                        resource['type'],
                        resource.get('name', 'N/A'),
                        str(resource.get('config', {}))
                    )
                
                console.print(table)
                return
            
            # Execute provisioning
            result = cli_app.engine.provision_infrastructure(
                provider=provider,
                environment=environment,
                region=region,
                config=infra_config
            )
            
            progress.update(task, completed=True)
            
            # Show results
            console.print("\n[green]✓ Infrastructure provisioned successfully![/green]")
            
            # Output summary
            self._display_provisioning_summary(result)
            
        except Exception as e:
            logger.error(f"Provisioning failed: {e}", exc_info=ctx.obj['debug'])
            console.print(f"[red]✗ Provisioning failed: {e}[/red]")
            sys.exit(1)

@infra.command()
@click.argument('resource_type')
@click.option('--name', '-n', help='Resource name')
@click.option('--all', '-a', is_flag=True, help='Destroy all resources')
@click.option('--force', '-f', is_flag=True, help='Force destroy without confirmation')
@click.pass_context
def destroy(ctx, resource_type, name, all, force):
    """Destroy cloud resources."""
    cli_app = ctx.obj['cli']
    
    if not force:
        if all:
            confirmation = click.confirm(
                "Are you sure you want to destroy ALL resources? This cannot be undone!",
                default=False
            )
        else:
            confirmation = click.confirm(
                f"Destroy {resource_type} resource '{name}'?",
                default=False
            )
        
        if not confirmation:
            console.print("[yellow]Operation cancelled[/yellow]")
            return
    
    try:
        if all:
            result = cli_app.engine.destroy_all_resources(resource_type)
        else:
            result = cli_app.engine.destroy_resource(resource_type, name)
        
        console.print(f"[green]✓ Resources destroyed successfully![/green]")
        console.print(f"Destroyed: {result['destroyed_count']} resources")
        
    except Exception as e:
        logger.error(f"Destroy failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Destroy failed: {e}[/red]")
        sys.exit(1)

@cli.group()
def k8s():
    """Kubernetes operations."""
    pass

@k8s.command()
@click.option('--name', '-n', required=True, help='Cluster name')
@click.option('--provider', '-p', default='eks',
              type=click.Choice(['eks', 'aks', 'gke', 'kind']),
              help='Kubernetes provider')
@click.option('--version', '-v', default='1.24',
              help='Kubernetes version')
@click.option('--nodes', default=3, help='Number of worker nodes')
@click.option('--node-type', default='t3.medium',
              help='Node instance type')
@click.pass_context
def create_cluster(ctx, name, provider, version, nodes, node_type):
    """Create a Kubernetes cluster."""
    cli_app = ctx.obj['cli']
    
    console.print(f"[blue]Creating {provider} cluster '{name}'...[/blue]")
    
    try:
        result = cli_app.engine.create_kubernetes_cluster(
            name=name,
            provider=provider,
            version=version,
            node_count=nodes,
            node_type=node_type
        )
        
        console.print(f"[green]✓ Cluster '{name}' created successfully![/green]")
        console.print(f"Endpoint: {result.get('endpoint')}")
        console.print(f"Status: {result.get('status')}")
        
        # Generate kubeconfig
        if result.get('kubeconfig'):
            kubeconfig_path = Path.home() / '.kube' / f'config-{name}'
            with open(kubeconfig_path, 'w') as f:
                f.write(result['kubeconfig'])
            console.print(f"Kubeconfig saved to: {kubeconfig_path}")
        
    except Exception as e:
        logger.error(f"Cluster creation failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Cluster creation failed: {e}[/red]")
        sys.exit(1)

@k8s.command()
@click.option('--name', '-n', required=True, help='Cluster name')
@click.option('--namespace', default='default',
              help='Namespace to deploy to')
@click.option('--file', '-f', type=click.Path(exists=True),
              help='Manifest file or directory')
@click.option('--chart', '-c', help='Helm chart to deploy')
@click.option('--values', type=click.Path(exists=True),
              help='Helm values file')
@click.pass_context
def deploy(ctx, name, namespace, file, chart, values):
    """Deploy application to Kubernetes."""
    cli_app = ctx.obj['cli']
    
    if not file and not chart:
        console.print("[red]Error: Either --file or --chart must be specified[/red]")
        sys.exit(1)
    
    try:
        if file:
            console.print(f"[blue]Deploying from {file}...[/blue]")
            result = cli_app.engine.deploy_kubernetes_manifests(
                cluster_name=name,
                namespace=namespace,
                manifest_path=Path(file)
            )
        else:
            console.print(f"[blue]Deploying Helm chart {chart}...[/blue]")
            result = cli_app.engine.deploy_helm_chart(
                cluster_name=name,
                namespace=namespace,
                chart=chart,
                values_file=Path(values) if values else None
            )
        
        console.print(f"[green]✓ Deployment successful![/green]")
        
        # Show deployment status
        table = Table(title="Deployment Status")
        table.add_column("Resource", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Ready", style="yellow")
        
        for resource in result.get('deployed_resources', []):
            table.add_row(
                resource['name'],
                resource['status'],
                f"{resource['ready']}/{resource['desired']}"
            )
        
        console.print(table)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Deployment failed: {e}[/red]")
        sys.exit(1)

@cli.group()
def pipeline():
    """CI/CD pipeline management."""
    pass

@pipeline.command()
@click.option('--provider', '-p', default='github',
              type=click.Choice(['github', 'gitlab', 'jenkins', 'circleci']),
              help='CI/CD provider')
@click.option('--type', '-t', default='nodejs',
              type=click.Choice(['nodejs', 'python', 'java', 'go', 'react']),
              help='Project type')
@click.option('--output', '-o', default='.github/workflows',
              type=click.Path(),
              help='Output directory for pipeline files')
@click.option('--overwrite', is_flag=True,
              help='Overwrite existing pipeline files')
@click.pass_context
def generate(ctx, provider, type, output, overwrite):
    """Generate CI/CD pipeline configuration."""
    cli_app = ctx.obj['cli']
    
    output_path = Path(output)
    
    try:
        console.print(f"[blue]Generating {provider} pipeline for {type} project...[/blue]")
        
        result = cli_app.engine.generate_pipeline_config(
            provider=provider,
            project_type=type,
            output_dir=output_path,
            overwrite=overwrite
        )
        
        console.print(f"[green]✓ Pipeline configuration generated![/green]")
        console.print(f"Files created in: {output_path}")
        
        # List generated files
        table = Table(title="Generated Pipeline Files")
        table.add_column("File", style="cyan")
        table.add_column("Description", style="green")
        
        for file_info in result.get('generated_files', []):
            table.add_row(
                file_info['name'],
                file_info['description']
            )
        
        console.print(table)
        
        # Show next steps
        console.print("\n[yellow]Next steps:[/yellow]")
        console.print("1. Review the generated pipeline configuration")
        console.print("2. Customize as needed for your project")
        console.print("3. Commit and push to trigger the pipeline")
        
    except Exception as e:
        logger.error(f"Pipeline generation failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Pipeline generation failed: {e}[/red]")
        sys.exit(1)

@pipeline.command()
@click.option('--project', '-p', required=True,
              type=click.Path(exists=True),
              help='Project directory')
@click.option('--target', '-t', default='staging',
              type=click.Choice(['dev', 'staging', 'prod']),
              help='Deployment target')
@click.option('--image-tag', help='Docker image tag (default: git commit SHA)')
@click.pass_context
def deploy(ctx, project, target, image_tag):
    """Deploy project through CI/CD pipeline."""
    cli_app = ctx.obj['cli']
    
    project_path = Path(project)
    
    try:
        console.print(f"[blue]Deploying project from {project_path} to {target}...[/blue]")
        
        result = cli_app.engine.execute_pipeline_deployment(
            project_path=project_path,
            target_environment=target,
            image_tag=image_tag
        )
        
        console.print(f"[green]✓ Deployment pipeline executed![/green]")
        
        # Show pipeline execution details
        console.print(f"\nPipeline ID: {result.get('pipeline_id')}")
        console.print(f"Status: {result.get('status')}")
        console.print(f"Logs: {result.get('logs_url', 'N/A')}")
        
        if result.get('deployment_url'):
            console.print(f"Deployment URL: {result['deployment_url']}")
        
    except Exception as e:
        logger.error(f"Pipeline deployment failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Pipeline deployment failed: {e}[/red]")
        sys.exit(1)

@cli.group()
def monitoring():
    """Monitoring and observability."""
    pass

@monitoring.command()
@click.option('--cluster', '-c', required=True,
              help='Kubernetes cluster name')
@click.option('--namespace', '-n', default='monitoring',
              help='Namespace for monitoring stack')
@click.option('--prometheus', is_flag=True,
              help='Deploy Prometheus')
@click.option('--grafana', is_flag=True,
              help='Deploy Grafana')
@click.option('--alertmanager', is_flag=True,
              help='Deploy AlertManager')
@click.option('--exporters', is_flag=True,
              help='Deploy node exporters')
@click.option('--all', '-a', is_flag=True,
              help='Deploy complete monitoring stack')
@click.pass_context
def setup(ctx, cluster, namespace, prometheus, grafana, 
          alertmanager, exporters, all):
    """Setup monitoring stack in Kubernetes."""
    cli_app = ctx.obj['cli']
    
    # If --all is specified, enable all components
    if all:
        prometheus = grafana = alertmanager = exporters = True
    
    try:
        console.print(f"[blue]Setting up monitoring stack in {cluster}...[/blue]")
        
        result = cli_app.engine.setup_monitoring_stack(
            cluster_name=cluster,
            namespace=namespace,
            deploy_prometheus=prometheus,
            deploy_grafana=grafana,
            deploy_alertmanager=alertmanager,
            deploy_exporters=exporters
        )
        
        console.print(f"[green]✓ Monitoring stack deployed![/green]")
        
        # Show access information
        table = Table(title="Monitoring Stack Access")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Access", style="yellow")
        
        for component in result.get('components', []):
            table.add_row(
                component['name'],
                component['status'],
                component.get('access_url', 'N/A')
            )
        
        console.print(table)
        
        # Show Grafana credentials if available
        if grafana and result.get('grafana_credentials'):
            creds = result['grafana_credentials']
            console.print(f"\n[yellow]Grafana Credentials:[/yellow]")
            console.print(f"URL: {creds.get('url')}")
            console.print(f"Username: {creds.get('username')}")
            console.print(f"Password: {creds.get('password')}")
        
    except Exception as e:
        logger.error(f"Monitoring setup failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Monitoring setup failed: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--format', '-f', default='table',
              type=click.Choice(['table', 'json', 'yaml']),
              help='Output format')
@click.pass_context
def status(ctx, format):
    """Show current status of managed resources."""
    cli_app = ctx.obj['cli']
    
    try:
        console.print("[blue]Fetching resource status...[/blue]")
        
        status_data = cli_app.engine.get_resource_status()
        
        if format == 'json':
            import json
            console.print(json.dumps(status_data, indent=2))
        elif format == 'yaml':
            import yaml
            console.print(yaml.dump(status_data, default_flow_style=False))
        else:
            # Display as table
            self._display_status_table(status_data)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Status check failed: {e}[/red]")
        sys.exit(1)

def _display_status_table(self, status_data: Dict[str, Any]):
    """Display status data in a formatted table."""
    # Infrastructure status
    if status_data.get('infrastructure'):
        infra_table = Table(title="Infrastructure Status")
        infra_table.add_column("Provider", style="cyan")
        infra_table.add_column("Environment", style="green")
        infra_table.add_column("Resources", style="yellow")
        infra_table.add_column("Status", style="magenta")
        
        for infra in status_data['infrastructure']:
            infra_table.add_row(
                infra['provider'],
                infra['environment'],
                str(infra['resource_count']),
                infra['status']
            )
        
        console.print(infra_table)
    
    # Kubernetes clusters
    if status_data.get('kubernetes_clusters'):
        k8s_table = Table(title="Kubernetes Clusters")
        k8s_table.add_column("Name", style="cyan")
        k8s_table.add_column("Provider", style="green")
        k8s_table.add_column("Version", style="yellow")
        k8s_table.add_column("Nodes", style="magenta")
        k8s_table.add_column("Status", style="blue")
        
        for cluster in status_data['kubernetes_clusters']:
            k8s_table.add_row(
                cluster['name'],
                cluster['provider'],
                cluster['version'],
                str(cluster['node_count']),
                cluster['status']
            )
        
        console.print(k8s_table)
    
    # Deployments
    if status_data.get('deployments'):
        deploy_table = Table(title="Deployments")
        deploy_table.add_column("Application", style="cyan")
        deploy_table.add_column("Cluster", style="green")
        deploy_column("Namespace", style="yellow")
        deploy_table.add_column("Status", style="magenta")
        deploy_table.add_column("Age", style="blue")
        
        for deploy in status_data['deployments']:
            deploy_table.add_row(
                deploy['name'],
                deploy['cluster'],
                deploy['namespace'],
                deploy['status'],
                deploy['age']
            )
        
        console.print(deploy_table)

@cli.command()
@click.option('--cleanup-days', default=30,
              help='Cleanup resources older than X days')
@click.option('--dry-run', is_flag=True,
              help='Show what would be cleaned up')
@click.option('--force', '-f', is_flag=True,
              help='Force cleanup without confirmation')
@click.pass_context
def cleanup(ctx, cleanup_days, dry_run, force):
    """Cleanup old/unused resources."""
    cli_app = ctx.obj['cli']
    
    if not force and not dry_run:
        confirmation = click.confirm(
            f"Cleanup resources older than {cleanup_days} days?",
            default=False
        )
        if not confirmation:
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
    
    try:
        console.print(f"[blue]Cleaning up resources older than {cleanup_days} days...[/blue]")
        
        result = cli_app.engine.cleanup_resources(
            older_than_days=cleanup_days,
            dry_run=dry_run
        )
        
        if dry_run:
            console.print("\n[yellow]DRY RUN - No resources will be deleted[/yellow]")
        
        # Show cleanup results
        table = Table(title="Cleanup Results")
        table.add_column("Resource Type", style="cyan")
        table.add_column("Cleaned Up", style="green")
        table.add_column("Skipped", style="yellow")
        
        for resource_type, counts in result.items():
            table.add_row(
                resource_type,
                str(counts.get('cleaned', 0)),
                str(counts.get('skipped', 0))
            )
        
        console.print(table)
        
        # Show estimated savings if available
        if result.get('estimated_savings'):
            savings = result['estimated_savings']
            console.print(f"\n[yellow]Estimated monthly savings: ${savings:.2f}[/yellow]")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=ctx.obj['debug'])
        console.print(f"[red]✗ Cleanup failed: {e}[/red]")
        sys.exit(1)

def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        console.print(f"[red]✗ Unexpected error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
