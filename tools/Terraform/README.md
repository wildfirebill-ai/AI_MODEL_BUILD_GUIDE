# Terraform — Infrastructure as Code

Terraform is an open-source Infrastructure as Code (IaC) tool that lets you define and provision cloud resources using declarative configuration files.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Provider** | Plugin that interacts with cloud APIs (AWS, GCP, Azure) |
| **Resource** | A piece of infrastructure (VM, DB, bucket) |
| **Data Source** | Read-only query of existing infrastructure |
| **Module** | Reusable container of multiple resources |
| **State** | JSON file mapping config to real-world resources |
| **Plan** | Preview of changes before applying |
| **Apply** | Executes the planned changes |

## Basic Configuration

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

resource "aws_s3_bucket" "models" {
  bucket = "ml-models-prod"
  tags = {
    Environment = "prod"
  }
}
```

## Provisioning an ML Training Cluster

```hcl
resource "aws_ecs_cluster" "ml" {
  name = "ml-training"
}

resource "aws_ecs_task_definition" "train" {
  family                   = "train-job"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "4096"
  memory                   = "8192"

  container_definitions = jsonencode([
    {
      name  = "trainer"
      image = "myrepo/trainer:latest"
      environment = [
        { name = "BUCKET", value = aws_s3_bucket.models.id }
      ]
    }
  ])
}
```

## Using Modules

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "ml-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["us-west-2a", "us-west-2b"]
}
```

## Data Sources

```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }
}
```

## State Management

```bash
terraform init                     # Initialize project and providers
terraform plan -out=tfplan         # Preview changes
terraform apply tfplan             # Apply changes
terraform destroy                  # Tear down all resources
```

## Remote State (S3 + DynamoDB)

```hcl
terraform {
  backend "s3" {
    bucket         = "ml-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

## Integration Patterns

- **ML Training/Serving Clusters**: Provision ECS, EKS, GPU instances
- **Data Pipelines**: S3 buckets, Glue jobs, Lambda triggers
- **Networking**: VPC, subnets, security groups for ML services
- **CI/CD Infrastructure**: Build servers, artifact storage

## References

- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)
- [Terraform Registry](https://registry.terraform.io/)
